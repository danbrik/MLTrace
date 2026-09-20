from copy import deepcopy
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from sqlalchemy.orm import Session

from app import models
from app.time_series import data, networks, training_service as service, results
from app.time_series.definitions import defaults, definition
from app.time_series.engine import execute
from app.time_series.schemas import PipelineInput
from test_time_series import client_db, upload


def fixture_data(n=150, test_outlier=20):
    times = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC')
    values = np.sin(np.arange(n) / 10) + 3
    values[110:] = test_outlier
    frame = pd.DataFrame({'time': [t.isoformat() for t in times], 'sensor': values, 'constant': np.where(np.arange(n) < 110, 5, 7)})
    dataset = dict(selected_columns=['time', 'sensor', 'constant'], timestamp_column='time', timestamp_format='ISO8601')
    intervals = [dict(id=str(i), start=times[a].isoformat(), end=times[b].isoformat(), subset=subset, tags=['Normal' if subset != 'test' else 'Anomaly']) for i, (a, b, subset) in enumerate([(0, 69, 'train'), (70, 109, 'validation'), (110, n - 1, 'test')])]
    return frame.to_csv(index=False).encode(), dataset, intervals


def test_train_scaling_and_shared_windows():
    content, dataset, intervals = fixture_data()
    prepared = data.prepare(content, dataset, intervals, None)
    assert prepared.window_length == 36
    assert prepared.summary['sampling_interval_seconds'] == 300
    assert prepared.summary['nominal_history_seconds'] == 10800
    assert prepared.summary['timestamp_span_seconds'] == 10500
    assert prepared.summary['counts']['train'] == dict(rows=70, windows=35, warmup_rows=35, segments=1, gaps=0)
    assert prepared.summary['scaler']['constant_columns'] == ['constant']
    assert prepared.scaled[110, 0] > 1 and prepared.scaled[110, 1] == 2
    assert np.array_equal(prepared.endpoints, np.r_[35:70, 105:110, 145:150])
    changed, _, _ = fixture_data(test_outlier=20000)
    other = data.prepare(changed, dataset, intervals, 36)
    assert prepared.summary['scaler'] == other.summary['scaler']
    assert np.array_equal(prepared.windows(prepared.endpoints[:35]), other.windows(other.endpoints[:35]))
    tagged = deepcopy(intervals)
    tagged[0]['tags'] = ['Anomaly', 'Ignore?', 'Normal']
    assert np.array_equal(data.prepare(content, dataset, tagged, 36).scaled, prepared.scaled)


def test_tag_changes_gap_and_excluded_rows():
    content, dataset, intervals = fixture_data()
    frame = pd.read_csv(io.BytesIO(content))
    times = frame.time.tolist()
    tagged = [dict(intervals[0], end=times[20]), dict(intervals[0], id='extra', start=times[21]), *intervals[1:]]
    assert data.prepare(content, dataset, tagged, 36).summary['counts']['train']['windows'] == 35
    excluded = deepcopy(tagged)
    excluded[1]['start'] = times[22]
    p = data.prepare(content, dataset, excluded, 36)
    assert p.summary['counts']['train']['windows'] == 13
    frame.loc[30:, 'time'] = [(pd.Timestamp(t) + pd.Timedelta(minutes=20)).isoformat() for t in frame.time[30:]]
    # Extend split boundaries after shifting the timestamps.
    shifted = [dict(intervals[0], end=frame.time[69]), dict(intervals[1], start=frame.time[70], end=frame.time[109]), dict(intervals[2], start=frame.time[110], end=frame.time[149])]
    p = data.prepare(frame.to_csv(index=False).encode(), dataset, shifted, 36)
    assert p.summary['detected_gaps'] == 1
    assert p.summary['counts']['train']['windows'] == 5


@pytest.mark.parametrize('invalid', ['nan', 'inf', 'text', '', '1,2,3', '1.608,83', '1,608.83', '1608,83"'])
def test_invalid_sensor_rejected(invalid):
    content, dataset, intervals = fixture_data()
    frame = pd.read_csv(io.BytesIO(content), dtype=str)
    frame.loc[1, 'sensor'] = invalid
    with pytest.raises(ValueError, match='sensor.*numerischer'):
        data.prepare(frame.to_csv(index=False).encode(), dataset, intervals, 36)


@pytest.mark.parametrize('delimiter', [',', ';', '\t', '|'])
def test_decimal_comma_sensor_values_preserve_scaling_windows_and_source(delimiter):
    content, dataset, intervals = fixture_data()
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    frame.loc[0, 'sensor'] = '65.67'
    frame.loc[110, 'sensor'] = '1608.83'
    frame.loc[111, 'sensor'] = '-2.5e1'
    dotted = frame.to_csv(index=False).encode()
    expected = data.prepare(dotted, dataset, intervals, 36)
    # Mix both decimal conventions and whitespace in the same sensor column.
    frame.loc[0, 'sensor'] = ' 65,67 '
    frame.loc[110, 'sensor'] = '1608,83'
    frame.loc[111, 'sensor'] = '-2,5e1'
    comma_csv = frame.to_csv(index=False, sep=delimiter).encode()
    actual = data.prepare(comma_csv, dataset, intervals, 36)
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.scaled, expected.scaled)
    np.testing.assert_array_equal(actual.endpoints, expected.endpoints)
    assert actual.summary['scaler'] == expected.summary['scaler']
    assert actual.values[110, 0] == 1608.83
    assert actual.scaled[110, 0] > 1  # Outlier cannot affect Train-only scaling.
    assert data.read_csv(comma_csv).iloc[0]['sensor'] == ' 65,67 '


def test_duplicates_and_short_validation():
    content, dataset, intervals = fixture_data()
    frame = pd.read_csv(io.BytesIO(content))
    frame.loc[1, 'time'] = frame.time[0]
    with pytest.raises(ValueError, match='Doppelte'):
        data.prepare(frame.to_csv(index=False).encode(), dataset, intervals, 36)
    intervals[1]['end'] = intervals[1]['start']
    assert any('validation' in e for e in data.prepare(content, dataset, intervals, 36).summary['errors'])


def test_scores_earlier_errors_and_vae_state_reset():
    x = torch.zeros(2, 36, 2)
    recon = x.clone(); recon[:, 0] = 6
    for kind in ['usad', 'tcn_ae']:
        error = networks.errors(kind, x, dict(reconstruction=recon, cascade=recon * 2), dict(alpha=.5))
        assert torch.equal(error[:, -1].mean(1), torch.zeros(2))
        assert torch.allclose(error.mean((1, 2)), torch.full((2,), 2.5 if kind == 'usad' else 1.0))
    torch.manual_seed(42)
    vae = networks.build_model('lstm_vae', 36, 2, defaults('lstm_vae', 'architecture')).eval()
    first = vae(x)
    vae(torch.ones_like(x))
    second = vae(x)
    assert torch.equal(first['latent'], second['latent'])
    assert torch.equal(first['latent'], first['z_mean'][:, -1])
    expected = .5 * (np.log(2 * np.pi) + first['variance'].log() + first['reconstruction'].square() / first['variance'])
    assert torch.allclose(networks.errors('lstm_vae', x, first, {}), expected)
    assert defaults('lstm_vae', 'training')['noise_std'] == 0


@pytest.mark.parametrize('length', [36, 37])
def test_tcn_shapes_and_partial_pooling(length):
    cfg = defaults('tcn_ae', 'architecture')
    model = networks.build_model('tcn_ae', length, 2, cfg)
    output = model(torch.randn(2, length, 2))
    assert output['reconstruction'].shape == (2, length, 2)
    assert output['latent'].shape == (2, int(np.ceil(length / 6)) * 8)
    assert networks.representation_metadata('tcn_ae', length, cfg)['pooling_factor'] == 6


@pytest.mark.parametrize('kind', ['usad', 'tcn_ae', 'lstm_vae'])
def test_cpu_training_snapshots_exports_and_replay(client_db, tmp_path, monkeypatch, kind):
    client, engine = client_db
    monkeypatch.setattr(service, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(results, 'artifact_dir', service.artifact_dir)
    content, dataset, intervals = fixture_data()
    if kind == 'usad':
        # Exercise import, preview, enqueue, real training, export and replay
        # with quoted decimal-comma cells, not only the preparation helper.
        frame = pd.read_csv(io.BytesIO(content), dtype=str)
        for column in ['sensor', 'constant']:
            frame[column] = frame[column].str.replace('.', ',', regex=False)
        content = frame.to_csv(index=False).encode()
    dataset_id = upload(client, content, name='Sensors', **dataset).json()['id']
    response = client.post('/api/time-series/splits', json=dict(name='Temporal', dataset_id=dataset_id, tags=['Normal', 'Anomaly'], intervals=intervals))
    assert response.status_code == 201, response.text
    split_id = response.json()['id']
    models_list = client.get('/api/time-series/models').json()
    model = next(m for m in models_list if m['kind'] == kind)
    preview = client.post('/api/time-series/pipelines/preview', json=dict(split_id=split_id, model_id=model['id']))
    assert preview.status_code == 200, preview.text
    assert preview.json()['window_length'] == 36
    config = defaults(kind, 'training') | dict(epochs=2, batch_size=16)
    payload = dict(name=kind, split_id=split_id, model_id=model['id'], window_length=36, training=config)
    response = client.post('/api/time-series/pipelines', json=payload)
    assert response.status_code == 201, response.text
    pipeline_id = response.json()['id']
    with Session(engine) as db:
        queued = service.enqueue(db, pipeline_id, wake_scheduler=False)
    for resource, record_id in [('models', model['id']), ('datasets', dataset_id), ('splits', split_id), ('pipelines', pipeline_id)]:
        assert client.delete(f'/api/time-series/{resource}/{record_id}').status_code == 409
    run_id = queued['id']
    metrics = []
    manifest = execute(service.artifact_dir(run_id), queued['snapshot'], lambda **v: metrics.append(v) if 'epoch' in v else None)
    assert len(metrics) == 2
    assert manifest['checkpoint']['epoch'] in (1, 2)
    assert manifest['checkpoint']['selection'] == ('validation_reconstruction_score' if kind == 'usad' else 'validation_loss')
    assert sum(chunk['count'] for chunk in manifest['chunks']) == 45
    with Session(engine) as db:
        row = db.get(models.TimeSeriesRun, run_id)
        row.status = 'finished'; row.result = manifest; row.checkpoint = manifest['checkpoint']; db.commit()
        rows = list(results.result_rows(row, include_latent=True, scaled=True))
        assert len(rows) == 45
        replay = results.replay(row, rows[-1]['endpoint_index'])
        np.testing.assert_allclose(replay['outputs']['latent'], rows[-1]['z_sensor'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(replay['outputs']['reconstruction'][-1][0], rows[-1]['reconstruction'], rtol=1e-5, atol=1e-6)
        assert rows[-1]['timestamp'] == rows[-1]['window_end']
        assert rows[-1]['tags'] == ['Anomaly']
        with zipfile.ZipFile(io.BytesIO(results.latent_export(row))) as archive:
            assert 'metadata.json' in archive.namelist()
            exported = pd.read_csv(archive.open('endpoints.csv'))
            assert exported.timestamp.tolist() == [r['timestamp'] for r in rows]
        csv = ''.join(results.export_csv(row))
        assert len(pd.read_csv(io.StringIO(csv))) == 45
        snapshot = deepcopy(row.snapshot)
        split = db.get(models.TimeSeriesSplit, split_id); split.name = 'Changed'; db.commit()
        assert row.snapshot == snapshot
    response = client.get(f'/api/time-series/runs/{run_id}/series?subset=test&limit=2')
    assert response.status_code == 200, response.text
    assert response.json()['total'] == 5
    assert len(response.json()['rows']) == 2
    assert client.delete(f'/api/time-series/runs/{run_id}').status_code == 204


def test_provenance():
    for kind in ['usad', 'tcn_ae', 'lstm_vae']:
        assert definition(kind)['input']['window_length']['origin'] == 'MLTrace-Standard'
    assert definition('tcn_ae')['architecture']['pooling_factor']['origin'] == 'MLTrace-Standard'
    assert definition('lstm_vae')['architecture']['hidden_size']['origin'] == 'MLTrace-Standard'


@pytest.mark.parametrize('with_validation', [True, False])
def test_checkpoint_ties_final_epoch_and_test_independence(tmp_path, with_validation):
    content, dataset, intervals = fixture_data()
    if not with_validation:
        intervals = [i for i in intervals if i['subset'] != 'validation']
    prepared = data.prepare(content, dataset, intervals, 36)
    snapshots = []
    for copy_index, extreme in enumerate([20, 20000]):
        folder = tmp_path / str(copy_index); folder.mkdir()
        values = prepared.scaled.copy(); values[prepared.groups == 2] = extreme
        service.write_npz(folder / 'input.npz', scaled=values, groups=prepared.groups, endpoints=prepared.endpoints)
        # A tiny learning rate that cannot change float32 weights gives an exact
        # validation tie; selection must retain epoch one, independently of test.
        snapshot = dict(model=dict(kind='usad', config=defaults('usad', 'architecture'), version='1'),
                        training=defaults('usad', 'training') | dict(epochs=3, lr=1e-30, batch_size=16),
                        window_length=36, preview=prepared.summary,
                        checkpoint_metric=dict(selection='validation_reconstruction_score' if with_validation else 'final_epoch'))
        manifest = execute(folder, snapshot)
        snapshots.append(manifest['checkpoint'])
    assert snapshots[0]['sha256'] == snapshots[1]['sha256']
    assert snapshots[0]['metric_value'] == snapshots[1]['metric_value']
    assert snapshots[0]['epoch'] == (1 if with_validation else 3)


def test_tcn_pooling_validation_never_changes_length(client_db):
    client, _ = client_db
    content, dataset, intervals = fixture_data()
    dataset_id = upload(client, content, **dataset).json()['id']
    split_id = client.post('/api/time-series/splits', json=dict(name='Temporal', dataset_id=dataset_id, tags=['Normal', 'Anomaly'], intervals=intervals)).json()['id']
    model = client.post('/api/time-series/models', json=dict(name='Author pooling', kind='tcn_ae', config={'pooling_factor': 42})).json()
    response = client.post('/api/time-series/pipelines/preview', json=dict(split_id=split_id, model_id=model['id'], window_length=36))
    assert response.status_code == 422
    assert 'L bleibt unverändert' in response.text


def test_nanosecond_endpoint_precision():
    content, dataset, intervals = fixture_data()
    frame = pd.read_csv(io.BytesIO(content))
    frame['time'] = [(pd.Timestamp(t) + pd.Timedelta(123, unit='ns')).isoformat() for t in frame.time]
    for interval in intervals:
        for key in ['start', 'end']:
            interval[key] = (pd.Timestamp(interval[key]) + pd.Timedelta(123, unit='ns')).isoformat()
    p = data.prepare(frame.to_csv(index=False).encode(), dataset, intervals, 36)
    assert data.timestamp(p.timestamps[p.endpoints[0]]).endswith('.000000123Z')


def test_model_variant_crud_and_deleted_presets_stay_deleted(client_db):
    client, _ = client_db
    presets = client.get('/api/time-series/models').json()
    preset = presets[0]
    created = client.post('/api/time-series/models', json=dict(name='Custom encoder', kind=preset['kind'], config={'latent_dim': 12}))
    assert created.status_code == 201
    row = created.json()
    assert row['provenance']['architecture']['latent_dim']['overridden'] is True
    assert row['provenance']['architecture']['latent_dim']['origin'] == 'Paper'
    assert client.get(f"/api/time-series/models/{row['id']}").json()['config']['latent_dim'] == 12
    updated = client.put(f"/api/time-series/models/{row['id']}", json=dict(name='Renamed', kind=row['kind'], config=row['config']))
    assert updated.json()['name'] == 'Renamed'
    assert client.delete(f"/api/time-series/models/{row['id']}").status_code == 204
    assert client.delete(f"/api/time-series/models/{preset['id']}").status_code == 204
    assert preset['id'] not in [r['id'] for r in client.get('/api/time-series/models').json()]
    assert client.get(f"/api/time-series/models/{preset['id']}").status_code == 404
