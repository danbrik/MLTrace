"""Label import must partition observations without exposing labels to models."""
import json

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TimeSeriesDataset, TimeSeriesSplit
from app.time_series import service, data
from test_time_series import client_db, upload


def labeled_csv(labels=None, order=None, precise=False):
    labels = labels or ['normal', 'normal', 'before_anomaly', 'anomaly', 'anomaly', 'cooldown', 'normal', 'normal']
    times = pd.date_range('2026-01-21T22:00:00Z', periods=len(labels), freq='ns' if precise else 'min')
    frame = pd.DataFrame({'time': [service.iso(t) for t in times], 'sensor': range(len(labels)), ' Label ': labels})
    if order is not None:
        frame = frame.iloc[order]
    return frame.to_csv(index=False).encode()


def imported(client, content, **kwargs):
    return upload(client, content, selected_columns=['time', 'sensor'], label_column=' Label ', auto_split=True, **kwargs)


def test_preview_import_roundtrip_and_feature_exclusion(client_db):
    client, engine = client_db
    content = labeled_csv([' NORMAL ', 'normal', 'before_anomaly', 'ANOMALY', 'anomaly', 'cooldown', 'normal', 'normal'], [7, 2, 0, 5, 4, 1, 3, 6], precise=True)
    initial = client.post('/api/time-series/preview', files={'file': ('x.csv', content)}).json()
    assert initial['detected_label_column'] == ' Label '
    response = client.post('/api/time-series/preview', files={'file': ('x.csv', content)}, data={'timestamp_column': 'time', 'label_column': ' Label '})
    assert response.status_code == 200, response.text
    preview = response.json()['label_split']
    assert preview['counts'] == {'train': {'rows': 4, 'intervals': 2}, 'test': {'rows': 4, 'intervals': 3}, 'validation': {'rows': 0, 'intervals': 0}}
    assert [i['tags'] for i in preview['intervals']] == [[], ['before_anomaly'], ['anomaly'], ['cooldown'], []]
    result = imported(client, content, split_name='Mein Label-Split')
    assert result.status_code == 201, result.text
    dataset = result.json()
    assert dataset['split_count'] == 1
    assert dataset['label_column'] == ' Label '
    saved = client.get('/api/time-series/splits').json()[0]
    assert saved['name'] == 'Mein Label-Split'
    assert saved['intervals'] == preview['intervals']
    assert saved['tags'] == preview['tags']
    assert saved['intervals'][1]['start'] == saved['intervals'][1]['end'] == '2026-01-21T22:00:00.000000002Z'
    with Session(engine) as db:
        assert db.get(TimeSeriesDataset, dataset['id']).source_csv == content
    update = client.put(f"/api/time-series/datasets/{dataset['id']}", json={'name': 'Renamed', 'selected_columns': ['time', 'sensor', ' Label ']})
    assert update.status_code == 422
    assert 'Annotation' in update.json()['detail']
    assert client.get(f"/api/time-series/datasets/{dataset['id']}").json()['selected_columns'] == ['time', 'sensor']
    prepared = data.prepare(content, dataset, saved['intervals'], 2)
    assert prepared.columns == ['sensor']
    assert prepared.summary['counts']['train']['windows'] == 2
    # Test windows cross changes of annotation, but never a Train/Test boundary.
    assert prepared.summary['counts']['test']['windows'] == 3
    assert prepared.endpoints.tolist() == [1, 3, 4, 5, 7]
    with pytest.raises(ValueError, match='Annotation'):
        data.prepare(content, dict(dataset, selected_columns=['time', 'sensor', ' Label ']), saved['intervals'], 2)
    saved['intervals'][1]['tags'] = ['cooldown']
    edited = client.put(f"/api/time-series/splits/{saved['id']}", json={k: saved[k] for k in ['name', 'dataset_id', 'tags', 'intervals']})
    assert edited.status_code == 200
    assert client.get(f"/api/time-series/splits/{saved['id']}").json()['intervals'][1]['tags'] == ['cooldown']


@pytest.mark.parametrize('label', ['', 'broken', 'puffer'])
def test_invalid_labels_fail_preview_and_import_but_plain_import_works(client_db, label):
    client, _ = client_db
    content = labeled_csv(['normal', label, 'anomaly'])
    response = client.post('/api/time-series/preview', files={'file': ('x.csv', content)}, data={'timestamp_column': 'time', 'label_column': ' Label '})
    assert response.status_code == 422
    assert 'CSV-Zeilen 3:' in response.json()['detail']
    assert imported(client, content).status_code == 422
    assert client.get('/api/time-series/datasets').json() == []
    assert client.get('/api/time-series/splits').json() == []
    plain = upload(client, content, selected_columns=['time', 'sensor'], label_column=' Label ')
    assert plain.status_code == 201
    assert plain.json()['split_count'] == 0


def test_duplicate_timestamps_rejected_without_partial_import(client_db):
    client, _ = client_db
    content = b'time,sensor, Label \n2026-01-01T00:00:00Z,1,normal\n2026-01-01T00:00:00Z,2,anomaly\n'
    response = imported(client, content)
    assert response.status_code == 422
    assert 'Doppelte Zeitstempel' in response.json()['detail']
    assert client.get('/api/time-series/datasets').json() == []


def test_transaction_rolls_back_if_split_persistence_fails(client_db, monkeypatch):
    client, engine = client_db
    original = service.save_split
    def fail_after_flush(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError('Simulierter Fehler nach Split-Speicherung')
    monkeypatch.setattr(service, 'save_split', fail_after_flush)
    assert imported(client, labeled_csv()).status_code == 422
    with Session(engine) as db:
        assert db.scalars(select(TimeSeriesDataset)).all() == []
        assert db.scalars(select(TimeSeriesSplit)).all() == []


def test_default_name_and_normal_only_import(client_db):
    client, _ = client_db
    response = imported(client, labeled_csv(['normal', 'normal']), name='x' * 255)
    assert response.status_code == 201
    saved = client.get('/api/time-series/splits').json()[0]
    assert len(saved['name']) <= 255
    assert saved['name'].endswith(' – Label-Split')
    assert saved['tags'] == []
    assert len(saved['intervals']) == 1


def test_additive_migration_preserves_existing_data(tmp_path):
    from pathlib import Path
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect, text
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / 'alembic.ini'))
    config.set_main_option('script_location', str(root / 'backend/alembic'))
    url = f'sqlite:///{tmp_path / "migration.db"}'
    config.set_main_option('sqlalchemy.url', url)
    command.upgrade(config, '0059_time_series_training')
    engine = create_engine(url)
    with engine.begin() as db:
        db.execute(text("INSERT INTO time_series_datasets (id,name,filename,source_csv,columns,selected_columns,timestamp_column,timestamp_format,row_count,start,end) VALUES (1,'Existing','old.csv',X'61','[\"time\",\"sensor\"]','[\"time\",\"sensor\"]','time','ISO8601',1,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')"))
    command.upgrade(config, 'head')
    command.upgrade(config, 'head')
    with engine.connect() as db:
        assert db.execute(text('SELECT name,label_column FROM time_series_datasets')).one() == ('Existing', None)
    command.downgrade(config, '0059_time_series_training')
    assert 'label_column' not in {column['name'] for column in inspect(engine).get_columns('time_series_datasets')}
    command.upgrade(config, 'head')
    engine.dispose()
