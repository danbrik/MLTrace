import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from app.mean_variance import engine, service
from app.mean_variance.schemas import VarianceConfig, VariancePair, HeatmapScale
from tests.test_reference_image import source
from tests.test_mean_variance import comparison


def paired(cfg, count=1):
    period = lambda value: value.model_dump(include={'start', 'end'})
    return VarianceConfig(training_dataset_id=cfg.training_dataset_id, preprocessing_pipeline_id=cfg.preprocessing_pipeline_id,
                          pairs=[VariancePair(normal=period(cfg.reference), anomaly=period(cfg.anomaly)) for _ in range(count)])


def test_contract_and_shared_selection(comparison, monkeypatch):
    from app import image_selection
    db, old, _ = comparison
    cfg = paired(old, 6)
    calls = []
    original = image_selection.enumerate_training_dataset_image_records
    def enumerate_once(dataset):
        calls.append(dataset.id)
        return original(dataset)
    monkeypatch.setattr(image_selection, 'enumerate_training_dataset_image_records', enumerate_once)
    cfg.sampling_rate = 2
    preview = service.preview(db, cfg)
    assert len(calls) == 1 and len(preview.pairs) == 6
    assert all(p.reference.selected == 2 and p.anomaly.selected == 4 for p in preview.pairs)
    for changes in [{'pairs': []}, {'pairs': cfg.pairs * 2}, {'pairs': [{'normal': cfg.pairs[0].normal}]}, {'sampling_rate': 0}, {'sampling_rate': 1.5}, {'mean_scale': {}}, {'sampling_rate': True}]:
        with pytest.raises(ValidationError):
            VarianceConfig.model_validate({**cfg.model_dump(), **changes})
    cfg.sampling_rate = 100
    assert len(service.preview(db, cfg).errors) == 12
    with pytest.raises(ValueError, match='u1'):
        service.enqueue(db, cfg, wake_scheduler=False)


@pytest.mark.parametrize('count', [1, 6])
def test_pairs_run_export_reopen_and_snapshot(comparison, count):
    db, old, pipeline = comparison
    cfg = paired(old, count)
    if count > 1:
        cfg.pairs[1] = VariancePair(normal=cfg.pairs[0].anomaly, anomaly=cfg.pairs[0].normal)
        cfg.pairs[2].anomaly = cfg.pairs[2].normal.model_copy()
    run = service.enqueue(db, cfg, wake_scheduler=False)
    pipeline.graph = {'nodes': [], 'edges': []}; db.commit()
    service.run_scheduled(run.id); db.expire_all()
    saved = service.get_run(db, run.id)
    assert saved.status == 'finished', saved.error_message
    assert saved.config.version == 2 and saved.config == cfg
    result = service.results(db, run.id)
    assert result['version'] == 2 and len(result['pairs']) == count
    assert result['pairs'][0]['counts'] == {'normal': 4, 'anomaly': 8}
    assert result['pairs'][0]['maps']['normal']['minimum'] == pytest.approx(50000)
    assert result['pairs'][0]['maps']['anomaly']['minimum'] == pytest.approx(210000)
    assert result['pairs'][0]['maps']['difference']['minimum'] == pytest.approx(160000)
    if count > 1:
        assert result['pairs'][1]['maps']['difference']['maximum'] == pytest.approx(-160000)
        assert result['pairs'][2]['maps']['difference']['all_zero']
    assert result['variance_scale_limit'] == 210000 and result['difference_scale_limit'] == 160000
    path = service.artifact_path(db, run.id, 'variance_comparison.png')
    with Image.open(path) as image:
        metadata = json.loads(image.info['Description'])
        assert metadata['unit'] == 'gray value²'
        assert metadata['origin'] == 'upper' and metadata['interpolation'] == 'nearest'
        assert len(metadata['pairs']) == count and metadata['dataset_name'] == run.training_dataset_name
        assert metadata['pipeline_name'] == 'Original'
        assert metadata['pairs'][0]['periods'] == cfg.pairs[0].model_dump(mode='json')
    assert service.artifact_path(db, run.id, 'mean_difference.png') is None
    assert sorted(p.name for p in service.artifact_dir(run.id).iterdir()) == ['manifest.json', 'results.json', 'variance_comparison.png']
    assert saved.processed_images == saved.total_images == result['total_images']


def test_pairs_single_points_and_zero(comparison):
    db, old, _ = comparison
    cfg = paired(old)
    for period in (cfg.pairs[0].normal, cfg.pairs[0].anomaly):
        period.start = period.end = old.reference.start
    run = service.enqueue(db, cfg, wake_scheduler=False)
    service.run_scheduled(run.id); db.expire_all()
    result = service.results(db, run.id)
    assert result['variance_scale_limit'] == result['difference_scale_limit'] == 0
    assert all(v['all_zero'] for v in result['pairs'][0]['maps'].values())
    assert sum('Nur ein Bild' in w for w in result['warnings']) == 2


@pytest.mark.parametrize('failure', ['changed', 'missing', 'size', 'nonfinite', 'export', 'abort'])
def test_pairs_failure_and_abort_cleanup(comparison, monkeypatch, failure):
    db, old, _ = comparison
    cfg = paired(old, 2)
    run = service.enqueue(db, cfg, wake_scheduler=False)
    manifest = json.loads((service.artifact_dir(run.id) / 'manifest.json').read_text())
    path = Path(manifest['samples']['1_anomaly'][1]['file_path'])
    if failure == 'changed':
        path.write_bytes(b'changed')
    elif failure == 'missing':
        path.unlink()
    elif failure in {'size', 'nonfinite'}:
        original = service.read_frozen_image
        def broken(sample, pipeline):
            if sample['file_path'] == str(path):
                return np.full((3, 2), np.nan if failure == 'nonfinite' else 0)
            return original(sample, pipeline)
        monkeypatch.setattr(service, 'read_frozen_image', broken)
    elif failure == 'export':
        def broken(*args, **kwargs):
            raise OSError('PNG export failed')
        monkeypatch.setattr(service, 'render_comparison', broken)
    event = threading.Event()
    if failure == 'abort':
        original = service.render_comparison
        def abort(*args, **kwargs):
            event.set()
            return original(*args, **kwargs)
        monkeypatch.setattr(service, 'render_comparison', abort)
    service.run_scheduled(run.id, event); db.expire_all()
    assert service.get_run(db, run.id).status == ('aborted' if failure == 'abort' else 'failed')
    assert service.results(db, run.id) is None
    assert service.artifact_path(db, run.id, 'variance_comparison.png') is None
    assert not (service.artifact_dir(run.id) / 'exporting').exists()


def test_renderer_global_scales_orientation_and_labels(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    cfg = VarianceConfig(training_dataset_id=1, preprocessing_pipeline_id=1,
                         pairs=[{'normal': {'start': '2026-01-01', 'end': '2026-01-01'}, 'anomaly': {'start': '2026-01-01', 'end': '2026-01-01'}}])
    pairs = []
    for index, factor in enumerate([1, 5]):
        maps = {}
        for role, data in [('normal', [[0, 2], [3, 4]]), ('anomaly', [[2, 0], [3, 8]]), ('difference', [[2, -2], [0, 4]])]:
            values = np.array(data, dtype=float) * factor
            path = tmp_path / f'{index}_{role}.npy'; np.save(path, values)
            maps[role] = {**engine.map_statistics(values), 'path': str(path)}
        pairs.append({'label': f'u{index + 1}', 'periods': cfg.pairs[0].model_dump(mode='json'), 'counts': {'normal': 2, 'anomaly': 2}, 'maps': maps})
    original = Figure.savefig
    def inspect(fig, *args, **kwargs):
        plots = fig.axes[:6]
        assert [ax.get_title() for ax in plots[:3]] == ['Normalzustand', 'Unruhe', 'Differenz']
        assert [text.get_text() for text in fig.texts] == ['u1', 'u2', 'y (Pixel)', 'x (Pixel)']
        for i, ax in enumerate(plots):
            image = ax.images[0]
            assert image.origin == 'upper' and image.get_interpolation() == 'nearest'
            assert ax.get_aspect() == 1
            assert image.norm.vmax == (20 if i % 3 == 2 else 40)
            assert image.norm.vmin == (-20 if i % 3 == 2 else 0)
            assert bool(ax.get_xticklabels()) == (i >= 3)
            assert bool(ax.get_yticklabels()) == (i % 3 == 0)
        assert fig.axes[6].get_xlabel() == 'Variance (gray value²)'
        assert fig.axes[7].get_xlabel() == 'Variance difference (gray value²)'
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', inspect)
    result = engine.render_comparison(pairs, cfg, tmp_path / 'comparison.png', 'Dataset', 'Pipeline', (2, 2))
    assert result['variance_scale_limit'] == 40
    for signed in [False, True]:
        cmap, norm = engine.comparison_style(2, signed)
        np.testing.assert_array_equal(cmap(norm([4, -4])), cmap(norm([2, -2])))
        np.testing.assert_array_equal(cmap(norm(0)), [1, 1, 1, 1])


def test_v2_api_download_and_explicit_format(comparison, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.database import get_db
    from app.mean_variance.api import router
    db, old, _ = comparison
    cfg = paired(old)
    app = FastAPI(); app.include_router(router); app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(service.scheduler, 'wake', lambda: None)
    client = TestClient(app); base = '/api/mean-variance-analysis'
    payload = cfg.model_dump(mode='json')
    assert client.post(base + '/preview', json=payload).json()['pairs'][0]['reference']['selected'] == 4
    response = client.post(base + '/runs', json=payload)
    assert response.status_code == 200, response.text
    run_id = response.json()['id']
    url = f'{base}/runs/{run_id}'
    assert client.get(url + '/artifacts/variance_comparison.png').status_code == 404
    service.run_scheduled(run_id); db.expire_all()
    assert client.get(url).json()['config'] == payload
    download = client.get(url + '/artifacts/variance_comparison.png?download=true')
    assert download.status_code == 200 and download.headers['content-type'] == 'image/png'
    assert download.headers['content-disposition'].startswith('attachment')
    assert client.get(url + '/artifacts/0_normal.npy').status_code == 404
    assert client.delete(url).status_code == 204
    assert not service.artifact_dir(run_id).exists()


def test_manual_limits_change_only_colors_and_shape_is_global(comparison, monkeypatch):
    db, old, _ = comparison
    cfg = paired(old, 2)
    cfg.variance_scale = HeatmapScale(mode='manual', limit=1)
    cfg.difference_scale = HeatmapScale(mode='manual', limit=2)
    run = service.enqueue(db, cfg, wake_scheduler=False)
    service.run_scheduled(run.id); db.expire_all()
    result = service.results(db, run.id)
    assert result['variance_scale_limit'] == 1 and result['difference_scale_limit'] == 2
    assert result['pairs'][0]['maps']['difference']['minimum'] == pytest.approx(160000)
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    original = service.read_frozen_image
    calls = 0
    def other_shape_in_second_pair(sample, pipeline):
        nonlocal calls
        calls += 1
        return np.zeros((3, 3)) if calls == 13 else original(sample, pipeline)
    monkeypatch.setattr(service, 'read_frozen_image', other_shape_in_second_pair)
    service.run_scheduled(queued.id); db.expire_all()
    assert service.get_run(db, queued.id).status == 'failed'
    assert service.results(db, queued.id) is None
