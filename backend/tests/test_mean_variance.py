from datetime import datetime
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from app import models
from app.mean_variance import engine, service
from app.mean_variance.schemas import HeatmapScale, MeanVarianceConfig
from tests.test_reference_image import source, interval, records


def moments(images):
    accumulator = engine.OnlineMoments()
    for image in images:
        accumulator.add(image)
    return accumulator.finish()


def test_known_population_moments_and_signed_differences():
    normal = moments([np.array([[0, 0], [2, 8]], dtype=np.uint16), np.array([[2, 2], [4, 10]], dtype=np.uint16)])
    anomaly = moments([np.array([[2, 0], [3, 9]], dtype=np.uint16), np.array([[6, 0], [3, 9]], dtype=np.uint16)])
    np.testing.assert_array_equal(normal[0], [[1, 1], [3, 9]])
    np.testing.assert_array_equal(normal[1], np.ones((2, 2)))
    mean, variance = engine.difference_maps(normal, anomaly)
    np.testing.assert_array_equal(mean, [[3, 1], [0, 0]])
    np.testing.assert_array_equal(variance, [[3, -1], [-1, -1]])
    for value in engine.difference_maps(normal, normal):
        assert np.count_nonzero(value) == 0


def test_fractional_16bit_single_image_and_stability():
    mean, var = moments([np.array([[65534]], dtype=np.uint16), np.array([[65535]], dtype=np.uint16)])
    assert mean.dtype == np.float64 and mean.item() == 65534.5
    assert var.item() == .25
    _, var = moments([np.array([[65535]], dtype=np.uint16)])
    assert var.item() == 0
    images = [np.full((2, 2), 1e12 + i) for i in range(10)]
    mean, var = moments(iter(images))
    np.testing.assert_allclose(mean, 1e12 + 4.5)
    np.testing.assert_allclose(var, 8.25)


@pytest.mark.parametrize('images', [[], [np.array([[np.nan]])], [np.array([[np.inf]])], [np.zeros((2, 2)), np.zeros((3, 2))], [np.zeros((2, 2, 3))], [np.array([[1e308]]), np.array([[-1e308]])]])
def test_invalid_images(images):
    with pytest.raises(ValueError):
        moments(images)


def test_scales_color_direction_zero_and_saturation():
    values = np.array([[-4., 0, 4.]])
    cmap, norm, metadata = engine.heatmap_style(values, HeatmapScale(), True)
    assert metadata['scale_limit'] == 4
    rgb = cmap(norm(values))[0]
    assert rgb[0, 2] > rgb[0, 0] and rgb[2, 0] > rgb[2, 2]
    np.testing.assert_array_equal(rgb[1], [1, 1, 1, 1])
    cmap, norm, metadata = engine.heatmap_style(values, HeatmapScale(mode='manual', limit=2), True)
    np.testing.assert_array_equal(cmap(norm([-4, 4])), cmap(norm([-2, 2])))
    assert metadata['maximum_absolute'] == 4 and metadata['scale_limit'] == 2
    for signed in [True, False]:
        cmap, norm, metadata = engine.heatmap_style(np.zeros((2, 2)), HeatmapScale(), signed)
        assert metadata['scale_limit'] == 0 and metadata['all_zero']
        assert np.unique(cmap(norm(np.zeros((2, 2)))).reshape(-1, 4), axis=0).shape[0] == 1


@pytest.fixture
def comparison(source, monkeypatch, tmp_path):
    db, config, pipeline = source
    values = config.model_dump(mode='json', include={'training_dataset_id', 'preprocessing_pipeline_id', 'reference', 'anomaly'})
    cfg = MeanVarianceConfig.model_validate(values)
    monkeypatch.setattr(service, 'artifact_dir', lambda id: tmp_path / 'comparisons' / str(id))
    return db, cfg, pipeline


def test_selection_saved_sampling_random_bounds(comparison):
    from app.reference_image.engine import select_records
    db, cfg, _ = comparison
    assert service.preview(db, cfg).reference.selected == 4  # Dataset stride 2.
    cfg.reference.sampling_rate = 2
    assert service.preview(db, cfg).reference.selected == 2
    cfg.reference = cfg.reference.model_copy(update={'sampling_rate': 15, 'start': datetime(2026, 1, 1), 'end': datetime(2026, 1, 1, 0, 0, 45)})
    selection, preview = select_records(records(47) + list(reversed(records(47))), cfg)
    assert [item['file_path'] for item in selection['reference']] == ['14', '29', '44']
    assert preview.reference.remainder == 1
    cfg.reference.mode = 'random'; cfg.reference.count = 5
    a, _ = select_records(records(47), cfg)
    b, _ = select_records(list(reversed(records(47))), cfg)
    assert a == b and len({item['file_path'] for item in a['reference']}) == 5
    cfg.reference.count = 99
    _, preview = select_records(records(47), cfg)
    assert preview.errors
    with pytest.raises(ValueError, match='Datensatzgrenzen'):
        service.preview(db, cfg)


def test_run_frozen_config_results_and_png_metadata(comparison):
    db, cfg, pipeline = comparison
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    pipeline.graph = {'nodes': [], 'edges': []}; db.commit()
    cfg.mean_scale = HeatmapScale(mode='manual', limit=1)
    service.run_scheduled(queued.id); db.expire_all()
    run = service.get_run(db, queued.id)
    assert run.status == 'finished', run.error_message
    assert run.config.mean_scale.mode == 'auto'
    result = service.results(db, run.id)
    assert result['reference_count'] == 4 and result['anomaly_count'] == 8
    assert result['maps']['mean']['minimum'] == pytest.approx(1200)
    assert result['maps']['variance']['minimum'] == pytest.approx(160000)
    assert result['width'] == 321 and result['height'] == 181
    for key in ('mean', 'variance'):
        path = service.artifact_path(db, run.id, result['maps'][key]['filename'])
        with Image.open(path) as png:
            assert png.format == 'PNG'
            info = json.loads(png.info['Description'])
            assert info['ddof'] == 0 and info['origin'] == 'upper'
            assert info['interpolation'] == 'nearest'
            assert info['counts'] == {'reference': 4, 'anomaly': 8}
            assert cfg.reference.start.isoformat(sep=' ') in info['periods']
    assert not (service.artifact_dir(run.id) / 'exporting').exists()
    assert service.artifact_path(db, run.id, 'manifest.json') is None


def test_single_image_and_overlapping_ranges(comparison):
    db, cfg, _ = comparison
    cfg.reference.end = cfg.reference.start
    cfg.anomaly.start = cfg.anomaly.end = cfg.reference.start
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    service.run_scheduled(queued.id); db.expire_all()
    result = service.results(db, queued.id)
    assert result['maps']['mean']['all_zero'] and result['maps']['variance']['all_zero']
    assert len(result['warnings']) == 2


@pytest.mark.parametrize('failure', ['missing', 'changed', 'size', 'nonfinite', 'export'])
def test_failure_never_publishes_partial_downloads(comparison, monkeypatch, failure):
    db, cfg, _ = comparison
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    manifest = json.loads((service.artifact_dir(queued.id) / 'manifest.json').read_text())
    path = Path(manifest['samples']['reference'][0]['file_path'])
    if failure == 'missing':
        path.unlink()
    elif failure == 'changed':
        path.write_bytes(b'changed')
    elif failure in {'size', 'nonfinite'}:
        original = service.read_frozen_image
        def broken(sample, pipeline):
            if sample['file_path'] == manifest['samples']['anomaly'][1]['file_path']:
                return np.full((3, 2), np.nan if failure == 'nonfinite' else 0)
            return original(sample, pipeline)
        monkeypatch.setattr(service, 'read_frozen_image', broken)
    else:
        original = service.render_heatmap
        def broken(values, scale, signed, *args):
            if signed:
                raise OSError('PNG export failed')
            return original(values, scale, signed, *args)
        monkeypatch.setattr(service, 'render_heatmap', broken)
    service.run_scheduled(queued.id); db.expire_all()
    assert service.get_run(db, queued.id).status == 'failed'
    assert service.results(db, queued.id) is None
    assert service.artifact_path(db, queued.id, 'mean_difference.png') is None
    assert not (service.artifact_dir(queued.id) / 'exporting').exists()


def test_abort_scheduler_dependencies_and_registry_cleanup(comparison, monkeypatch):
    from app import services
    from app.registry.specs import ENTITY_SPECS
    from app.registry.service import delete_entities, _used_ids
    from app.training.scheduler import _KINDS, normalize_queue_ranks
    db, cfg, _ = comparison
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    assert _KINDS['mean_variance']['force_cpu']
    normalize_queue_ranks(db)
    assert service.get_run(db, queued.id).queue_rank == 1
    assert cfg.training_dataset_id in _used_ids(db, 'training_dataset')
    assert any(item.entity_type == 'mean_variance_run' for item in ENTITY_SPECS['training_dataset'].dependents(db, cfg.training_dataset_id))
    with pytest.raises(ValueError, match='variance comparisons'):
        services.delete_training_dataset(db, cfg.training_dataset_id)
    with pytest.raises(ValueError):
        service.delete_run(db, queued.id)
    assert service.abort_run(db, queued.id).status == 'aborted'
    result = delete_entities(db, [('mean_variance_run', queued.id)], cascade=False)
    assert result['deleted'] == {'mean_variance_run': 1}
    assert not service.artifact_dir(queued.id).exists()
    running = service.enqueue(db, cfg, wake_scheduler=False)
    original = service.calculate
    def cancel(run, report, event):
        def intercept(step, done, total):
            if step == 'rendering':
                event.set()
            report(step, done, total)
        return original(run, intercept, event)
    monkeypatch.setattr(service, 'calculate', cancel)
    service.run_scheduled(running.id); db.expire_all()
    assert service.get_run(db, running.id).status == 'aborted'
    assert service.results(db, running.id) is None


def test_api_reopen_downloads_and_validation(comparison, monkeypatch):
    from app.database import get_db
    from app.mean_variance.api import router
    db, cfg, _ = comparison
    app = FastAPI(); app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(service.scheduler, 'wake', lambda: None)
    client = TestClient(app)
    base = '/api/mean-variance-analysis'
    payload = cfg.model_dump(mode='json')
    assert client.post(base + '/preview', json=payload).json()['reference']['selected'] == 4
    assert client.post(base + '/runs', json={**payload, 'mean_scale': {'mode': 'manual', 'limit': 0}}).status_code == 422
    queued = client.post(base + '/runs', json=payload)
    assert queued.status_code == 200, queued.text
    url = f'{base}/runs/{queued.json()["id"]}'
    assert client.get(url + '/results').status_code == 404
    assert client.delete(url).status_code == 409
    service.run_scheduled(queued.json()['id']); db.expire_all()
    assert TestClient(app).get(url).json()['status'] == 'finished'
    assert client.get(base + '/runs').json()[0]['config'] == payload
    assert client.get(url + '/results').json()['ddof'] == 0
    for filename in ['mean_difference.png', 'variance_difference.png']:
        download = client.get(url + '/artifacts/' + filename + '?download=true')
        assert download.headers['content-type'] == 'image/png'
        assert download.headers['content-disposition'].startswith('attachment')
    for filename in ['manifest.json', 'results.json', 'reference.npy', 'video.mp4']:
        assert client.get(url + '/artifacts/' + filename).status_code == 404
    assert client.get(url + '/log').status_code == 200
    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404


@pytest.mark.parametrize("version", [1, 2])
def test_migration_project_isolation_scheduler_and_reopen(tmp_path, monkeypatch, version):
    from app import database, projects
    from app.main import app
    from tests.test_projects import configure_catalog
    configure_catalog(monkeypatch, tmp_path); projects.initialize_catalog()
    first, second = [projects.create_project(name, 'Testing') for name in ['Mean one', 'Mean two']]
    cfg = MeanVarianceConfig(training_dataset_id=1, preprocessing_pipeline_id=1, reference=interval(0, 2), anomaly=interval(4, 6))
    if version == 2:
        from app.mean_variance.schemas import VarianceConfig
        cfg = VarianceConfig(training_dataset_id=1, preprocessing_pipeline_id=1, pairs=[{
            'normal': cfg.reference.model_dump(include={'start', 'end'}),
            'anomaly': cfg.anomaly.model_dump(include={'start', 'end'}),
        }])
    with database.project_context(first.database_url, first.artifact_dir):
        with database.SessionLocal() as db:
            dataset = models.TrainingDataset(name='Only first project', usage_label='test')
            db.add(dataset); db.flush()
            db.add(models.MeanVarianceRun(training_dataset_id=dataset.id, training_dataset_name=dataset.name, config=cfg.model_dump(mode='json'), dataset_snapshot={}, pipeline_snapshot={}, status='finished'))
            db.commit()
    client = TestClient(app); base = '/api/mean-variance-analysis/runs'
    assert client.get(base).status_code == 400
    h1, h2 = [{'X-MLTrace-Project-ID': p.id} for p in (first, second)]
    assert client.get(base, headers=h1).json()[0]['training_dataset_name'] == 'Only first project'
    assert client.get(base, headers=h2).json() == []
    assert client.get(base + '/1', headers=h2).status_code == 404
    jobs = client.get('/api/scheduler/jobs?scope=all').json()
    assert any(job['kind'] == 'mean_variance' and job['project_id'] == first.id for job in jobs)
    projects.initialize_catalog()
    assert TestClient(app).get(base + '/1', headers=h1).json()['config'] == cfg.model_dump(mode='json')


def test_additive_migration_preserves_presets(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect, text
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / 'alembic.ini'))
    config.set_main_option('script_location', str(root / 'backend/alembic'))
    url = f'sqlite:///{tmp_path / "migration.db"}'
    config.set_main_option('sqlalchemy.url', url)
    command.upgrade(config, '0062_time_range_presets')
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO time_range_presets(name,name_key,start,end) VALUES ('Existing','existing','2026-01-01','2026-01-02')"))
    command.upgrade(config, 'head')
    command.upgrade(config, 'head')
    assert 'mean_variance_runs' in inspect(engine).get_table_names()
    with engine.connect() as connection:
        assert connection.execute(text('SELECT name FROM time_range_presets')).scalar() == 'Existing'
    command.downgrade(config, '0062_time_range_presets')
    assert 'mean_variance_runs' not in inspect(engine).get_table_names()
    engine.dispose()
