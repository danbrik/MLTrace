import json
import shutil
import threading
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from app import models
from app.mean_variance import service, roi_service as roi
from app.mean_variance import roi_engine as engine
from app.mean_variance.roi_api import router
from tests.test_reference_image import source
from tests.test_mean_variance import comparison
from tests.test_variance_pairs import paired


@pytest.fixture
def setup(comparison, monkeypatch, tmp_path):
    monkeypatch.setattr(roi, 'artifact_dir', lambda id: tmp_path / 'roi_jobs' / str(id))
    monkeypatch.setattr(roi.scheduler, 'wake', lambda: None)
    return comparison


def completed(setup, count=1, legacy=False):
    db, old, _ = setup
    run = service.enqueue(db, old if legacy else paired(old, count), wake_scheduler=False)
    service.run_scheduled(run.id); db.expire_all()
    assert service.get_run(db, run.id).status == 'finished'
    return run


def config(**kwargs):
    return engine.RoiConfig(roi={'version': 2, 'center_x': 25, 'center_y': 40, 'width': 30, 'height': 40, 'angle_degrees': 0}, **kwargs)


def test_positive_metrics_signed_cancellation_and_boundaries():
    data = np.array([[4., -4], [2, 0]])
    rect = engine.Rectangle(x=0, y=0, width=1, height=1)
    result = engine.positive_metrics(data, rect)
    assert result['positive_total'] == 6 and result['positive_roi'] == 4
    assert result['area_percent'] == 25
    assert result['increase_percent'] == pytest.approx(100 * 4 / 6)
    assert engine.positive_metrics(data, engine.Rectangle(x=0, y=0, width=2, height=2))['increase_percent'] == 100
    assert engine.positive_metrics(-np.ones((2, 2)), rect)['increase_percent'] is None
    assert engine.positive_metrics(np.zeros((2, 2)), rect)['increase_percent'] is None
    assert engine.positive_metrics(data, engine.Rectangle(x=1, y=1, width=1, height=1))['positive_roi'] == 0
    with pytest.raises(ValueError, match='innerhalb'):
        engine.positive_metrics(data, engine.Rectangle(x=1, y=1, width=2, height=2))
    with pytest.raises(ValueError, match='endlich'):
        engine.positive_metrics(np.full((2, 2), 1e308), rect)


@pytest.mark.parametrize('payload', [
    {'x': -1, 'y': 0, 'width': 1, 'height': 1}, {'x': 0.5, 'y': 0, 'width': 1, 'height': 1},
    {'x': 0, 'y': 0, 'width': 0, 'height': 1}, {'x': True, 'y': 0, 'width': 1, 'height': 1},
])
def test_rectangle_validation(payload):
    with pytest.raises(ValidationError):
        engine.Rectangle(**payload)


@pytest.mark.parametrize('count', [1, 6])
def test_basis_and_export_roundtrip(setup, count):
    db, _, _ = setup
    run = completed(setup, count)
    state = roi.state(db, run.id)
    assert state['ready'] and len(state['basis']['pairs']) == count
    np.testing.assert_array_equal(np.load(roi.basis_dir(run.id) / '0_mean.npy'), 1300)
    assert np.load(roi.basis_dir(run.id) / '0_difference.npy').dtype == np.float64
    job = roi.enqueue(db, run.id, 'evaluate', config(), wake_scheduler=False)
    with pytest.raises(ValueError, match='bereits'):
        roi.enqueue(db, run.id, 'evaluate', config(), wake_scheduler=False)
    with pytest.raises(ValueError, match='Aktive ROI'):
        service.delete_run(db, run.id)
    roi.run_scheduled(job['id']); db.expire_all()
    state = roi.state(db, run.id)
    assert state['job']['status'] == 'finished', state['job']['error_message']
    result = state['saved']['result']
    assert result['roi'] == config().roi.model_dump()
    assert result['mean_increase_percent'] == pytest.approx(1200 / (181 * 321) * 100)
    assert result['valid_pairs'] == count
    for name in (result['plot'], result['table']):
        with Image.open(roi.artifact_dir(job['id']) / name) as image:
            metadata = json.loads(image.info['Description'])
            assert metadata['roi'] == result['roi'] and metadata['config'] == run.config.model_dump(mode='json')
            assert metadata['pipeline_name'] == 'Original'
    assert service.get_run(db, run.id).status == 'finished'
    db.expire_all()
    assert roi.state(db, run.id)['saved']['config'] == config().model_dump()


@pytest.mark.parametrize('legacy', [True, False])
def test_backfill_preserves_original_frozen_selection_and_pipeline(setup, legacy):
    db, _, pipeline = setup
    run = completed(setup, legacy=legacy)
    before = service.get_run(db, run.id).model_dump(mode='json')
    expected = np.load(roi.basis_dir(run.id) / '0_difference.npy')
    shutil.rmtree(roi.basis_dir(run.id))
    pipeline.graph = {'nodes': [], 'edges': []}; db.commit()
    assert not roi.state(db, run.id)['ready']
    job = roi.enqueue(db, run.id, 'prepare', wake_scheduler=False)
    roi.run_scheduled(job['id']); db.expire_all()
    state = roi.state(db, run.id)
    assert state['job']['status'] == 'finished', state['job']['error_message']
    assert state['ready']
    np.testing.assert_array_equal(np.load(roi.basis_dir(run.id, db) / '0_difference.npy'), expected)
    assert service.get_run(db, run.id).model_dump(mode='json') == before
    export = roi.enqueue(db, run.id, 'evaluate', config(), wake_scheduler=False)
    with pytest.raises(ValueError, match='aktiven ROI'):
        roi.delete_job(db, job['id'])
    roi.abort_job(db, export['id'])
    roi.delete_job(db, job['id'])
    assert roi.state(db, run.id)['ready']  # Parent keeps completed numerical preparation.


@pytest.mark.parametrize('failure', ['missing', 'changed', 'abort'])
def test_backfill_failure_does_not_publish(setup, failure):
    db, _, _ = setup
    run = completed(setup)
    shutil.rmtree(roi.basis_dir(run.id))
    files = json.loads((service.artifact_dir(run.id) / 'manifest.json').read_text())['samples']
    first = Path(files['0_reference'][0]['file_path'])
    if failure == 'missing': first.unlink()
    if failure == 'changed': first.write_bytes(b'changed')
    job = roi.enqueue(db, run.id, 'prepare', wake_scheduler=False)
    event = threading.Event()
    if failure == 'abort': event.set()
    roi.run_scheduled(job['id'], event); db.expire_all()
    state = roi.state(db, run.id)
    assert state['job']['status'] == ('aborted' if failure == 'abort' else 'failed')
    assert not state['ready'] and state['saved'] is None
    assert service.results(db, run.id) is not None


@pytest.mark.parametrize('failure', ['render', 'abort', 'late_abort'])
def test_new_export_failure_keeps_previous_result(setup, monkeypatch, failure):
    db, _, _ = setup
    run = completed(setup)
    first = roi.enqueue(db, run.id, 'evaluate', config(), wake_scheduler=False)
    roi.run_scheduled(first['id']); db.expire_all()
    original = roi.state(db, run.id)['saved']
    second = roi.enqueue(db, run.id, 'evaluate', config(opacity=.8), wake_scheduler=False)
    event = threading.Event()
    original_export = roi.export_roi
    def fail(*args):
        if failure == 'render': raise OSError('PNG export failed')
        if failure == 'late_abort':
            result = original_export(*args)
            event.set()
            return result
        event.set()
        return original_export(*args)
    monkeypatch.setattr(roi, 'export_roi', fail)
    roi.run_scheduled(second['id'], event); db.expire_all()
    state = roi.state(db, run.id)
    assert state['job']['status'] == ('failed' if failure == 'render' else 'aborted')
    assert state['saved'] == original
    assert not (roi.artifact_dir(second['id']) / 'exporting').exists()


def test_api_artifacts_abort_registry_and_cleanup(setup):
    from app.database import get_db
    from app.registry.specs import ENTITY_SPECS, DELETE_ORDER
    from app.registry.service import delete_entities
    from app.training.scheduler import _KINDS
    db, _, _ = setup
    run = completed(setup)
    app = FastAPI(); app.include_router(router); app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app); base = f'/api/mean-variance-analysis/runs/{run.id}/roi'
    assert client.get(base).json()['ready']
    assert client.post(base+'/evaluate', json={'roi': {'x': 0, 'y': 0, 'width': 900, 'height': 1}}).status_code == 409
    for pair, layer, status in [(0, 'background', 200), (0, 'heatmap', 200), (0, 'mean.npy', 404), (8, 'background', 404), (-1, 'background', 404)]:
        assert client.get(f'{base}/images/{pair}/{layer}').status_code == status
    job = client.post(base+'/evaluate', json=config().model_dump()).json()
    artifact = f"{base}/artifacts/{job['id']}/roi_comparison.png"
    assert client.get(artifact).status_code == 404
    assert _KINDS['variance_roi']['force_cpu'] and DELETE_ORDER.index('variance_roi_job') < DELETE_ORDER.index('mean_variance_run')
    dependents = ENTITY_SPECS['mean_variance_run'].dependents(db, run.id)
    assert dependents[0].entity_type == 'variance_roi_job'
    roi.run_scheduled(job['id']); db.expire_all()
    response = client.get(artifact+'?download=true')
    assert response.status_code == 200 and 'attachment' in response.headers['content-disposition']
    assert client.get(f"{base}/artifacts/{job['id']}/results.json").status_code == 404
    queued = client.post(base+'/evaluate', json=config().model_dump()).json()
    assert client.post(f"/api/mean-variance-analysis/roi-jobs/{queued['id']}/abort").json()['status'] == 'aborted'
    result = delete_entities(db, [('mean_variance_run', run.id)], cascade=True)
    assert result['deleted']['mean_variance_run'] == 1
    assert not service.artifact_dir(run.id).exists() and not roi.artifact_dir(job['id']).exists()
    assert list(db.scalars(select(models.VarianceRoiJob))) == []


def test_renderer_signed_color_shared_scales_mean_nulls_and_layout(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    directory = tmp_path / 'basis'; directory.mkdir()
    normal = np.array([[0., 1, 2], [3, 4, 5]])
    for index, data in enumerate([[[4, -4, 0], [2, 0, 0]], [[-1, -1, 0], [0, 0, 0]], [[0, 2, 0], [0, 0, 6]]]):
        np.save(directory / f'{index}_mean.npy', normal + index * 5)
        np.save(directory / f'{index}_difference.npy', np.array(data, dtype=float))
    result = {'version': 2, 'width': 3, 'height': 2, 'difference_scale_limit': 2,
              'pairs': [{'label': f'u{i+1}', 'periods': {}, 'counts': {}} for i in range(3)]}
    engine.finish_basis(directory, {}, result, 'Dataset', 'Pipeline')
    basis = engine.load_basis(directory)
    assert basis['background_min'] == 0 and basis['background_max'] == 15
    heatmap = np.array(Image.open(directory/'0_heatmap.png'))
    assert heatmap[0, 0, 0] > heatmap[0, 0, 2] and heatmap[0, 1, 2] > heatmap[0, 1, 0]
    np.testing.assert_array_equal(heatmap[0, 2], [255, 255, 255])
    np.testing.assert_array_equal(engine.composite(directory, 0, 0), np.array(Image.open(directory/'0_background.png')))
    np.testing.assert_array_equal(engine.composite(directory, 0, 1), heatmap)
    saved = []
    original = Figure.savefig
    def inspect(fig, path, *args, **kwargs):
        if Path(path).name == 'roi_comparison.png':
            texts = [t.get_text() for t in fig.texts]
            assert texts.count('x (Pixel)') == texts.count('y (Pixel)') == 1
            assert all(f'u{i+1}' in texts for i in range(3))
            assert 'Gesamtbild mit ROI' in texts and 'ROI-Ausschnitt' in texts
            assert fig.axes[-1].get_xlabel() == 'Variance difference (gray value²)'
            for i in range(3):
                left, right = fig.axes[2*i:2*i+2]
                np.testing.assert_array_equal(right.images[0].get_array(), left.images[0].get_array()[0:1,0:1])
                assert left.images[0].origin == 'upper' and right.images[0].get_interpolation() == 'nearest'
                assert len(left.patches) == 1 and not right.patches
            saved.append(True)
        return original(fig, path, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', inspect)
    exported = engine.export_roi(directory, tmp_path, engine.RoiConfig(roi={'x':0,'y':0,'width':1,'height':1}))
    assert exported['valid_pairs'] == 2 and exported['mean_increase_percent'] == pytest.approx(100/3)
    assert exported['rows'][1]['increase_percent'] is None and saved


def test_roi_project_isolation_restart_migration_and_scheduler(tmp_path, monkeypatch):
    from app import database, projects
    from app.main import app
    from tests.test_projects import configure_catalog
    from sqlalchemy import inspect
    configure_catalog(monkeypatch, tmp_path); projects.initialize_catalog()
    monkeypatch.setattr(roi.scheduler, 'wake', lambda: None)
    first, second = [projects.create_project(name, 'ROI test') for name in ['ROI one', 'ROI two']]
    cfg = {
        'version': 2, 'training_dataset_id': 1, 'preprocessing_pipeline_id': 1, 'sampling_rate': 1,
        'pairs': [{'normal': {'start': '2026-01-01T00:00:00', 'end': '2026-01-01T00:00:01'},
                   'anomaly': {'start': '2026-01-01T00:00:02', 'end': '2026-01-01T00:00:03'}}],
        'variance_scale': {'mode': 'auto', 'limit': None}, 'difference_scale': {'mode': 'auto', 'limit': None}}
    for project in (first, second):
        with database.project_context(project.database_url, project.artifact_dir), database.SessionLocal() as db:
            assert 'variance_roi_jobs' in inspect(db.get_bind()).get_table_names()
            dataset = models.TrainingDataset(name=project.name, usage_label='test')
            db.add(dataset); db.flush()
            db.add(models.MeanVarianceRun(training_dataset_id=dataset.id, training_dataset_name=dataset.name,
                config=cfg, dataset_snapshot={}, pipeline_snapshot={'name': 'Frozen pipeline', 'graph': {}}, status='finished'))
            db.commit()
    client = TestClient(app); base = '/api/mean-variance-analysis'
    h1, h2 = [{'X-MLTrace-Project-ID': p.id} for p in (first, second)]
    assert client.get(base+'/runs/1/roi').status_code == 400
    job = client.post(base+'/runs/1/roi/prepare', headers=h1).json()
    assert job['status'] == 'queued'
    assert client.get(base+'/runs/1/roi', headers=h2).json()['job'] is None
    assert client.post(base+f"/roi-jobs/{job['id']}/abort", headers=h2).status_code == 404
    assert client.get(base+f"/runs/1/roi/artifacts/{job['id']}/roi_comparison.png", headers=h2).status_code == 404
    assert client.get(base+'/roi-jobs', headers=h2).json() == []
    jobs = client.get('/api/scheduler/jobs?scope=all').json()
    assert any(j['kind'] == 'variance_roi' and j['project_id'] == first.id for j in jobs)
    other = client.post(base+'/runs/1/roi/prepare', headers=h2).json()
    projects.ensure_queue_entry(first.id, 'variance_roi', job['id'])
    projects.ensure_queue_entry(second.id, 'variance_roi', other['id'])
    assert client.post(f"/api/scheduler/jobs/variance_roi/{job['id']}/move", headers=h1, json={'direction': 'down'}).status_code == 200
    projects.initialize_catalog()
    reopened = TestClient(app).get(base+'/runs/1/roi', headers=h1).json()
    assert reopened['job']['id'] == job['id']
    assert client.post(base+f"/roi-jobs/{job['id']}/abort", headers=h1).json()['status'] == 'aborted'
    assert client.delete(base+f"/roi-jobs/{job['id']}", headers=h1).status_code == 204


def test_constant_background_all_zero_export_and_one_pixel_crop(tmp_path):
    directory = tmp_path / 'basis'; directory.mkdir()
    np.save(directory / '0_mean.npy', np.full((2, 3), 65535., dtype=np.float64))
    np.save(directory / '0_difference.npy', np.zeros((2, 3), dtype=np.float64))
    result = {'version': 2, 'width': 3, 'height': 2, 'difference_scale_limit': 0,
              'pairs': [{'label': 'u1', 'periods': {}, 'counts': {'normal':1, 'anomaly':1}}]}
    engine.finish_basis(directory, {}, result, 'Dataset', 'Pipeline')
    np.testing.assert_array_equal(np.array(Image.open(directory/'0_background.png')), 128)
    np.testing.assert_array_equal(np.array(Image.open(directory/'0_heatmap.png')), 255)
    result = engine.export_roi(directory, tmp_path, engine.RoiConfig(roi={'x':2,'y':1,'width':1,'height':1}))
    assert result['mean_increase_percent'] is None and result['valid_pairs'] == 0
    assert result['area_percent'] == pytest.approx(100/6) and result['warnings']
    assert (tmp_path / result['table']).is_file()
