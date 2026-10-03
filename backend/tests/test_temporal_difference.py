from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app import models
from app.temporal_difference import engine, service
from app.temporal_difference.schemas import TemporalDifferenceConfig, PlotSettings
from app.training.data import ResolvedDatasetImage
from tests.test_reference_image import source

BASE = datetime(2026, 1, 1)


def period(start=0, end=6):
    return {"start": BASE + timedelta(seconds=start), "end": BASE + timedelta(seconds=end)}


def config(**changes):
    return TemporalDifferenceConfig.model_validate(dict(training_dataset_id=1, preprocessing_pipeline_id=1,
        reference=period(), comparison=period(8, 22), deltas_seconds=[1, 2, 5, 15, 30, 60]) | changes)


def records(seconds):
    return [ResolvedDatasetImage(str(i), BASE + timedelta(seconds=value), "images", "/images", 1, ".", str(i)) for i, value in enumerate(seconds)]


@pytest.fixture
def temporal(source, tmp_path, monkeypatch):
    db, old, pipeline = source
    monkeypatch.setattr(service, "artifact_dir", lambda id: tmp_path / "temporal" / str(id))
    return db, config(training_dataset_id=old.training_dataset_id, preprocessing_pipeline_id=pipeline.id), pipeline


@pytest.mark.parametrize("deltas", [[], [0], [-1], [1.5], [True], ["1"], [1, 1], [9007199254740992]])
def test_invalid_deltas(deltas):
    with pytest.raises(ValidationError):
        config(deltas_seconds=deltas)


def test_defaults_and_sorted_deltas():
    payload = config().model_dump(); payload.pop("deltas_seconds")
    assert TemporalDifferenceConfig.model_validate(payload).deltas_seconds == [1, 2, 5, 15, 30, 60]
    assert config(deltas_seconds=[60, 1, 15]).deltas_seconds == [1, 15, 60]
    with pytest.raises(ValidationError):
        PlotSettings(x_range={"minimum": 3, "maximum": 2})
    with pytest.raises(ValidationError):
        PlotSettings(reference_color="red")


def test_exact_pairs_gaps_bounds_unsorted_and_duplicates():
    cfg = config(reference=period(0, 6), comparison=period(0, 6), deltas_seconds=[1, 2, 60])
    rows = records([0, 1, 3, 6, 7])
    samples, pairs, preview = engine.select_pairs(list(reversed(rows)) + rows, cfg)
    assert len(samples['reference']) == 4
    assert preview['periods']['reference']['deltas'] == [
        {'delta_seconds': 1, 'pair_count': 1, 'missing_targets': 3},
        {'delta_seconds': 2, 'pair_count': 1, 'missing_targets': 3},
        {'delta_seconds': 60, 'pair_count': 0, 'missing_targets': 4}]
    assert len(pairs) == 4 and not preview['errors']
    duplicate = records([0])[0]
    duplicate = ResolvedDatasetImage('other', duplicate.timestamp_parsed, 'images', '/images', 1, '.', 'other')
    with pytest.raises(ValueError, match='2026-01-01 00:00:00'):
        engine.select_pairs(rows + [duplicate], cfg)
    _, _, empty = engine.select_pairs(rows, config(reference=period(2, 2), comparison=period(2, 2)))
    assert len(empty['errors']) == 2


@pytest.mark.parametrize('local,utc', [('2025-09-15T20:45:00','2025-09-15T18:45:00+00:00'),
    ('2025-01-15T20:45:00','2025-01-15T19:45:00+00:00'), ('2025-09-15T00:45:00','2025-09-14T22:45:00+00:00')])
def test_utc(local, utc):
    assert engine.utc_instant(datetime.fromisoformat(local)).isoformat() == utc


@pytest.mark.parametrize('local', ['2025-03-30T02:30:00', '2025-10-26T02:30:00'])
def test_ambiguous_or_nonexistent_times(local):
    with pytest.raises(ValueError, match=local.replace('T', ' ')):
        engine.utc_instant(datetime.fromisoformat(local))


def test_elapsed_seconds_across_spring_change():
    cfg = config(reference={'start': '2025-03-30T01:59:59', 'end': '2025-03-30T03:00:00'},
                 comparison={'start': '2025-03-30T01:59:59', 'end': '2025-03-30T03:00:00'}, deltas_seconds=[1])
    rows = [ResolvedDatasetImage(str(i), datetime.fromisoformat(t), 'images', '/', 1, '.', str(i))
            for i, t in enumerate(['2025-03-30T01:59:59', '2025-03-30T03:00:00'])]
    _, pairs, preview = engine.select_pairs(rows, cfg)
    assert len(pairs) == 2 and not preview['errors']


def test_values_and_linear_quantiles():
    assert engine.absolute_change(np.array([[65000, 100]], dtype=np.uint16), np.array([[0, 200]], dtype=np.uint16)) == 32550
    assert engine.absolute_change(np.ones((2, 3)), np.ones((2, 3))) == 0
    assert engine.statistics([0, 1, 3, 10]) == dict(pair_count=4, median=2, q1=.75, q3=4.75, iqr=4)
    assert engine.statistics([7]) == dict(pair_count=1, median=7, q1=7, q3=7, iqr=0)
    assert engine.statistics([])['median'] is None
    for bad in [np.array([[np.nan]]), np.ones((1, 1, 3)), np.empty((0, 2))]:
        with pytest.raises(ValueError):
            engine.absolute_change(np.ones((1, 1)), bad)
    with pytest.raises(ValueError, match='Bildgrößen'):
        engine.absolute_change(np.ones((1, 1)), np.ones((2, 2)))


def test_full_run_frozen_sampling_tables_and_reopen(temporal, monkeypatch):
    db, cfg, pipeline = temporal
    cfg.reference.start += timedelta(seconds=1)
    preview = service.preview(db, cfg)
    assert preview['periods']['reference']['image_count'] == 3  # saved stride from original rule, then clip
    assert preview['periods']['reference']['deltas'][1]['pair_count'] == 2
    run = service.enqueue(db, cfg, wake_scheduler=False)
    assert service.summaries(db, run.id) is None and service.values(db, run.id) is None
    manifest = json.loads((service.artifact_dir(run.id) / 'manifest.json').read_text())
    assert manifest['selection'] == preview
    assert all('mtime_ns' in sample for group in manifest['samples'].values() for sample in group)
    pipeline.graph = {'nodes': [], 'edges': []}; db.commit()
    service.run_scheduled(run.id); db.expire_all()
    saved = service.get_run(db, run.id)
    assert saved.status == 'finished', saved.error_message
    summary = service.summaries(db, run.id)
    assert len(summary) == 12
    assert all(row['median'] == 200 and row['iqr'] == 0 for row in summary if row['delta_seconds'] == 2)
    assert all(row['median'] is None for row in summary if row['delta_seconds'] != 2)
    values = service.values(db, run.id, offset=1, limit=2, role='reference', delta=2)
    assert values['total'] == 2 and len(values['items']) == 1
    assert values['items'][0]['first_timestamp'].endswith('00:00:04')
    assert values['items'][0]['value'] == 200
    assert saved.result['total_pairs'] == 9
    raw = service.csv_path(db, run.id, 'summary').read_text()
    assert '2,2,200.0,200.0,200.0,0.0,7,200.0' in raw
    assert '"200.0"' not in raw
    assert len(list(csv.DictReader(raw.splitlines()))) == 6
    assert len(list(csv.DictReader(service.csv_path(db, run.id, 'pairs').read_text().splitlines()))) == 9
    def no_read(*args):
        raise AssertionError('Finished results must not access images')
    monkeypatch.setattr(service, 'read_frozen_image', no_read)
    settings = PlotSettings(title='Saved plot', x_range={'minimum': 0, 'maximum': 65})
    assert service.save_plot(db, run.id, settings) == settings
    assert service.get_run(db, run.id).plot_settings == settings
    assert service.summaries(db, run.id) == summary
    assert service.csv_path(db, run.id, 'summary').exists()
    assert service.delete_run(db, run.id)
    assert db.scalar(select(func.count()).select_from(models.TemporalDifferenceValue)) == 0


@pytest.mark.parametrize('failure', ['changed', 'missing', 'abort', 'shape', 'nonfinite'])
def test_failures_do_not_publish_results(temporal, monkeypatch, failure):
    db, cfg, _ = temporal
    run = service.enqueue(db, cfg, wake_scheduler=False)
    manifest = json.loads((service.artifact_dir(run.id) / 'manifest.json').read_text())
    sample = manifest['samples']['reference'][1]
    if failure == 'changed':
        Path(sample['file_path']).write_bytes(b'changed')
    if failure == 'missing':
        Path(sample['file_path']).unlink()
    if failure in {'shape', 'nonfinite'}:
        original = service.read_frozen_image
        monkeypatch.setattr(service, 'read_frozen_image', lambda row, pipe: np.full((2, 2), np.nan if failure == 'nonfinite' else 0)
            if row['file_path'] == sample['file_path'] else original(row, pipe))
    event = threading.Event()
    if failure == 'abort':
        event.set()
    service.run_scheduled(run.id, event); db.expire_all()
    assert service.get_run(db, run.id).status == ('aborted' if failure == 'abort' else 'failed')
    assert service.summaries(db, run.id) is None
    assert service.csv_path(db, run.id, 'summary') is None
    assert db.scalar(select(func.count()).select_from(models.TemporalDifferenceValue)) == 0


def test_bounds_and_no_pairs_rejected_on_preview_and_start(temporal):
    db, cfg, _ = temporal
    cfg.reference.start -= timedelta(seconds=1)
    for operation in (service.preview, service.enqueue):
        with pytest.raises(ValueError, match='Datensatzgrenzen'):
            operation(db, cfg)
    cfg.reference.start += timedelta(seconds=1)
    cfg.deltas_seconds = [1]
    assert service.preview(db, cfg)['errors']
    with pytest.raises(ValueError, match='Bildpaare'):
        service.enqueue(db, cfg, wake_scheduler=False)
    assert not service.list_runs(db)


def test_api_lifecycle_and_pagination(temporal, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.database import get_db
    from app.temporal_difference.api import router
    db, cfg, _ = temporal
    app = FastAPI(); app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(service.scheduler, 'wake', lambda: None)
    with TestClient(app) as client:
        payload = cfg.model_dump(mode='json')
        assert client.post('/api/temporal-difference/preview', json=payload).status_code == 200
        created = client.post('/api/temporal-difference/runs', json=payload)
        assert created.status_code == 200, created.text
        path = f"/api/temporal-difference/runs/{created.json()['id']}"
        assert client.get(path + '/summary').status_code == 404
        service.run_scheduled(created.json()['id']); db.expire_all()
        assert client.get(path + '/summary').status_code == 200
        assert client.get(path + '/pairs?limit=1').json()['total'] == 10
        assert client.get(path + '/pairs?offset=-1').status_code == 422
        assert client.get(path + '/csv/summary').headers['content-type'].startswith('text/csv')
        assert client.put(path + '/plot-settings', json=PlotSettings(title='Changed').model_dump()).status_code == 200
        assert client.get(path).json()['plot_settings']['title'] == 'Changed'
        assert client.delete(path).status_code == 204
        assert client.get(path).status_code == 404


def test_additive_migration_and_project_isolation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import database, projects
    from app.main import app
    from tests.test_projects import configure_catalog
    from sqlalchemy import inspect
    configure_catalog(monkeypatch, tmp_path)
    projects.initialize_catalog()
    first, second = [projects.create_project(name, 'Temporal test') for name in ['First', 'Second']]
    with database.project_context(first.database_url, first.artifact_dir):
        with database.SessionLocal() as db:
            assert {'temporal_difference_runs', 'temporal_difference_values', 'temporal_difference_summaries'} <= set(inspect(db.bind).get_table_names())
            dataset = models.TrainingDataset(name='Only first', usage_label='test')
            db.add(dataset); db.flush()
            db.add(models.TemporalDifferenceRun(training_dataset_id=dataset.id, training_dataset_name=dataset.name,
                config=config().model_dump(mode='json'), dataset_snapshot={}, pipeline_snapshot={'id': 1, 'name': 'Saved'},
                plot_settings=PlotSettings().model_dump(), status='finished'))
            db.commit()
    client = TestClient(app)
    base = '/api/temporal-difference/runs'
    h1, h2 = [{'X-MLTrace-Project-ID': project.id} for project in (first, second)]
    assert client.get(base).status_code == 400
    assert len(client.get(base, headers=h1).json()) == 1
    assert client.get(base, headers=h2).json() == []
    for endpoint in ['/1', '/1/summary', '/1/pairs', '/1/csv/pairs']:
        assert client.get(base + endpoint, headers=h2).status_code == 404
    assert client.put(base + '/1/plot-settings', headers=h2, json=PlotSettings().model_dump()).status_code == 404
    jobs = client.get('/api/scheduler/jobs?scope=all').json()
    assert any(job['kind'] == 'temporal_difference' and job['project_id'] == first.id for job in jobs)
    projects.initialize_catalog()
    assert TestClient(app).get(base + '/1', headers=h1).json()['config'] == config().model_dump(mode='json')


def test_upgrade_and_downgrade_preserve_existing_data(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect, text
    root = Path(__file__).resolve().parents[2]
    settings = Config(str(root / 'alembic.ini'))
    settings.set_main_option('script_location', str(root / 'backend/alembic'))
    url = f'sqlite:///{tmp_path / "migration.db"}'
    settings.set_main_option('sqlalchemy.url', url)
    command.upgrade(settings, '0064_variance_roi')
    db = create_engine(url)
    with db.begin() as connection:
        connection.execute(text("INSERT INTO time_range_presets(name,name_key,start,end) VALUES ('Saved','saved','2026-01-01','2026-01-02')"))
    command.upgrade(settings, 'head')
    command.upgrade(settings, 'head')
    assert 'temporal_difference_values' in inspect(db).get_table_names()
    command.downgrade(settings, '0064_variance_roi')
    assert 'temporal_difference_values' not in inspect(db).get_table_names()
    with db.connect() as connection:
        assert connection.execute(text('SELECT name FROM time_range_presets')).scalar() == 'Saved'
    db.dispose()


def test_scheduler_worker_and_registry_contract(temporal):
    from app.training.scheduler import _KINDS
    from app.registry.specs import ENTITY_SPECS, _training_dataset_dependents
    db, cfg, _ = temporal
    run = service.enqueue(db, cfg, wake_scheduler=False)
    assert _KINDS['temporal_difference']['force_cpu']
    assert _KINDS['temporal_difference']['model'] is models.TemporalDifferenceRun
    assert ENTITY_SPECS['temporal_difference_run'].model is models.TemporalDifferenceRun
    assert any(item.entity_type == 'temporal_difference_run' for item in _training_dataset_dependents(db, cfg.training_dataset_id))
    with pytest.raises(ValueError):
        service.delete_run(db, run.id)
    assert service.abort_run(db, run.id).status == 'aborted'
    assert service.delete_run(db, run.id)


def test_cancellation_during_calculation_clears_partial_values(temporal, monkeypatch):
    db, cfg, _ = temporal
    run = service.enqueue(db, cfg, wake_scheduler=False)
    event = threading.Event()
    original = service.absolute_change
    def abort(first, second):
        event.set()
        return original(first, second)
    monkeypatch.setattr(service, 'absolute_change', abort)
    service.run_scheduled(run.id, event); db.expire_all()
    assert service.get_run(db, run.id).status == 'aborted'
    assert service.summaries(db, run.id) is None
    assert db.scalar(select(func.count()).select_from(models.TemporalDifferenceValue)) == 0
