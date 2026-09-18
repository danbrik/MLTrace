from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import io
import json

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.data_quality import characterization as service, characterization_engine as engine
from app.data_quality import service as quality
from app.data_quality.schemas import Parameters
from app.redundancy.service import create_source


def test_summary_quantiles_and_text():
    a = engine.summary('a', 'numeric', np.array([1., 2., 3., 4., 5.]))
    assert a['mean'] == a['median'] == 3
    assert a['std'] == pytest.approx(np.sqrt(2.5))
    assert [a[k] for k in ('min', 'q01', 'q05', 'q25', 'q75', 'q95', 'q99', 'max', 'iqr')] == pytest.approx([1, 1.04, 1.2, 2, 4, 4.8, 4.96, 5, 2])
    assert engine.summary('text', 'text', ['x', 'y'])['mean'] is None
    assert engine.summary('empty', 'numeric', [])['valid_n'] == 0
    assert engine.summary('one', 'numeric', [5])['std'] is None


def test_dynamics_only_adjacent_valid_grid_points():
    result = engine.details([0, 1, 3, 4, 8], [1, 1, 10, 14, 900])
    dynamics = result['dynamics']
    assert dynamics['pair_count'] == 2
    assert dynamics['median_abs'] == 2
    assert dynamics['q95_abs'] == pytest.approx(3.8)
    assert dynamics['q99_abs'] == pytest.approx(3.96)
    assert dynamics['max_abs'] == 4
    assert dynamics['unchanged_percent'] == 50
    assert sum(result['delta_histogram']['counts']) == 2
    assert engine.details([0, 2], [1, 10])['dynamics']['unchanged_percent'] is None
    constant = engine.details([0, 1, 2], [5, 5, 5])
    assert constant['dynamics']['unchanged_percent'] == 100
    assert constant['histogram']['counts'] == [3]
    assert engine.details([], [])['box'] is None


def test_histogram_counts_and_box_outliers():
    values = np.array([0., 1., 2., 3., 4., 5., 100.])
    data = engine.details(np.arange(7), values)
    assert sum(data['histogram']['counts']) == len(values)
    assert 1 <= len(data['histogram']['counts']) <= 100
    assert data['box']['outlier_count'] == 1
    assert data['box']['upper'] == 5
    fallback = engine.histogram(np.array([0.] * 99 + [100.]))
    assert len(fallback['counts']) == 10 and sum(fallback['counts']) == 100
    assert engine.histogram(np.arange(10000.))['counts']


def test_temporal_raw_aggregation_and_no_gap_bridging():
    meta = dict(grid_start='2026-01-01T00:00:00', grid_count=3000, interval_seconds=60)
    indices = np.array([0, 1, 3, 4, 5, 7, 8, 9])
    values = indices.astype(float)
    view = engine.temporal(meta, indices, values)
    assert view['aggregated'] and len(view['points']) == 1000
    assert view['points'][0]['value'] == .5
    assert view['points'][0]['q25'] == .25
    assert view['points'][1]['connect_previous'] is False  # Missing index 2.
    assert view['points'][2]['connect_previous'] is False  # Missing index 6.
    assert view['points'][3]['connect_previous'] is True
    detail = engine.temporal(meta, indices, values, datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 4))
    assert not detail['aggregated'] and len(detail['points']) == 5
    assert detail['points'][2]['value'] is None
    assert detail['points'][3]['connect_previous'] is False
    internal = engine.temporal(meta, np.array([0, 2, 3]), np.array([1, 2, 3]))
    assert internal['points'][0]['has_internal_gap']
    assert internal['points'][1]['connect_previous'] is False
    assert engine.temporal(meta, indices, values, datetime(2027, 1, 1), datetime(2027, 1, 2))['points'] == []


@pytest.fixture
def db(tmp_path, monkeypatch):
    from app.redundancy import service as source_service
    monkeypatch.setattr(source_service, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(service, 'data_dir', lambda: tmp_path)
    engine_db = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine_db)
    with Session(engine_db) as session:
        yield session
    engine_db.dispose()


def parent(db):
    source = create_source(db, io.BytesIO(b'time,z,a,text\n2026-01-01T00:00:00,1,,x\n2026-01-01T00:00:00,99,2,y\n2026-01-01T00:01:00,1,3,x\n2026-01-01T00:03:00,10,4,x\n'), 'characterization.csv')
    row, _ = quality.create(db, Parameters(source_id=source.id, time_column='time', selected_columns=['a', 'z', 'text'], data_types={'a': 'numeric', 'z': 'numeric', 'text': 'text'}, start_timestamp=datetime(2026, 1, 1), end_timestamp=datetime(2026, 1, 1, 0, 3), interval_seconds=60))
    quality.calculate(db, row.id)
    return row


def test_lifecycle_cache_files_and_registry_blocker(db, monkeypatch):
    from app.registry.specs import ENTITY_SPECS
    p = parent(db)
    row, created = service.start(db, p.id)
    assert created and not service.start(db, p.id)[1]
    assert ENTITY_SPECS['data_quality_analysis'].blockers(p)
    with pytest.raises(ValueError, match='characterization'):
        quality.delete_analysis(db, p.id)
    service.calculate(db, row.id)
    db.refresh(row)
    assert row.job_status == 'ready'
    assert [s['variable'] for s in row.result['summary']] == ['z', 'a', 'text']
    assert row.result['summary'][0]['mean'] == 4  # first valid duplicate wins
    folder = Path(row.artifact_path)
    assert len(list(folder.glob('*.npz'))) == 2
    assert not list(folder.glob('*.json'))  # Details remain lazy.
    detail = service.detail(row, 'z')
    assert detail['dynamics']['pair_count'] == 1
    assert detail['dynamics']['unchanged_percent'] == 100
    monkeypatch.setattr(engine, 'details', lambda *args: pytest.fail('Cached detail was recalculated'))
    assert service.detail(row, 'z') == detail
    assert service.series(row, 'z')['points'][2]['value'] is None
    assert service.start(db, p.id)[0].id == row.id
    with pytest.raises(ValueError, match='numeric'):
        service.detail(row, 'text')
    exported = service.export(row, sort='mean', descending=True)
    assert exported.splitlines()[1].startswith('z,')
    assert exported.splitlines()[-1].startswith('text,')
    assert len(service.export(row, search='a', data_type='numeric').splitlines()) == 2
    assert quality.delete_analysis(db, p.id)
    assert not folder.exists()


def test_cancel_restart_retry_and_failure(db, monkeypatch):
    p = parent(db)
    row, _ = service.start(db, p.id)
    row.cancel_requested = True
    db.commit()
    service.calculate(db, row.id)
    db.refresh(row)
    assert row.job_status == 'cancelled' and row.result is None
    assert not service.root(row.id).exists()
    assert service.retry(db, row)
    service.reconcile(db)
    db.refresh(row)
    assert row.job_status == 'failed' and row.stage == 'Interrupted'
    assert service.retry(db, row)
    def broken(*args):
        raise RuntimeError('Controlled failure')
    monkeypatch.setattr(service, 'normalized_csv', broken)
    service.calculate(db, row.id)
    db.refresh(row)
    assert row.job_status == 'failed' and row.error_message == 'Controlled failure'
    assert not service.root(row.id).exists()


def test_api_and_export_contract(db):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import get_db
    p = parent(db)
    app.dependency_overrides[get_db] = lambda: db
    try:
        c = TestClient(app)
        url = f'/api/data-quality/analyses/{p.id}/characterization'
        assert c.get(url).json() is None
        assert c.post(url).status_code == 200
        assert c.get(url).json()['job_status'] == 'ready'
        detail = c.get(url + '/detail', params={'sensor': 'a'}).json()
        assert detail['dynamics']['pair_count'] == 1
        assert c.get(url + '/series', params={'sensor': 'a', 'start': '2026-01-01T00:01:00', 'end': '2026-01-01T00:03:00'}).status_code == 200
        assert c.get(url + '/detail', params={'sensor': 'text'}).status_code == 400
        assert c.get(url + '/export', params={'search': 'text'}).text.splitlines()[1].startswith('text,text,3,')
        assert c.post(url + '/retry').status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_parallel_start_is_one_job(tmp_path, monkeypatch):
    from app.redundancy import service as source_service
    monkeypatch.setattr(source_service, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(service, 'data_dir', lambda: tmp_path)
    db_engine = create_engine(f'sqlite:///{tmp_path / "concurrent.db"}', connect_args={'check_same_thread': False, 'timeout': 10})
    Base.metadata.create_all(db_engine)
    with Session(db_engine) as db:
        p = parent(db)
        parent_id = p.id
    barrier = Barrier(2)
    def submit(_):
        with Session(db_engine) as db:
            barrier.wait()
            row, created = service.start(db, parent_id)
            return row.id, created
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    assert len({r[0] for r in results}) == 1 and sum(r[1] for r in results) == 1
    db_engine.dispose()
