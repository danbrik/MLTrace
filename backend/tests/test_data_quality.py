from datetime import datetime
from pathlib import Path
import io

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.data_quality.engine import analyze, heatmap, missing_runs
from app.data_quality.schemas import Parameters
from app.data_quality import service
from app.redundancy.service import create_source, delete_source


def params(**changes):
    value = dict(source_id=1, time_column='time', selected_columns=['a'], data_types={'a': 'numeric'},
                 start_timestamp='2026-01-01T00:00:00', end_timestamp='2026-01-01T00:04:00', interval_seconds=60)
    value.update(changes)
    return Parameters(**value).model_dump(mode='json')


def run(tmp_path, text, **changes):
    path = tmp_path / 'input.csv'
    path.write_text(text)
    return analyze(path, ',', params(**changes), len(text.splitlines()) - 1)


@pytest.mark.parametrize('values,expected', [(['1', '2', '3', '4', '5'], 0), (['1', '', '3', '4', '5'], 1),
    (['1', '', '', '', '5'], 3), (['', '', '', '4', '5'], 3), (['1', '2', '', '', ''], 3), (['', '', '', '', ''], 5)])
def test_gap_counts_expected_observations(tmp_path, values, expected):
    text = 'time,a\n' + ''.join(f'2026-01-01T00:0{i}:00,{v}\n' for i, v in enumerate(values))
    result, _ = run(tmp_path, text)
    assert result['quality'][0]['longest_gap_minutes'] == expected


def test_five_minute_gap_and_absent_rows(tmp_path):
    result, runs = run(tmp_path, 'time,a\n2026-01-01T00:00:00,1\n2026-01-01T00:10:00,\n2026-01-01T00:20:00,5\n',
                       interval_seconds=300, end_timestamp='2026-01-01T00:20:00')
    assert result['quality'][0]['longest_gap_minutes'] == 15
    assert result['quality'][0]['missing_percent'] == 60
    assert result['summary']['missing_timepoints'] == 2
    assert runs['a'] == [[1, 4]]


def test_duplicates_diagnostics_and_statistics(tmp_path):
    result, runs = run(tmp_path, 'time,a,b\n'
        '2026-01-01T00:00:00,,x\n2026-01-01T00:00:00,1,y\n'
        '2026-01-01T00:00:00,99,x\n2026-01-01T00:03:00,4,x\n'
        '2026-01-01T00:02:00,3,\n2026-01-01T00:01:00,2,x\n'
        '2026-01-01T00:04:00,5,x\n2026-01-01T00:01:15,900,z\ninvalid,7,z\n',
        selected_columns=['b', 'a'], data_types={'a': 'numeric', 'b': 'text'})
    summary = result['summary']
    assert summary['duplicate_timestamps'] == 2
    assert summary['non_monotone_timestamps'] == 3
    assert summary['off_grid_rows'] == summary['invalid_timestamps'] == 1
    assert summary['present_timepoints'] == 5
    a, b = result['quality']
    assert a['valid_n'] == 5
    assert a['conflict_n'] == b['conflict_n'] == 1
    assert a['std'] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert [a[k] for k in ('min', 'q01', 'median', 'q99', 'max', 'iqr')] == pytest.approx([1, 1.04, 3, 4.96, 5, 2])
    assert b['constant'] and b['unique'] == 1 and b['min'] is None
    assert heatmap(result, runs)['z'][1] == [0, 0, 1, 0, 0]


def test_origin_does_not_move_with_window(tmp_path):
    result, _ = run(tmp_path, 'time,a\n2026-01-01T00:00:00,1\n2026-01-01T00:01:00,2\n2026-01-01T00:02:00,3\n',
                    start_timestamp='2026-01-01T00:00:30', end_timestamp='2026-01-01T00:02:30')
    assert result['grid_start'] == '2026-01-01T00:01:00'
    assert result['grid_count'] == 2
    assert result['quality'][0]['valid_n'] == 2


def test_invalid_numeric_and_singleton(tmp_path):
    result, _ = run(tmp_path, 'time,a\n2026-01-01T00:00:00,-9999\n2026-01-01T00:01:00,broken\n2026-01-01T00:02:00,inf\n')
    a = result['quality'][0]
    assert a['min'] == -9999 and a['valid_n'] == 1 and a['invalid_n'] == 2
    assert a['std'] is None and not a['constant']


def test_heatmap_aggregation_preserves_short_outage():
    result = dict(grid_start='2026-01-01T00:00:00', grid_count=10001, interval_seconds=60)
    runs = {'a': [[10, 11], [5000, 6000]], 'b': [[0, 10001]]}
    view = heatmap(result, runs)
    assert len(view['x']) <= 1000 and view['aggregated']
    assert view['z'][0][0] == pytest.approx(1 / 11)
    assert all(v == 1 for v in view['z'][1])
    detail = heatmap(result, runs, datetime(2026, 1, 1, 0, 9), datetime(2026, 1, 1, 0, 11))
    assert detail['z'][0] == [0, 1, 0] and not detail['aggregated']


def test_validation(tmp_path):
    with pytest.raises(ValueError):
        Parameters(**params(selected_columns=[]))
    with pytest.raises(ValueError):
        Parameters(**params(end_timestamp='2025-01-01T00:00:00'))
    with pytest.raises(ValueError, match='no expected'):
        run(tmp_path, 'time,a\n2026-01-01T00:00:00,1\n', start_timestamp='2026-01-01T00:00:10', end_timestamp='2026-01-01T00:00:20')


@pytest.fixture
def db(tmp_path, monkeypatch):
    from app.redundancy import service as source_service
    monkeypatch.setattr(source_service, 'data_dir', lambda: tmp_path)
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_cache_lifecycle_and_shared_source_protection(db):
    source = create_source(db, io.BytesIO(b'time,a,b\n2026-01-01T00:00:00,1,x\n'), 'test.csv')
    payload = Parameters(**params(source_id=source.id, selected_columns=['b', 'a'], data_types={'a': 'numeric', 'b': 'text'}))
    row, created = service.create(db, payload)
    assert created
    other, created = service.create(db, Parameters(**{**payload.model_dump(), 'selected_columns': ['a', 'b']}))
    assert not created and row.id == other.id
    with pytest.raises(ValueError, match='used by'):
        delete_source(db, source.id)
    with pytest.raises(ValueError, match='running'):
        service.delete_analysis(db, row.id)
    service.calculate(db, row.id)
    db.refresh(row)
    assert row.job_status == 'ready' and row.progress == 1 and row.missing_runs
    assert service.lookup(db, payload).id == row.id
    changed, created = service.create(db, Parameters(**{**payload.model_dump(), 'interval_seconds': 120}))
    assert created and changed.id != row.id
    service.reconcile_interrupted(db)
    db.refresh(changed)
    assert changed.job_status == 'failed'
    assert service.retry(db, changed)
    changed.cancel_requested = True
    db.commit()
    service.calculate(db, changed.id)
    db.refresh(changed)
    assert changed.job_status == 'cancelled' and changed.result is None
    assert service.delete_analysis(db, changed.id)
    assert service.delete_analysis(db, row.id)
    assert delete_source(db, source.id)


def test_api_contract(db):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import get_db
    source = create_source(db, io.BytesIO(b'time,a\n2026-01-01T00:00:00,1\n'), 'test.csv')
    app.dependency_overrides[get_db] = lambda: db
    try:
        client = TestClient(app)
        payload = params(source_id=source.id)
        assert client.post('/api/data-quality/analyses/lookup', json=payload).json() is None
        response = client.post('/api/data-quality/analyses', json=payload)
        assert response.status_code == 200, response.text
        analysis_id = response.json()['id']
        saved = client.get(f'/api/data-quality/analyses/{analysis_id}').json()
        assert saved['job_status'] == 'ready'
        assert 'missing_runs' not in saved
        assert client.get(f'/api/data-quality/analyses/{analysis_id}/heatmap').json()['z'] == [[0, 1, 1, 1, 1]]
        assert client.post('/api/data-quality/analyses', json=payload).json()['id'] == analysis_id
        assert client.post(f'/api/data-quality/analyses/{analysis_id}/retry').status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_cache_identity_covers_all_calculation_parameters(db):
    source = create_source(db, io.BytesIO(b'time,other,a,b\n2026-01-01T00:00:00,2026-01-01T00:00:00,1,2\n'), 'test.csv')
    base = params(source_id=source.id, selected_columns=['a', 'b'], data_types={'a': 'numeric', 'b': 'numeric'})
    original = service.identity(db, Parameters(**base))[2]
    for changes in [dict(time_column='other'), dict(selected_columns=['a']), dict(data_types={'a': 'text', 'b': 'numeric'}),
                    dict(start_timestamp='2026-01-01T00:01:00'), dict(end_timestamp='2026-01-01T00:03:00'), dict(interval_seconds=120)]:
        assert service.identity(db, Parameters(**{**base, **changes}))[2] != original
    repeated = create_source(db, io.BytesIO(b'time,other,a,b\n2026-01-01T00:00:00,2026-01-01T00:00:00,1,2\n'), 'renamed.csv')
    assert repeated.id == source.id


def test_progress_cancellation_and_failure(db, tmp_path):
    from app.redundancy.engine import AnalysisCancelled
    path = tmp_path / 'input.csv'
    path.write_text('time,a\n2026-01-01T00:00:00,1\n')
    steps = []
    analyze(path, ',', params(), 1, lambda value, stage: steps.append((value, stage)))
    assert [v for v, _ in steps] == sorted(v for v, _ in steps)
    assert steps[-1][1] == 'Saving results'
    with pytest.raises(AnalysisCancelled):
        analyze(path, ',', params(), 1, cancelled=lambda: True)
    source = create_source(db, io.BytesIO(b'time,a\n2026-01-01T00:00:00,1\n'), 'failure.csv')
    row, _ = service.create(db, Parameters(**params(source_id=source.id)))
    Path(source.artifact_path).unlink()
    service.calculate(db, row.id)
    db.refresh(row)
    assert row.job_status == 'failed' and row.error_message and row.result is None


def test_registry_tracks_shared_source_and_job_blocker(db):
    from app.registry.specs import ENTITY_SPECS as SPECS
    from app.registry.service import _used_ids
    source = create_source(db, io.BytesIO(b'time,a\n2026-01-01T00:00:00,1\n'), 'registry.csv')
    row, _ = service.create(db, Parameters(**params(source_id=source.id)))
    assert source.id in _used_ids(db, 'redundancy_csv_source')
    assert SPECS['data_quality_analysis'].blockers(row)
    dependents = SPECS['redundancy_csv_source'].dependents(db, source.id)
    assert any(d.entity_type == 'data_quality_analysis' for d in dependents)


def test_concurrent_requests_share_one_job(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from app.redundancy import service as source_service
    monkeypatch.setattr(source_service, 'data_dir', lambda: tmp_path)
    engine = create_engine(f'sqlite:///{tmp_path / "concurrent.db"}', connect_args={'check_same_thread': False, 'timeout': 10})
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        source = create_source(db, io.BytesIO(b'time,a\n2026-01-01T00:00:00,1\n'), 'concurrent.csv')
        payload = Parameters(**params(source_id=source.id))
    barrier = Barrier(2)
    def submit():
        with Session(engine) as db:
            barrier.wait()
            row, created = service.create(db, payload)
            return row.id, created
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert len({id for id, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    engine.dispose()
