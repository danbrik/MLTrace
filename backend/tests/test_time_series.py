"""CSV persistence and inclusive temporal partition invariants."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import TimeSeriesDataset, TimeSeriesSplit
from app.time_series.api import router
from app.time_series import service

CSV = b'time;temperature;unused\n2026-01-01T00:00:03Z;3;x\n2026-01-01T00:00:00Z;0;y\n2026-01-01T00:00:01Z;1;z\n2026-01-01T00:00:02Z;2;q\n2026-01-01T00:00:02Z;20;r\n'


@pytest.fixture
def client_db():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    @event.listens_for(engine, 'connect')
    def foreign_keys(conn, _):
        conn.execute('PRAGMA foreign_keys=ON')
    Base.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(router)
    def session():
        with Session(engine) as db:
            yield db
    app.dependency_overrides[get_db] = session
    with TestClient(app) as client:
        yield client, engine
    engine.dispose()


def upload(client, content=CSV, **changes):
    metadata = dict(name='Maschine', selected_columns=['time', 'temperature'], timestamp_column='time', timestamp_format='ISO8601')
    metadata.update(changes)
    return client.post('/api/time-series/datasets', files={'file': ('machine.csv', content, 'text/csv')}, data={'metadata': json.dumps(metadata)})


def interval(start, end, subset='train', id='a', tags=None):
    return dict(id=id, start=f'2026-01-01T00:00:0{start}Z', end=f'2026-01-01T00:00:0{end}Z', subset=subset, tags=tags or [])


def split_payload(dataset_id, intervals):
    return dict(dataset_id=dataset_id, name='Split A', tags=['Normalzustand', 'Anomalie', 'Puffer'], intervals=intervals)


def test_dataset_crud_and_persistence(client_db):
    client, engine = client_db
    preview = client.post('/api/time-series/preview', files={'file': ('test.csv', CSV)}, data={'timestamp_column': 'time'})
    assert preview.status_code == 200
    assert preview.json()['start'] == '2026-01-01T00:00:00Z'
    created = upload(client)
    assert created.status_code == 201, created.text
    dataset = created.json()
    assert dataset['columns'] == ['time', 'temperature', 'unused']
    assert dataset['rows'][0] == ['2026-01-01T00:00:03Z', '3']
    assert dataset['row_count'] == 5
    assert dataset['end'] == '2026-01-01T00:00:03Z'
    with Session(engine) as fresh_db:
        assert fresh_db.get(TimeSeriesDataset, dataset['id']).source_csv == CSV
    update = client.put(f"/api/time-series/datasets/{dataset['id']}", json={'name': 'Umbenannt', 'selected_columns': ['time', 'unused']})
    assert update.status_code == 200
    assert update.json()['rows'][0] == ['2026-01-01T00:00:03Z', 'x']
    assert client.get('/api/time-series/datasets').json()[0]['name'] == 'Umbenannt'
    assert client.delete(f"/api/time-series/datasets/{dataset['id']}").status_code == 204
    assert client.get(f"/api/time-series/datasets/{dataset['id']}").status_code == 404


def test_split_roundtrip_sorted_tags_edit_and_cascade(client_db):
    client, engine = client_db
    dataset_id = upload(client).json()['id']
    payload = split_payload(dataset_id, [interval(3, 3, 'validation', 'c', ['Normalzustand']), interval(2, 2, 'test', 'b', ['Anomalie', 'Puffer']), interval(0, 1, tags=['Normalzustand'])])
    response = client.post('/api/time-series/splits', json=payload)
    assert response.status_code == 201, response.text
    saved = response.json()
    assert [item['id'] for item in saved['intervals']] == ['a', 'b', 'c']
    assert [item['row_count'] for item in saved['intervals']] == [2, 2, 1]
    assert saved['intervals'][1]['tags'] == ['Anomalie', 'Puffer']
    assert client.get(f"/api/time-series/splits/{saved['id']}").json() == saved
    payload['name'] = 'Geändert'
    payload['intervals'] = [interval(0, 3, 'test', tags=['Puffer'])]
    changed = client.put(f"/api/time-series/splits/{saved['id']}", json=payload)
    assert changed.status_code == 200
    assert changed.json()['intervals'][0]['row_count'] == 5
    assert client.get('/api/time-series/datasets').json()[0]['split_count'] == 1
    client.delete(f'/api/time-series/datasets/{dataset_id}')
    assert client.get('/api/time-series/splits').json() == []
    with Session(engine) as db:
        assert db.scalars(select(TimeSeriesSplit)).all() == []


@pytest.mark.parametrize('ranges', [
    [interval(0, 2), interval(1, 3, 'test', 'b')],
    [interval(0, 2), interval(2, 3, 'validation', 'b')],
    [interval(0, 3), interval(1, 2, 'train', 'b')],
    [interval(1, 2), interval(1, 2, 'test', 'b')],
    [interval(0, 1), interval(1, 1, 'test', 'b')],
])
def test_overlap_rejected_across_all_groups(client_db, ranges):
    client, _ = client_db
    dataset_id = upload(client).json()['id']
    response = client.post('/api/time-series/splits', json=split_payload(dataset_id, ranges))
    assert response.status_code == 422
    assert 'überschneiden' in response.text
    assert client.get('/api/time-series/splits').json() == []


def test_rejected_edit_leaves_saved_split_unchanged_and_delete_releases_ranges(client_db):
    client, _ = client_db
    dataset_id = upload(client).json()['id']
    original = client.post('/api/time-series/splits', json=split_payload(dataset_id, [interval(0, 1), interval(2, 3, 'test', 'b')])).json()
    response = client.put(f"/api/time-series/splits/{original['id']}", json=split_payload(dataset_id, [interval(0, 2), interval(1, 3, 'test', 'b')]))
    assert response.status_code == 422
    assert client.get(f"/api/time-series/splits/{original['id']}").json() == original
    assert client.delete(f"/api/time-series/splits/{original['id']}").status_code == 204
    assert client.get('/api/time-series/splits').json() == []
    assert client.post('/api/time-series/splits', json=split_payload(dataset_id, [interval(0, 3)])).status_code == 201


@pytest.mark.parametrize('changes', [
    {'selected_columns': ['temperature', 'unused']}, {'selected_columns': ['time']},
    {'selected_columns': ['time', 'missing']}, {'selected_columns': ['time', 'time']},
    {'name': ' '}, {'timestamp_column': 'missing'},
])
def test_invalid_import_configuration(client_db, changes):
    client, _ = client_db
    assert upload(client, **changes).status_code == 422
    assert client.get('/api/time-series/datasets').json() == []


@pytest.mark.parametrize('content', [b'', b'time,x\n', b'time,time\na,b\n', b'time,\na,b\n', b'time,x\na,b,c\n', b'time,x\ninvalid,2\n', b'time,x\n,2\n', b'\xff\xff'])
def test_invalid_csv_not_persisted(client_db, content):
    client, _ = client_db
    assert upload(client, content, selected_columns=['time', 'x']).status_code == 422


def test_utf8_bom_quoted_cells_and_timestamp_formats(client_db):
    client, _ = client_db
    raw = '\ufefftime,value\r\n"20.09.2026 12:00:00","a,b"\r\n'.encode()
    result = upload(client, raw, selected_columns=['time', 'value'], timestamp_format='%d.%m.%Y %H:%M:%S')
    assert result.status_code == 201
    assert result.json()['start'] == '2026-09-20T12:00:00Z'
    assert result.json()['rows'][0][1] == 'a,b'
    result = upload(client, b'time,value\n1000000000000000001,1\n', selected_columns=['time', 'value'], timestamp_format='unix_ns')
    assert result.status_code == 201
    assert result.json()['start'] == '2001-09-09T01:46:40.000000001Z'


def test_timezone_nanoseconds_and_empty_ranges(client_db):
    client, _ = client_db
    raw = b'time,x\n2026-01-01T01:00:00.000000001+01:00,1\n2026-01-01T00:00:03.000000002Z,2\n'
    dataset_id = upload(client, raw, selected_columns=['time', 'x']).json()['id']
    payload = split_payload(dataset_id, [interval(1, 2)])
    assert 'Datenzeile' in client.post('/api/time-series/splits', json=payload).text
    payload['intervals'] = [dict(id='a', start='2026-01-01T00:00:00.000000001Z', end='2026-01-01T00:00:03.000000002Z', subset='train', tags=[])]
    assert client.post('/api/time-series/splits', json=payload).status_code == 201
    payload['intervals'][0]['start'] = '2026-01-01T00:00:00Z'
    assert client.post('/api/time-series/splits', json=payload).status_code == 422


@pytest.mark.parametrize('ranges', [
    [interval(3, 2)], [interval(0, 4)], [interval(0, 1, tags=['unknown'])],
    [interval(0, 1), interval(2, 3)], [dict(id='a', start='NaT', end='today', subset='train', tags=[])],
])
def test_invalid_ranges(client_db, ranges):
    client, _ = client_db
    dataset_id = upload(client).json()['id']
    assert client.post('/api/time-series/splits', json=split_payload(dataset_id, ranges)).status_code == 422


def test_dataset_cannot_be_switched_after_split_saved(client_db):
    client, _ = client_db
    first, second = upload(client).json()['id'], upload(client).json()['id']
    saved = client.post('/api/time-series/splits', json=split_payload(first, [interval(0, 1)])).json()
    assert client.put(f"/api/time-series/splits/{saved['id']}", json=split_payload(second, [interval(0, 1)])).status_code == 422


def test_upload_limit(client_db, monkeypatch):
    client, _ = client_db
    monkeypatch.setattr(service, 'MAX_CSV_BYTES', 10)
    assert upload(client).status_code == 413


def test_project_database_isolation(client_db):
    client, _ = client_db
    upload(client)
    other = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(other)
    def other_session():
        with Session(other) as db:
            yield db
    client.app.dependency_overrides[get_db] = other_session
    assert client.get('/api/time-series/datasets').json() == []
    assert client.get('/api/time-series/datasets/1').status_code == 404
    other.dispose()
