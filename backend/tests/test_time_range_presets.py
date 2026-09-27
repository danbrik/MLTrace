from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from app import models
from app.database import get_db
from app.registry import service as registry
from app.time_range_presets.api import router
from tests.test_training_scheduler import make_db

BASE = "/api/time-range-presets"
VALUES = {"name": " Normalbetrieb ", "start": "2026-01-01T10:00:00", "end": "2026-01-01T11:00:00"}


@pytest.fixture
def api():
    db = make_db()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        yield client, db
    db.close()


def test_crud_and_registry(api):
    client, db = api
    response = client.post(BASE, json=VALUES)
    assert response.status_code == 201, response.text
    row = response.json()
    assert row["name"] == "Normalbetrieb"
    assert row["created_at"] and row["updated_at"]
    assert "name_key" not in row
    url = f'{BASE}/{row["id"]}'
    assert client.get(BASE).json() == [row]
    edited = client.put(url, json={**VALUES, "name": "Normal", "end": VALUES["start"]})
    assert edited.status_code == 200, edited.text
    assert edited.json()["start"] == edited.json()["end"]
    assert edited.json()["updated_at"] >= row["updated_at"]
    assert edited.json()["created_at"] == row["created_at"]
    summary = registry.registry_summary(db)
    assert next(item for item in summary["types"] if item["key"] == "time_range_preset")["count"] == 1
    listing = registry.list_registry_rows(db, "time_range_preset", search="Normal")
    assert listing["rows"][0]["name"] == "Normal"
    assert client.delete(url).status_code == 204
    assert client.get(BASE).json() == []
    assert client.delete(url).status_code == 404
    assert client.put(url, json=VALUES).status_code == 404


@pytest.mark.parametrize("changes", [
    {"name": "   "}, {"name": "x" * 256}, {"start": "2026-01-02T00:00:00"},
    {"start": "nonsense"}, {"sampling_rate": 15},
])
def test_invalid_values(api, changes):
    client, _ = api
    assert client.post(BASE, json={**VALUES, **changes}).status_code == 422
    assert client.get(BASE).json() == []


def test_casefold_uniqueness_and_local_time(api):
    client, _ = api
    first = client.post(BASE, json={**VALUES, "name": " Straße ", "start": "2026-01-01T10:00:00+02:00", "end": "2026-01-01T11:00:00Z"}).json()
    assert first["start"] == VALUES["start"]
    assert first["end"] == VALUES["end"]
    assert client.post(BASE, json={**VALUES, "name": "STRASSE"}).status_code == 409
    # Identical time values under another name are allowed.
    second = client.post(BASE, json={**VALUES, "name": "Other"}).json()
    assert client.put(f'{BASE}/{second["id"]}', json={**VALUES, "name": "strasse"}).status_code == 409
    assert len(client.get(BASE).json()) == 2
    assert client.put(f'{BASE}/{first["id"]}', json={**VALUES, "name": "STRASSE"}).status_code == 200


def test_migration_and_reopen(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "migration.db"
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "backend/alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "0061_reference_image")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    with Session(engine) as db:
        row = models.TimeRangePreset(name="Persistent", name_key="persistent", start=datetime(2026, 1, 1), end=datetime(2026, 1, 1))
        db.add(row); db.commit()
    engine.dispose()
    command.upgrade(config, "head")
    reopened = create_engine(f"sqlite:///{path}")
    with Session(reopened) as db:
        assert db.get(models.TimeRangePreset, 1).name == "Persistent"
    reopened.dispose()
    command.downgrade(config, "0061_reference_image")
    engine = create_engine(f"sqlite:///{path}")
    assert "time_range_presets" not in inspect(engine).get_table_names()
    engine.dispose()


def test_projects_and_existing_runs_are_independent(tmp_path, monkeypatch):
    from app import database, projects
    from app.main import app
    from tests.test_projects import configure_catalog
    configure_catalog(monkeypatch, tmp_path)
    projects.initialize_catalog()
    first, second = [projects.create_project(name, "Test project") for name in ("First", "Second")]
    client = TestClient(app)
    assert client.get(BASE).status_code == 400
    h1 = {"X-MLTrace-Project-ID": first.id}
    h2 = {"X-MLTrace-Project-ID": second.id}
    a = client.post(BASE, headers=h1, json=VALUES).json()
    assert client.get(BASE, headers=h2).json() == []
    b = client.post(BASE, headers=h2, json=VALUES).json()
    assert a["name"] == b["name"]
    with database.project_context(first.database_url, first.artifact_dir):
        with database.SessionLocal() as db:
            dataset = models.TrainingDataset(name="Example", usage_label="test")
            db.add(dataset); db.flush()
            frozen = {"reference": {"start": a["start"], "end": a["end"], "sampling_rate": 15}}
            run = models.ReferenceImageRun(training_dataset_id=dataset.id, training_dataset_name=dataset.name,
                config=frozen, dataset_snapshot={}, pipeline_snapshot={}, status="finished")
            db.add(run); db.commit()
    assert client.put(f'{BASE}/{a["id"]}', headers=h1, json={**VALUES, "name": "Updated", "end": VALUES["start"]}).status_code == 200
    assert client.delete(f'{BASE}/{a["id"]}', headers=h1).status_code == 204
    assert client.get(BASE, headers=h1).json() == []
    assert client.get(BASE, headers=h2).json()[0]["name"] == "Normalbetrieb"
    with database.project_context(first.database_url, first.artifact_dir):
        with database.SessionLocal() as db:
            assert db.get(models.ReferenceImageRun, 1).config == frozen
    with database.project_context(second.database_url, second.artifact_dir):
        with database.SessionLocal() as db:
            registry.delete_entities(db, [("time_range_preset", b["id"])], cascade=False)
            assert db.get(models.TimeRangePreset, b["id"]) is None
