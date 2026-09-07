from datetime import datetime, timedelta
import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import database, models
from app.analysis import spatial_abort as cancellation, spatial_sensitivity as spatial
from app.database import Base, get_db, project_context
from app.main import app


@pytest.fixture
def factory(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(spatial.scheduler, "wake", lambda: None)
    with project_context(str(engine.url), tmp_path):
        yield factory
    engine.dispose()


def make_run(db, **kwargs):
    run = models.SpatialSensitivityRun(config={}, dataset_snapshot=[], **{
        "status": "running", "current_step": "calculating", "pid": 12345,
        "process_started_at": 100.0, "process_project_id": "project-a", **kwargs,
    })
    db.add(run); db.commit(); db.refresh(run)
    return run


def mock_process(monkeypatch, run, **overrides):
    process = Mock()
    process.create_time.return_value = 100.0
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_RUNNING
    process.cmdline.return_value = ["python", "-m", cancellation.WORKER_MODULE, str(run.id)]
    process.environ.return_value = {"MLTRACE_PROJECT_ID": "project-a"}
    for name, value in overrides.items():
        getattr(process, name).return_value = value
    monkeypatch.setattr(cancellation.psutil, "Process", lambda pid: process)
    return process


def test_abort_endpoint_idempotent_and_finished_unchanged(factory):
    with factory() as db:
        run = make_run(db)
        def override():
            with factory() as session: yield session
        app.dependency_overrides[get_db] = override
        try:
            client = TestClient(app)
            first = client.post(f"/api/spatial-sensitivity-runs/{run.id}/abort")
            second = client.post(f"/api/spatial-sensitivity-runs/{run.id}/abort")
            assert first.status_code == second.status_code == 200
            assert first.json()["abort_requested_at"] == second.json()["abort_requested_at"]
            assert first.json()["status"] == "running"
            assert client.get(f"/api/spatial-sensitivity-runs/{run.id}").json()["abort_requested_at"]
            finished = make_run(db, status="finished")
            response = client.post(f"/api/spatial-sensitivity-runs/{finished.id}/abort").json()
            assert response["status"] == "finished" and response["abort_requested_at"] is None
        finally:
            app.dependency_overrides.pop(get_db, None)


def test_queued_abort_is_immediate_and_idempotent(factory):
    with factory() as db:
        run = make_run(db, status="queued", pid=None)
        first = spatial.abort_run(db, run.id)
        second = spatial.abort_run(db, run.id)
        assert first.status == second.status == "aborted"
        assert first.abort_requested_at == second.abort_requested_at


def test_grace_period_and_cleanup_only_after_exit(factory, tmp_path, monkeypatch):
    with factory() as db:
        now = datetime(2026, 1, 1)
        run = make_run(db, abort_requested_at=now, log_path=str(tmp_path / "worker.log"))
        process = mock_process(monkeypatch, run)
        root = spatial._artifact_dir(run.id); root.mkdir(parents=True)
        (root / ".stack.dat").write_bytes(b"temporary")
        (root / "result.npy").write_bytes(b"historical")
        cancellation.reconcile_run(db, run, "project-a", now + timedelta(seconds=9))
        process.terminate.assert_called_once(); process.kill.assert_not_called()
        cancellation.reconcile_run(db, run, "project-a", now + timedelta(seconds=10))
        process.kill.assert_called_once()
        assert run.force_killed_at == now + timedelta(seconds=10)
        assert run.status == "running" and (root / ".stack.dat").exists()
        process.is_running.return_value = False
        cancellation.reconcile_run(db, run, "project-a", now + timedelta(seconds=11))
        assert run.status == "aborted" and run.pid is None
        assert not (root / ".stack.dat").exists() and (root / "result.npy").exists()
        assert "SIGKILL" in (tmp_path / "worker.log").read_text()


@pytest.mark.parametrize("mismatch", ["birth", "project", "module", "permission", "legacy_project"])
def test_identity_checks_never_signal_wrong_process(factory, monkeypatch, mismatch):
    with factory() as db:
        run = make_run(db, abort_requested_at=datetime(2020, 1, 1))
        process = mock_process(monkeypatch, run)
        if mismatch == "birth": process.create_time.return_value = 200.0
        if mismatch == "project": run.process_project_id = "other-project"
        if mismatch == "module": process.cmdline.return_value = ["unrelated", str(run.id)]
        if mismatch == "permission": process.cmdline.side_effect = psutil.AccessDenied(run.pid)
        if mismatch == "legacy_project":
            run.process_started_at = None; run.process_project_id = None
            process.environ.return_value = {"MLTRACE_PROJECT_ID": "other-project"}
        db.commit()
        cancellation.reconcile_run(db, run, "project-a")
        process.terminate.assert_not_called(); process.kill.assert_not_called()
        if mismatch == "birth": assert run.status == "aborted"
        else:
            assert run.status == "running"
            assert "nicht sicher" in run.error_message


def test_legacy_identity_adoption_and_signal_permission_failure(factory, monkeypatch):
    with factory() as db:
        run = make_run(db, process_started_at=None, process_project_id=None, abort_requested_at=datetime.now())
        process = mock_process(monkeypatch, run)
        process.terminate.side_effect = psutil.AccessDenied(run.pid)
        cancellation.reconcile_run(db, run, "project-a")
        assert run.process_started_at == 100.0 and run.process_project_id == "project-a"
        assert run.status == "running" and run.error_message


@pytest.mark.parametrize("alive", [False, True])
def test_restart_reconciles_persisted_abort(factory, tmp_path, monkeypatch, alive):
    module = importlib.import_module("app.training.scheduler")
    with factory() as db:
        run = make_run(db, abort_requested_at=models.utc_now() - timedelta(seconds=11))
        process = mock_process(monkeypatch, run, is_running=alive)
        project = SimpleNamespace(id="project-a", database_url=str(db.bind.url), artifact_dir=str(tmp_path))
        monkeypatch.setattr(module, "list_projects", lambda: [project])
        monkeypatch.setattr(module, "SessionLocal", factory)
        scheduler = type(spatial.scheduler)()
        monkeypatch.setattr(scheduler, "_apply_worker_results", lambda *a, **kw: None)
        scheduler._reconcile_on_startup()
        db.refresh(run)
        if alive:
            assert run.status == "running" and run.force_killed_at is not None
            process.kill.assert_called_once()
        else:
            assert run.status == "aborted" and run.pid is None
            process.kill.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_worker_cannot_overwrite_accepted_abort_with_result_or_error(factory, monkeypatch, tmp_path, fail):
    with factory() as db:
        run = make_run(db, pid=os.getpid())
        monkeypatch.setattr(database, "SessionLocal", factory)
        def calculate(*args):
            with factory() as other:
                spatial.abort_run(other, run.id)
            if fail: raise RuntimeError("Calculation failed concurrently")
            return {}, tmp_path / "results.csv", tmp_path / "all.zip", 1, 0
        monkeypatch.setattr(spatial, "calculate", calculate)
        if fail:
            with pytest.raises(RuntimeError): spatial.run_scheduled(run.id)
        else: spatial.run_scheduled(run.id)
        db.refresh(run)
        assert run.status == "running" and run.abort_requested_at is not None
        assert run.result is None


def test_successful_completion_before_abort_wins(factory, monkeypatch, tmp_path):
    with factory() as db:
        run = make_run(db, pid=os.getpid())
        monkeypatch.setattr(database, "SessionLocal", factory)
        monkeypatch.setattr(spatial, "calculate", lambda *args: ({"ok": True}, tmp_path / "r.csv", tmp_path / "r.zip", 1, 0))
        spatial.run_scheduled(run.id)
        result = spatial.abort_run(db, run.id)
        assert result.status == "finished" and result.abort_requested_at is None


def test_unclaimed_worker_never_cleans_another_workers_files(factory, monkeypatch):
    with factory() as db:
        run = make_run(db, status="aborted")
        monkeypatch.setattr(database, "SessionLocal", factory)
        cleanup = Mock()
        monkeypatch.setattr(spatial, "_cleanup_run_temporaries", cleanup)
        spatial.run_scheduled(run.id)
        cleanup.assert_not_called()


@pytest.mark.parametrize("ignore_term", [False, True])
def test_real_worker_cancellation(factory, tmp_path, ignore_term):
    # A disposable module with the production command shape; no application data.
    package = tmp_path / "app" / "analysis"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch(); (package / "__init__.py").touch()
    (package / "spatial_sensitivity_worker.py").write_text(
        "import signal,time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + "print('ready', flush=True)\ntime.sleep(60)\n")
    with factory() as db:
        run = make_run(db, process_started_at=None, process_project_id=None)
        proc = subprocess.Popen([sys.executable, "-m", cancellation.WORKER_MODULE, str(run.id)],
            cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path), "MLTRACE_PROJECT_ID": "project-a"},
            stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout.readline().strip() == "ready"
            run.pid = proc.pid; db.commit()
            spatial.abort_run(db, run.id)
            deadline = time.monotonic() + 15
            while proc.poll() is None and time.monotonic() < deadline:
                cancellation.reconcile_run(db, run, "project-a")
                time.sleep(0.1)
            assert proc.wait(timeout=2) == -(signal.SIGKILL if ignore_term else signal.SIGTERM)
            cancellation.reconcile_run(db, run, "project-a")
            assert run.status == "aborted"
            if ignore_term:
                assert run.force_killed_at is not None
                assert (run.force_killed_at - run.abort_requested_at).total_seconds() >= 10
            else:
                assert run.force_killed_at is None
        finally:
            if proc.poll() is None: proc.kill()
            proc.wait(timeout=5)
            proc.stdout.close()


def test_identity_rechecked_immediately_before_signal(factory, monkeypatch):
    with factory() as db:
        run = make_run(db, abort_requested_at=datetime.now())
        process = mock_process(monkeypatch, run)
        process.create_time.side_effect = [100.0, 200.0]
        cancellation.reconcile_run(db, run, "project-a")
        process.terminate.assert_not_called(); process.kill.assert_not_called()


def test_legacy_late_result_does_not_override_abort(factory, monkeypatch):
    with factory() as db:
        run = make_run(db, status="finished", result={"late": True}, abort_requested_at=datetime.now())
        assert spatial.get_run(db, run.id).status == "running"
        assert spatial.get_run(db, run.id).result is None
        assert spatial._matching_finished_run(db, run.config_signature) is None
        with pytest.raises(ValueError, match="worker to stop"):
            spatial.delete_run(db, run.id)
        process = mock_process(monkeypatch, run)
        cancellation.reconcile_project(db, "project-a")
        assert run.status == "running"
        process.is_running.return_value = False
        cancellation.reconcile_project(db, "project-a")
        assert run.status == "aborted" and run.result is None


def test_migration_preserves_existing_runs_with_null_abort_fields(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text
    backend = Path(__file__).resolve().parents[1]
    config = Config(str(backend.parent / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "alembic"))
    url = f"sqlite:///{tmp_path / 'migration.db'}"
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "0053_spatial_configs")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO spatial_sensitivity_runs (status, current_step, config, dataset_snapshot) VALUES ('finished', 'finished', '{}', '[]')"))
    command.upgrade(config, "head")
    with sessionmaker(bind=engine)() as db:
        run = db.get(models.SpatialSensitivityRun, 1)
        assert run.status == "finished"
        assert run.abort_requested_at is run.force_killed_at is run.process_started_at is run.process_project_id is None
    engine.dispose()


@pytest.mark.parametrize("cancel_during_launch", [False, True])
def test_scheduler_claim_persists_identity_without_resurrecting_cancelled_run(factory, monkeypatch, tmp_path, cancel_during_launch):
    module = importlib.import_module("app.training.scheduler")
    with factory() as db:
        run = make_run(db, status="queued", pid=None, process_started_at=None, process_project_id=None)
        project = SimpleNamespace(id="project-a", database_url=str(db.bind.url), artifact_dir=str(tmp_path))
        child = Mock(pid=12345)
        child.poll.return_value = None
        mock_process(monkeypatch, run)
        def popen(*args, **kwargs):
            if cancel_during_launch:
                with factory() as other:
                    spatial.abort_run(other, run.id)
            return child
        monkeypatch.setattr(module.subprocess, "Popen", popen)
        monkeypatch.setattr(module, "remove_queue_entry", lambda *a: None)
        scheduler = type(spatial.scheduler)()
        scheduler._launch(db, project, "spatial_sensitivity", run, None)
        db.refresh(run)
        if cancel_during_launch:
            assert run.status == "aborted" and run.pid is None
            child.kill.assert_called_once()
            assert not scheduler._processes
        else:
            assert run.status == "running" and run.pid == child.pid
            assert run.process_started_at == 100.0 and run.process_project_id == project.id
            child.kill.assert_not_called()
            assert (project.id, "spatial_sensitivity", run.id) in scheduler._processes
