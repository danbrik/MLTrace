"""Background scheduler that runs queued jobs as GPU-pinned subprocesses.

A single daemon thread (started from the FastAPI lifespan) dispatches both
**training** and **testing** jobs from the ``training_runs`` and ``testing_runs``
tables. While fewer than ``max_concurrent_trainings`` jobs are active it launches
the next queued job (across both kinds, oldest first) as a detached
``python -m app.{training|testing}.worker <id>`` process with
``CUDA_VISIBLE_DEVICES`` pinned to one free GPU index. stdout/stderr go to
``.mltrace/{runs|testing_runs}/<id>/worker.log``.

The worker processes are independent of the API process: if uvicorn restarts,
in-flight workers keep running and the scheduler reconciles them on startup via
the stored PIDs. Aborts are delivered as SIGTERM (the worker turns that into an
``aborted`` status); queued jobs are aborted directly in the DB.
"""

from __future__ import annotations

import logging
import os
import json
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from app import models
from app.config import get_settings
from app.database import SessionLocal, data_dir

logger = logging.getLogger("mltrace.scheduler")

# backend/ directory: cwd for workers so `python -m app.*.worker` resolves.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_POLL_INTERVAL_SECONDS = 2.0

# Per job-kind configuration: ORM model, worker module, and log/artifact subdir.
_KINDS: dict[str, dict] = {
    "train": {"model": models.TrainingRun, "module": "app.training.worker", "subdir": "runs"},
    "test": {"model": models.TestingRun, "module": "app.testing.worker", "subdir": "testing_runs"},
    "heatmap": {"model": models.HeatmapRangeRun, "module": "app.heatmap.worker", "subdir": "heatmap_ranges"},
}


def _settings_path() -> Path:
    return data_dir() / "scheduler_settings.json"


def detect_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 - torch is optional and GPU discovery is best effort
        return 0


def get_scheduler_settings() -> dict:
    detected = detect_gpu_count()
    fallback_slots = max(1, get_settings().max_concurrent_trainings)
    default_slots = max(1, min(fallback_slots, detected or fallback_slots))
    raw: dict = {}
    path = _settings_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - ignore corrupt local preference files
            raw = {}
    max_slots = int(raw.get("max_gpu_slots") or default_slots)
    if detected > 0:
        max_slots = min(max(1, max_slots), detected)
    else:
        max_slots = max(1, max_slots)
    return {
        "detected_gpu_count": detected,
        "max_gpu_slots": max_slots,
        "only_gpu": bool(raw.get("only_gpu", False)),
    }


def update_scheduler_settings(max_gpu_slots: int, only_gpu: bool) -> dict:
    detected = detect_gpu_count()
    slots = int(max_gpu_slots)
    if detected > 0:
        slots = min(max(1, slots), detected)
    else:
        slots = max(1, slots)
    payload = {"max_gpu_slots": slots, "only_gpu": bool(only_gpu)}
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"detected_gpu_count": detected, **payload}


def _worker_database_url(database_url: str) -> str:
    """Return a database URL that is safe to pass to worker subprocesses.

    The API commonly runs from the repository root, but workers are launched
    with ``cwd=backend/`` so ``python -m app.*.worker`` can import the app
    package. A relative SQLite URL would therefore point at a different file in
    the child process. Resolve SQLite file paths before spawning so API and
    worker always use the same database.
    """
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return database_url

    database_path = Path(url.database).expanduser()
    if database_path.is_absolute():
        return database_url

    absolute_path = database_path.resolve()
    return url.set(database=str(absolute_path)).render_as_string(hide_password=False)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _queue_rows(db) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    for kind, spec in _KINDS.items():
        model = spec["model"]
        for run in db.scalars(select(model).where(model.status == "queued")):
            rows.append((kind, run))
    rows.sort(
        key=lambda item: (
            getattr(item[1], "queue_rank", None) is None,
            getattr(item[1], "queue_rank", None) or 0,
            getattr(item[1], "enqueued_at", None) or datetime.min,
            item[1].id,
        )
    )
    return rows


def normalize_queue_ranks(db) -> bool:
    changed = False
    for rank, (_, run) in enumerate(_queue_rows(db), start=1):
        if getattr(run, "queue_rank", None) != rank:
            run.queue_rank = rank
            changed = True
    return changed


def next_queue_rank(db) -> int:
    normalize_queue_ranks(db)
    rows = _queue_rows(db)
    return (max((int(run.queue_rank or 0) for _, run in rows), default=0) + 1)


def move_queued_job(db, kind: str, run_id: int, direction: str):
    if kind not in _KINDS:
        raise ValueError(f"Unsupported scheduler job kind: {kind}")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'.")
    model = _KINDS[kind]["model"]
    run = db.get(model, run_id)
    if run is None:
        return None
    if run.status != "queued":
        raise ValueError("Only queued scheduler jobs can be moved.")

    normalize_queue_ranks(db)
    rows = _queue_rows(db)
    index = next((idx for idx, item in enumerate(rows) if item[0] == kind and item[1].id == run_id), -1)
    if index < 0:
        raise ValueError("Queued scheduler job was not found in the queue.")
    target = index - 1 if direction == "up" else index + 1
    if target < 0 or target >= len(rows):
        raise ValueError("Scheduler job is already at the queue boundary.")

    other = rows[target][1]
    run.queue_rank, other.queue_rank = other.queue_rank, run.queue_rank
    db.commit()
    db.refresh(run)
    scheduler.wake()
    return run


class JobScheduler:
    def __init__(self) -> None:
        # Keyed by (kind, run_id).
        self._processes: dict[tuple[str, int], subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._reconcile_on_startup()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="job-scheduler", daemon=True)
        self._thread.start()
        logger.info("Job scheduler started (%s)", get_scheduler_settings())

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Nudge the loop to dispatch immediately (e.g. right after enqueue)."""
        self._wake.set()

    # -- abort ---------------------------------------------------------------

    def request_abort(self, kind: str, run_id: int, pid: int | None) -> None:
        """Signal a running worker to stop (SIGTERM). Safe across API restarts."""
        with self._lock:
            proc = self._processes.get((kind, run_id))
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                logger.warning("Could not signal pid %s for %s run %s", pid, kind, run_id)

    # -- internal ------------------------------------------------------------

    def _reconcile_on_startup(self) -> None:
        """After an API restart we lost the Popen handles. Mark jobs whose worker
        process is gone as failed; leave still-alive detached workers running."""
        db = SessionLocal()
        try:
            for spec in _KINDS.values():
                model = spec["model"]
                for run in db.scalars(select(model).where(model.status == "running")):
                    if not _pid_alive(run.pid):
                        run.status = "failed"
                        run.ended_at = datetime.utcnow()
                        run.error_message = "Worker process was not running after a server restart."
            db.commit()
        finally:
            db.close()

    def _busy_gpus(self, db) -> set[int]:
        busy: set[int] = set()
        for spec in _KINDS.values():
            model = spec["model"]
            for index in db.scalars(select(model.gpu_index).where(model.status == "running")):
                if index is not None:
                    busy.add(index)
        return busy

    def _free_gpu(self, busy: set[int], limit: int) -> int | None:
        for index in range(limit):
            if index not in busy:
                return index
        return None

    def _active_count(self, db) -> int:
        total = 0
        for spec in _KINDS.values():
            model = spec["model"]
            total += db.scalar(select(func.count()).select_from(model).where(model.status == "running")) or 0
        return total

    def _queued_jobs(self, db) -> list[tuple[str, object]]:
        """All queued jobs across kinds, in user-controlled scheduler order."""
        if normalize_queue_ranks(db):
            db.commit()
        return _queue_rows(db)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - the scheduler thread must never die
                logger.exception("Scheduler tick failed")
            self._wake.wait(timeout=_POLL_INTERVAL_SECONDS)
            self._wake.clear()

    def _tick(self) -> None:
        scheduler_settings = get_scheduler_settings()
        detected_gpus = int(scheduler_settings["detected_gpu_count"])
        only_gpu = bool(scheduler_settings["only_gpu"])
        limit = max(1, int(scheduler_settings["max_gpu_slots"]))
        db = SessionLocal()
        try:
            # Reap finished processes; mark crashed workers (no terminal status) failed.
            with self._lock:
                finished = [key for key, proc in self._processes.items() if proc.poll() is not None]
                for key in finished:
                    self._processes.pop(key, None)
            for kind, run_id in finished:
                model = _KINDS[kind]["model"]
                run = db.get(model, run_id)
                if run is not None and run.status == "running":
                    run.status = "failed"
                    run.ended_at = datetime.utcnow()
                    run.error_message = run.error_message or "Worker exited without reporting a result."
            if finished:
                db.commit()

            busy = self._busy_gpus(db)
            active = self._active_count(db)
            if active >= limit:
                return
            if only_gpu and detected_gpus <= 0:
                return

            gpu_limit = min(limit, detected_gpus) if detected_gpus > 0 else 0

            for kind, run in self._queued_jobs(db):
                if active >= limit:
                    break
                gpu = self._free_gpu(busy, gpu_limit) if gpu_limit > 0 else None
                if gpu is None and (only_gpu or detected_gpus > 0):
                    break
                self._launch(db, kind, run, gpu)
                if gpu is not None:
                    busy.add(gpu)
                active += 1
        finally:
            db.close()

    def _launch(self, db, kind: str, run, gpu_index: int | None) -> None:
        spec = _KINDS[kind]
        artifact_dir = data_dir() / spec["subdir"] / str(run.id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_dir / "worker.log"
        device_label = f"GPU:{gpu_index}" if gpu_index is not None else "CPU"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu_index is None else str(gpu_index)
        env["DATABASE_URL"] = _worker_database_url(get_settings().database_url)

        with open(log_path, "a", encoding="utf-8") as parent_log:
            parent_log.write(
                f"{datetime.utcnow().isoformat()} scheduler: launching {kind} run {run.id} "
                f"on {device_label} with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r}\n"
            )
            parent_log.flush()

        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - handed to the child process
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", spec["module"], str(run.id)],
                cwd=str(_BACKEND_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()

        run.status = "running"
        run.started_at = datetime.utcnow()
        run.gpu_index = gpu_index
        run.device = device_label
        run.pid = proc.pid
        run.log_path = str(log_path)
        run.error_message = None
        db.commit()

        with self._lock:
            self._processes[(kind, run.id)] = proc
        logger.info("Launched %s run %s on %s (pid %s)", kind, run.id, f"GPU {gpu_index}" if gpu_index is not None else "CPU", proc.pid)


scheduler = JobScheduler()
