"""Background scheduler that runs queued training runs as GPU-pinned subprocesses.

A single daemon thread (started from the FastAPI lifespan) polls the
``training_runs`` table and, while fewer than ``max_concurrent_trainings`` runs
are active, launches the next queued run as a detached
``python -m app.training.worker <id>`` process with ``CUDA_VISIBLE_DEVICES``
pinned to one free GPU index. stdout/stderr go to ``.mltrace/runs/<id>/worker.log``.

The training processes are independent of the API process: if uvicorn restarts,
in-flight workers keep running and the scheduler reconciles them on startup via
the stored PIDs. Aborts are delivered as SIGTERM (the worker turns that into an
``aborted`` status); queued runs are aborted directly in the DB.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from app import models
from app.config import get_settings
from app.database import SessionLocal, data_dir

logger = logging.getLogger("mltrace.scheduler")

# backend/ directory: cwd for the worker so `python -m app.training.worker` resolves.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_POLL_INTERVAL_SECONDS = 2.0


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class TrainingScheduler:
    def __init__(self) -> None:
        self._processes: dict[int, subprocess.Popen] = {}  # run_id -> Popen
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
        self._thread = threading.Thread(target=self._loop, name="training-scheduler", daemon=True)
        self._thread.start()
        logger.info("Training scheduler started (max_concurrent=%s)", get_settings().max_concurrent_trainings)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Nudge the loop to dispatch immediately (e.g. right after enqueue)."""
        self._wake.set()

    # -- abort ---------------------------------------------------------------

    def request_abort(self, run_id: int, pid: int | None) -> None:
        """Signal a running worker to stop (SIGTERM). Safe across API restarts."""
        with self._lock:
            proc = self._processes.get(run_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                logger.warning("Could not signal pid %s for run %s", pid, run_id)

    # -- internal ------------------------------------------------------------

    def _reconcile_on_startup(self) -> None:
        """After an API restart we lost the Popen handles. Mark runs whose worker
        process is gone as failed; leave still-alive detached workers running."""
        db = SessionLocal()
        try:
            running = list(db.scalars(select(models.TrainingRun).where(models.TrainingRun.status == "running")))
            for run in running:
                if not _pid_alive(run.pid):
                    run.status = "failed"
                    run.ended_at = datetime.utcnow()
                    run.error_message = "Worker process was not running after a server restart."
            db.commit()
        finally:
            db.close()

    def _busy_gpus(self, db) -> set[int]:
        rows = db.scalars(
            select(models.TrainingRun.gpu_index).where(models.TrainingRun.status == "running")
        )
        return {index for index in rows if index is not None}

    def _free_gpu(self, busy: set[int], limit: int) -> int | None:
        for index in range(limit):
            if index not in busy:
                return index
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - the scheduler thread must never die
                logger.exception("Scheduler tick failed")
            self._wake.wait(timeout=_POLL_INTERVAL_SECONDS)
            self._wake.clear()

    def _tick(self) -> None:
        limit = max(1, get_settings().max_concurrent_trainings)
        db = SessionLocal()
        try:
            # Reap finished processes; mark crashed workers (no terminal status) failed.
            with self._lock:
                finished_ids = [rid for rid, proc in self._processes.items() if proc.poll() is not None]
                for rid in finished_ids:
                    self._processes.pop(rid, None)
            for rid in finished_ids:
                run = db.get(models.TrainingRun, rid)
                if run is not None and run.status == "running":
                    run.status = "failed"
                    run.ended_at = datetime.utcnow()
                    run.error_message = run.error_message or "Worker exited without reporting a result."
            if finished_ids:
                db.commit()

            busy = self._busy_gpus(db)
            active = len(busy)
            if active >= limit:
                return

            queued = list(
                db.scalars(
                    select(models.TrainingRun)
                    .where(models.TrainingRun.status == "queued")
                    .order_by(models.TrainingRun.enqueued_at.asc(), models.TrainingRun.id.asc())
                )
            )
            for run in queued:
                if active >= limit:
                    break
                gpu = self._free_gpu(busy, limit)
                if gpu is None:
                    break
                self._launch(db, run, gpu)
                busy.add(gpu)
                active += 1
        finally:
            db.close()

    def _launch(self, db, run: models.TrainingRun, gpu_index: int) -> None:
        artifact_dir = data_dir() / "runs" / str(run.id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_dir / "worker.log"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - handed to the child process
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "app.training.worker", str(run.id)],
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
        run.pid = proc.pid
        run.log_path = str(log_path)
        run.error_message = None
        db.commit()

        with self._lock:
            self._processes[run.id] = proc
        logger.info("Launched training run %s on GPU %s (pid %s)", run.id, gpu_index, proc.pid)


scheduler = TrainingScheduler()
