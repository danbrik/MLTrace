"""Scheduler-owned cancellation; never trust a stored PID on its own."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import psutil
from sqlalchemy import or_, select, update

from app import models

GRACE_SECONDS = 10
WORKER_MODULE = "app.analysis.spatial_sensitivity_worker"
logger = logging.getLogger("mltrace.scheduler")


class IdentityUncertain(Exception):
    pass


def worker_process(run, project_id: str):
    """None means the original worker is gone; uncertainty must never signal."""
    if not run.pid:
        return None
    try:
        process = psutil.Process(run.pid)
        birth = process.create_time()
        if run.process_started_at is not None and birth != run.process_started_at:
            return None  # PID reused: the original worker has exited.
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        argv = process.cmdline()
        if not any(argv[i:i + 3] == ["-m", WORKER_MODULE, str(run.id)] for i in range(len(argv))):
            raise IdentityUncertain("Worker-Modul oder Lauf-ID stimmen nicht mit dem Prozess überein.")
        if run.process_started_at is None or run.process_project_id is None:
            if process.environ().get("MLTRACE_PROJECT_ID") != project_id:
                raise IdentityUncertain("Der laufende Alt-Worker lässt sich nicht sicher diesem Projekt zuordnen.")
            run.process_started_at = birth
            run.process_project_id = project_id
        if run.process_project_id != project_id:
            raise IdentityUncertain("Die gespeicherte Prozessidentität gehört zu einem anderen Projekt.")
        return process
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as exc:
        raise IdentityUncertain("Keine Berechtigung zum Prüfen der Worker-Identität.") from exc


def log_abort(run, message: str) -> None:
    logger.info("Spatial run %s: %s", run.id, message)
    if run.log_path:
        try:
            with Path(run.log_path).open("a", encoding="utf-8") as log:
                log.write(f"{models.utc_now().isoformat()} scheduler: {message}\n")
        except OSError:
            logger.warning("Could not append spatial abort log", exc_info=True)


def reconcile_run(db, run, project_id: str, now: datetime | None = None) -> None:
    """Called on every tick and restart, including for detached workers."""
    now = now or models.utc_now()
    # Workers started before the deployment may still publish a terminal status
    # themselves. An accepted cancellation takes precedence, until actual exit.
    if run.abort_requested_at and run.pid and run.status != "running":
        run.status = "running"
        db.commit()
    try:
        process = worker_process(run, project_id)
        if process is None:
            aborted = run.abort_requested_at is not None
            message = ("Abgebrochen – Worker nach Zeitüberschreitung beendet." if run.force_killed_at
                       else "Spatial sensitivity analysis aborted by user.") if aborted else "Worker exited without reporting a result."
            # A worker may have committed its result since this tick's SELECT.
            values = dict(status="aborted" if aborted else "failed", current_step="aborted" if aborted else "failed",
                          ended_at=now, pid=None, error_message=message)
            if run.started_at:
                values["duration_seconds"] = max(0, (now - run.started_at).total_seconds())
            if aborted:
                values.update(result=None, csv_path=None, archive_path=None)
            changed = db.execute(update(models.SpatialSensitivityRun).where(
                models.SpatialSensitivityRun.id == run.id,
                models.SpatialSensitivityRun.status == "running",
                (models.SpatialSensitivityRun.abort_requested_at.is_not(None) if aborted
                 else models.SpatialSensitivityRun.abort_requested_at.is_(None)),
            ).values(**values))
            db.commit()
            if changed.rowcount:
                log_abort(run, message)
                from app.analysis.spatial_sensitivity import _cleanup_run_temporaries
                _cleanup_run_temporaries(run.id)
            return
        db.commit()  # Persist verified legacy identity before sending any signal.
        db.refresh(run)
        if run.status != "running" or run.abort_requested_at is None:
            return
        force = (now - run.abort_requested_at).total_seconds() >= GRACE_SECONDS
        # Recheck immediately before signaling; psutil also guards PID reuse.
        process = worker_process(run, project_id)
        if process is None:
            return  # Next tick confirms and finalizes the exit.
        if force:
            process.kill()
            if run.force_killed_at is None:
                run.force_killed_at = now
                log_abort(run, "SIGKILL sent after cancellation grace period.")
        else:
            process.terminate()
            if run.current_step != "abort_requested":
                log_abort(run, "SIGTERM sent; cancellation requested.")
        run.current_step = "abort_requested"
        run.error_message = None
        db.commit()
    except psutil.NoSuchProcess:
        db.rollback()  # Process exited between validation and signal; retry next tick.
    except (IdentityUncertain, psutil.AccessDenied, OSError) as exc:
        db.rollback()
        db.refresh(run)
        if run.abort_requested_at and run.status == "running":
            message = f"Abbruch angefordert, aber Prozess konnte nicht sicher beendet werden: {exc}"
            if run.error_message != message:
                log_abort(run, message)
            run.error_message = message
            db.commit()


def reconcile_project(db, project_id: str) -> None:
    for run in db.scalars(select(models.SpatialSensitivityRun).where(or_(
        models.SpatialSensitivityRun.status == "running",
        models.SpatialSensitivityRun.abort_requested_at.is_not(None) & models.SpatialSensitivityRun.pid.is_not(None),
    ))).all():
        reconcile_run(db, run, project_id)
