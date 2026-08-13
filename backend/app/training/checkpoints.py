from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from app import models
from app.config import get_settings
from app.database import is_sqlite_lock_error, retry_session_operation

logger = logging.getLogger("mltrace.training.checkpoints")
CHECKPOINT_VERSION = 1
FAST_ANOGAN_CHECKPOINT_INTERVAL = 1_000


def _global_checkpoint_root() -> Path:
    url = make_url(get_settings().database_url)
    if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
        return Path(url.database).expanduser().resolve().parent
    return Path(".mltrace").resolve()


def _checkpoint_capacity_warning(target: Path, payload_size: int) -> str | None:
    try:
        free = shutil.disk_usage(target.parent).free
    except OSError:
        return None
    # Atomic replacement briefly needs the current checkpoint plus one new
    # temporary file. Keep a small fixed reserve for the DB/WAL and logs.
    required = max(payload_size * 2, payload_size + 256 * 1024 * 1024)
    if free < required:
        return (
            f"Low disk space for the next atomic checkpoint: "
            f"{free / 1024 / 1024:.0f} MB free, about {required / 1024 / 1024:.0f} MB recommended."
        )
    return None


def source_signature(
    configuration,
    graph,
    training_parameters: dict,
    sources: list[str],
    split_configuration: dict | None = None,
) -> str:
    source_rows = []
    # STAE clips overlap heavily. Hash each physical source only once.
    for raw_path in dict.fromkeys(sources):
        path = Path(raw_path)
        try:
            stat = path.stat()
            source_rows.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
        except OSError:
            source_rows.append([str(path.resolve()), None, None])
    payload = {
        "builder_kind": configuration.builder_kind,
        "method_graph": configuration.method_graph,
        "method_config": configuration.method_config,
        "preprocessing": graph.model_dump(mode="json"),
        "training_parameters": training_parameters,
        "split_configuration": split_configuration or {},
        "sources": source_rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state(torch) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(torch, state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def _checkpoint_write_slot(root: Path, timeout_seconds: float = 300) -> Iterator[None]:
    """Cross-process, cross-platform lock without an additional dependency."""
    lock_dir = root / ".checkpoint-write.lock"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            lock_dir.mkdir(parents=False)
            break
        except FileExistsError:
            try:
                stale = time.time() - lock_dir.stat().st_mtime > timeout_seconds * 2
                if stale:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the checkpoint write slot.")
            time.sleep(0.25)
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pt"


def _metadata_request_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint-metadata.json"


def _write_metadata_request(
    target: Path,
    signature: str,
    *,
    epoch: int | None,
    phase: str,
    iteration: int | None,
    warning: str,
) -> None:
    request = _metadata_request_path(target.parent)
    temporary = request.with_name(f".{request.name}.{os.getpid()}.tmp")
    payload = {
        "checkpoint_at": datetime.utcnow().isoformat(),
        "checkpoint_epoch": epoch,
        "checkpoint_phase": phase,
        "checkpoint_iteration": iteration,
        "checkpoint_path": str(target),
        "checkpoint_size_bytes": target.stat().st_size,
        "checkpoint_signature": signature,
        "checkpoint_warning": warning,
    }
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, request)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(torch, path: str | Path, expected_signature: str) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Training checkpoint version is not supported.")
    if checkpoint.get("signature") != expected_signature:
        raise ValueError("Training sources or configuration changed since the checkpoint was created.")
    return checkpoint


def atomic_torch_save(torch, payload: object, target: Path, *, serialize: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        if serialize:
            with _checkpoint_write_slot(_global_checkpoint_root()):
                torch.save(payload, temporary)
                os.replace(temporary, target)
        else:
            torch.save(payload, temporary)
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def persist_checkpoint_metadata(
    run_id: int,
    path: Path,
    signature: str,
    *,
    epoch: int | None,
    phase: str,
    iteration: int | None,
    warning: str | None = None,
    db=None,
) -> None:
    saved_at = datetime.utcnow()
    size = path.stat().st_size if path.exists() else None

    def operation(db):
        run = db.get(models.TrainingRun, run_id)
        if run is None:
            return
        if warning is None:
            run.checkpoint_at = saved_at
            run.checkpoint_epoch = epoch
            run.checkpoint_phase = phase
            run.checkpoint_iteration = iteration
            run.checkpoint_path = str(path)
            run.checkpoint_size_bytes = size
            run.checkpoint_signature = signature
        run.checkpoint_warning = warning

    if db is not None:
        operation(db)
        db.commit()
    else:
        retry_session_operation(operation, attempts=1)


def save_training_checkpoint(
    torch,
    run_id: int,
    run_dir: Path,
    signature: str,
    payload: dict,
    *,
    epoch: int | None,
    phase: str,
    iteration: int | None = None,
    metadata_db=None,
) -> bool:
    target = checkpoint_path(run_dir)
    complete_payload = {"version": CHECKPOINT_VERSION, "signature": signature, **payload}
    try:
        atomic_torch_save(torch, complete_payload, target, serialize=True)
        persist_checkpoint_metadata(
            run_id, target, signature, epoch=epoch, phase=phase, iteration=iteration, db=metadata_db,
        )
        _metadata_request_path(run_dir).unlink(missing_ok=True)
        capacity_warning = _checkpoint_capacity_warning(target, target.stat().st_size)
        if capacity_warning:
            logger.warning("Training run %s: %s", run_id, capacity_warning)
            persist_checkpoint_metadata(
                run_id,
                target,
                signature,
                epoch=epoch,
                phase=phase,
                iteration=iteration,
                warning=capacity_warning,
                db=metadata_db,
            )
        return True
    except Exception as exc:  # checkpointing must never destroy a training run
        message = f"Checkpoint could not be updated: {exc}"
        logger.warning("Training run %s: %s", run_id, message, exc_info=True)
        if target.is_file():
            try:
                _write_metadata_request(
                    target,
                    signature,
                    epoch=epoch,
                    phase=phase,
                    iteration=iteration,
                    warning=message,
                )
            except OSError:
                logger.warning("Could not persist checkpoint metadata sidecar for run %s", run_id, exc_info=True)
        try:
            persist_checkpoint_metadata(
                run_id, target, signature, epoch=epoch, phase=phase, iteration=iteration, warning=message,
                db=metadata_db,
            )
        except Exception:
            logger.warning("Could not persist checkpoint warning for run %s", run_id, exc_info=True)
        return False


def persist_training_progress(
    run_id: int,
    *,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
    best_val_loss: float | None,
    db=None,
) -> bool:
    """Idempotent short transaction; safe to reconstruct after a lock failure."""
    def operation(db):
        run = db.get(models.TrainingRun, run_id)
        if run is None:
            return
        metric = db.scalar(select(models.TrainingRunMetric).where(
            models.TrainingRunMetric.training_run_id == run_id,
            models.TrainingRunMetric.epoch == epoch,
        ))
        if metric is None:
            metric = models.TrainingRunMetric(training_run_id=run_id, epoch=epoch)
            db.add(metric)
        metric.train_loss = train_loss
        metric.val_loss = val_loss
        run.epochs_completed = epoch
        run.train_loss = train_loss
        run.val_loss = val_loss
        run.best_val_loss = best_val_loss

    try:
        if db is not None:
            operation(db)
            db.commit()
            return True
        # SQLite already waits for the configured 60 second busy timeout. A
        # progress sample is non-critical, so never stop GPU work after that.
        retry_session_operation(operation, attempts=1)
        return True
    except OperationalError as exc:
        if not is_sqlite_lock_error(exc):
            raise
        logger.warning(
            "Training run %s progress for epoch/iteration %s was deferred after a database lock.",
            run_id,
            epoch,
        )
        return False


def trim_metrics_after_checkpoint(run_id: int, completed: int, *, db=None) -> None:
    operation = lambda session: session.execute(delete(models.TrainingRunMetric).where(
        models.TrainingRunMetric.training_run_id == run_id,
        models.TrainingRunMetric.epoch > completed,
    ))
    if db is not None:
        operation(db)
        db.commit()
    else:
        retry_session_operation(operation)


def remove_checkpoint(path: str | Path | None) -> None:
    if path:
        checkpoint = Path(path)
        checkpoint.unlink(missing_ok=True)
        _metadata_request_path(checkpoint.parent).unlink(missing_ok=True)
