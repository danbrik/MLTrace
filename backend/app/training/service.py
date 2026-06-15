"""Service layer for training runs: enqueue, list/filter, abort, restart, delete.

Run rows carry a denormalized snapshot of their pipeline (built here from
``serialize_training_pipeline``) so the overview can be filtered and sorted from
a single indexed table. Process control is delegated to the scheduler.
"""

from __future__ import annotations

import shutil
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app import models, services
from app.database import data_dir
from app.schemas import TrainingRunMetricRead, TrainingRunRead
from app.training.scheduler import scheduler

ACTIVE_STATUSES = {"queued", "running"}


class RunConflict(Exception):
    """Raised when an action conflicts with the run's current state (HTTP 409)."""


def _run_dir(run_id: int):
    return data_dir() / "runs" / str(run_id)


def _coerce_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _snapshot(db: Session, pipeline: models.TrainingPipeline) -> dict:
    read = services.serialize_training_pipeline(db, pipeline)
    dataset_names = [entry.name for entry in read.training_datasets]
    input_resolution = None
    if read.preprocessing_output_width and read.preprocessing_output_height:
        input_resolution = f"{read.preprocessing_output_width}x{read.preprocessing_output_height}"
    params = read.training_parameters or {}
    epochs = params.get("epochs")
    return {
        "training_pipeline_name": read.name,
        "method_type": read.method_type,
        "method_family": pipeline.method_configuration.method_family,
        "training_mode": read.training_mode,
        "builder_kind": read.builder_kind,
        "preprocessing_pipeline_name": read.preprocessing_pipeline_name,
        "dataset_names": dataset_names,
        "dataset_names_text": ", ".join(dataset_names),
        "shuffle": read.shuffle,
        "input_resolution": input_resolution,
        "epochs": epochs if isinstance(epochs, int) and not isinstance(epochs, bool) else None,
        "learning_rate": _coerce_number(params.get("learning_rate")),
        "training_parameters": params,
    }


def _reset_run_for_queue(run: models.TrainingRun, snapshot: dict) -> None:
    run.status = "queued"
    run.enqueued_at = datetime.utcnow()
    run.started_at = None
    run.ended_at = None
    run.duration_seconds = None
    run.gpu_index = None
    run.device = None
    run.pid = None
    run.log_path = None
    run.error_message = None
    run.epochs_total = None
    run.epochs_completed = 0
    run.train_loss = None
    run.val_loss = None
    run.best_val_loss = None
    run.image_count = None
    run.artifact_kind = None
    run.artifact_path = None
    run.artifact_size_bytes = None
    for key, value in snapshot.items():
        setattr(run, key, value)


def enqueue_training_run(db: Session, pipeline_id: int) -> TrainingRunRead:
    pipeline = db.get(models.TrainingPipeline, pipeline_id)
    if pipeline is None:
        raise ValueError(f"Training pipeline does not exist: {pipeline_id}")
    snapshot = _snapshot(db, pipeline)

    run = db.scalar(
        select(models.TrainingRun).where(models.TrainingRun.training_pipeline_id == pipeline_id)
    )
    if run is not None and run.status in ACTIVE_STATUSES:
        raise RunConflict("This training pipeline already has a queued or running run.")

    if run is None:
        run = models.TrainingRun(training_pipeline_id=pipeline_id, **snapshot)
        _reset_run_for_queue(run, snapshot)
        db.add(run)
    else:
        # Restart: reset the same row (one history per pipeline) and clear old state.
        db.execute(delete(models.TrainingRunMetric).where(models.TrainingRunMetric.training_run_id == run.id))
        _reset_run_for_queue(run, snapshot)
        shutil.rmtree(_run_dir(run.id), ignore_errors=True)

    db.commit()
    db.refresh(run)
    scheduler.wake()
    return serialize_training_run(db, run)


def restart_training_run(db: Session, run_id: int) -> TrainingRunRead | None:
    run = db.get(models.TrainingRun, run_id)
    if run is None:
        return None
    if run.status in ACTIVE_STATUSES:
        raise RunConflict("Run is already queued or running.")
    return enqueue_training_run(db, run.training_pipeline_id)


def abort_training_run(db: Session, run_id: int) -> TrainingRunRead | None:
    run = db.get(models.TrainingRun, run_id)
    if run is None:
        return None
    if run.status == "queued":
        run.status = "aborted"
        run.ended_at = datetime.utcnow()
        run.error_message = "Aborted before it started."
        db.commit()
        db.refresh(run)
    elif run.status == "running":
        scheduler.request_abort("train", run.id, run.pid)
        # The worker turns SIGTERM into the terminal 'aborted' status.
    else:
        raise RunConflict("Only queued or running runs can be aborted.")
    return serialize_training_run(db, run)


def delete_training_run(db: Session, run_id: int) -> bool:
    run = db.get(models.TrainingRun, run_id)
    if run is None:
        return False
    if run.status == "running":
        raise RunConflict("Abort the run before removing it.")
    testing_run_id = db.scalar(select(models.TestingRun.id).where(models.TestingRun.training_run_id == run_id))
    if testing_run_id is not None:
        raise RunConflict("Delete testing runs for this training run before removing it.")
    shutil.rmtree(_run_dir(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return True


def _query_with_metrics():
    return select(models.TrainingRun).options(selectinload(models.TrainingRun.metrics))


def list_training_runs(
    db: Session,
    *,
    status: str | None = None,
    method_type: str | None = None,
    training_mode: str | None = None,
    builder_kind: str | None = None,
    search: str | None = None,
    min_val_loss: float | None = None,
    max_val_loss: float | None = None,
    min_train_loss: float | None = None,
    max_train_loss: float | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> list[TrainingRunRead]:
    query = _query_with_metrics()
    if status:
        query = query.where(models.TrainingRun.status == status)
    if method_type:
        query = query.where(models.TrainingRun.method_type == method_type)
    if training_mode:
        query = query.where(models.TrainingRun.training_mode == training_mode)
    if builder_kind:
        query = query.where(models.TrainingRun.builder_kind == builder_kind)
    if search:
        like = f"%{search.lower()}%"
        query = query.where(
            (models.TrainingRun.training_pipeline_name.ilike(like))
            | (models.TrainingRun.dataset_names_text.ilike(like))
        )
    if min_val_loss is not None:
        query = query.where(models.TrainingRun.val_loss >= min_val_loss)
    if max_val_loss is not None:
        query = query.where(models.TrainingRun.val_loss <= max_val_loss)
    if min_train_loss is not None:
        query = query.where(models.TrainingRun.train_loss >= min_train_loss)
    if max_train_loss is not None:
        query = query.where(models.TrainingRun.train_loss <= max_train_loss)
    if min_duration is not None:
        query = query.where(models.TrainingRun.duration_seconds >= min_duration)
    if max_duration is not None:
        query = query.where(models.TrainingRun.duration_seconds <= max_duration)

    query = query.order_by(models.TrainingRun.created_at.desc())
    runs = list(db.scalars(query))
    return [serialize_training_run(db, run) for run in runs]


def get_training_run(db: Session, run_id: int) -> TrainingRunRead | None:
    run = db.scalar(_query_with_metrics().where(models.TrainingRun.id == run_id))
    if run is None:
        return None
    return serialize_training_run(db, run)


def read_run_log(db: Session, run_id: int, max_lines: int = 400) -> str | None:
    run = db.get(models.TrainingRun, run_id)
    if run is None:
        return None
    if not run.log_path:
        return ""
    try:
        with open(run.log_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return ""
    return "".join(lines[-max_lines:])


def serialize_training_run(db: Session, run: models.TrainingRun) -> TrainingRunRead:
    return TrainingRunRead(
        id=run.id,
        training_pipeline_id=run.training_pipeline_id,
        status=run.status,
        enqueued_at=run.enqueued_at,
        started_at=run.started_at,
        ended_at=run.ended_at,
        duration_seconds=run.duration_seconds,
        gpu_index=run.gpu_index,
        device=run.device,
        epochs_total=run.epochs_total,
        epochs_completed=run.epochs_completed,
        train_loss=run.train_loss,
        val_loss=run.val_loss,
        best_val_loss=run.best_val_loss,
        image_count=run.image_count,
        artifact_kind=run.artifact_kind,
        artifact_path=run.artifact_path,
        artifact_size_bytes=run.artifact_size_bytes,
        error_message=run.error_message,
        training_pipeline_name=run.training_pipeline_name,
        method_type=run.method_type,
        method_family=run.method_family,
        training_mode=run.training_mode,
        builder_kind=run.builder_kind,
        preprocessing_pipeline_name=run.preprocessing_pipeline_name,
        dataset_names=list(run.dataset_names or []),
        shuffle=run.shuffle,
        input_resolution=run.input_resolution,
        epochs=run.epochs,
        learning_rate=run.learning_rate,
        training_parameters=run.training_parameters or {},
        created_at=run.created_at,
        updated_at=run.updated_at,
        metrics=[
            TrainingRunMetricRead(epoch=metric.epoch, train_loss=metric.train_loss, val_loss=metric.val_loss)
            for metric in sorted(run.metrics, key=lambda item: item.epoch)
        ],
    )
