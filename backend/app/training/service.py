"""Service layer for training runs: enqueue, list/filter, abort, restart, delete.

Run rows carry a denormalized snapshot of their pipeline (built here from
``serialize_training_pipeline``) so the overview can be filtered and sorted from
a single indexed table. Process control is delegated to the scheduler.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app import models, services
from app.database import data_dir
from app.schemas import TrainingRunMetricRead, TrainingRunRead
from app.training.scheduler import next_queue_rank, scheduler

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


def _clear_checkpoint_fields(run: models.TrainingRun) -> None:
    run.checkpoint_at = None
    run.checkpoint_epoch = None
    run.checkpoint_phase = None
    run.checkpoint_iteration = None
    run.checkpoint_path = None
    run.checkpoint_size_bytes = None
    run.checkpoint_signature = None
    run.checkpoint_warning = None


def _reset_run_for_queue(
    db: Session,
    run: models.TrainingRun,
    snapshot: dict,
    *,
    preserve_checkpoint: bool = False,
) -> None:
    run.status = "queued"
    run.enqueued_at = datetime.utcnow()
    run.queue_rank = next_queue_rank(db)
    run.started_at = None
    run.ended_at = None
    run.duration_seconds = None
    run.gpu_index = None
    run.device = None
    run.pid = None
    run.log_path = None
    run.error_message = None
    if not preserve_checkpoint:
        run.epochs_total = None
        run.epochs_completed = 0
        run.train_loss = None
        run.val_loss = None
        run.best_val_loss = None
        run.image_count = None
    run.artifact_kind = None
    run.artifact_path = None
    run.artifact_size_bytes = None
    run.artifact_signature = None
    run.next_retry_at = None
    run.restart_mode = "checkpoint" if preserve_checkpoint else None
    if not preserve_checkpoint:
        _clear_checkpoint_fields(run)
        run.resume_count = 0
        run.auto_retry_count = 0
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

    staged_run_dir: tuple[Path, Path] | None = None
    if run is None:
        run = models.TrainingRun(training_pipeline_id=pipeline_id, **snapshot)
        _reset_run_for_queue(db, run, snapshot)
        db.add(run)
    else:
        # Restart: reset the same row (one history per pipeline) and clear old state.
        db.execute(delete(models.TrainingRunMetric).where(models.TrainingRunMetric.training_run_id == run.id))
        _reset_run_for_queue(db, run, snapshot)
        run_dir = _run_dir(run.id)
        if run_dir.exists():
            staged = run_dir.with_name(f".{run_dir.name}.restart-{uuid.uuid4().hex}")
            run_dir.replace(staged)
            staged_run_dir = (run_dir, staged)

    try:
        db.commit()
    except Exception:
        db.rollback()
        if staged_run_dir is not None:
            original, staged = staged_run_dir
            if staged.exists() and not original.exists():
                staged.replace(original)
        raise
    if staged_run_dir is not None:
        shutil.rmtree(staged_run_dir[1], ignore_errors=True)
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


def restart_training_run_from_checkpoint(db: Session, run_id: int) -> TrainingRunRead | None:
    run = db.get(models.TrainingRun, run_id)
    if run is None:
        return None
    if run.status not in {"failed", "aborted", "retry_wait"}:
        raise RunConflict("Only failed, aborted, or waiting training runs can resume from a backup.")
    if run.builder_kind == "form":
        raise RunConflict("This short fit method does not create training backups.")
    if not run.checkpoint_at or not run.checkpoint_path or not run.checkpoint_signature:
        raise RunConflict("No training backup is available.")
    checkpoint = Path(run.checkpoint_path)
    if not checkpoint.is_file():
        raise RunConflict("The training backup file is missing. Restart completely.")

    pipeline = db.get(models.TrainingPipeline, run.training_pipeline_id)
    if pipeline is None:
        raise ValueError("The training pipeline no longer exists.")
    snapshot = _snapshot(db, pipeline)
    _validate_training_checkpoint(db, run, pipeline, snapshot["training_parameters"], snapshot["shuffle"])
    _reset_run_for_queue(db, run, snapshot, preserve_checkpoint=True)
    run.resume_count = int(run.resume_count or 0) + 1
    db.commit()
    db.refresh(run)
    scheduler.wake()
    return serialize_training_run(db, run)


def _validate_training_checkpoint(
    db: Session,
    run: models.TrainingRun,
    pipeline: models.TrainingPipeline,
    training_parameters: dict,
    shuffle: bool,
) -> None:
    """Recompute the same signature now; the worker validates it again."""
    from app.schemas import PreprocessingGraph
    from app.training.checkpoints import load_checkpoint, source_signature
    from app.training.data import enumerate_training_pipeline_clip_samples, enumerate_training_pipeline_images

    configuration = pipeline.method_configuration
    graph = PreprocessingGraph.model_validate(pipeline.preprocessing_pipeline.graph)
    if configuration.builder_kind == "spatiotemporal_autoencoder":
        summary = enumerate_training_pipeline_clip_samples(pipeline, configuration.method_config)
        if not summary.clips:
            raise RunConflict("The training sources no longer produce compatible clips. Restart completely.")
        sources = [
            frame.file_path
            for clip in summary.clips
            for frame in (*clip.input_frames, *clip.future_frames)
        ]
    else:
        sources = enumerate_training_pipeline_images(db, pipeline)
        if not sources:
            raise RunConflict("The training sources are no longer available. Restart completely.")
    current_signature = source_signature(
        configuration,
        graph,
        training_parameters,
        sources,
        {"shuffle": shuffle},
    )
    if current_signature != run.checkpoint_signature:
        raise RunConflict(
            "Training sources, model, preprocessing, split, or parameters changed since the backup. Restart completely."
        )
    try:
        import torch

        load_checkpoint(torch, run.checkpoint_path, current_signature)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RunConflict(f"The training backup is invalid: {exc} Restart completely.") from exc


def abort_training_run(db: Session, run_id: int) -> TrainingRunRead | None:
    run = db.get(models.TrainingRun, run_id)
    if run is None:
        return None
    if run.status in {"queued", "retry_wait"}:
        was_retry_wait = run.status == "retry_wait"
        run.status = "aborted"
        run.ended_at = datetime.utcnow()
        run.next_retry_at = None
        run.error_message = "Automatic retry cancelled by user." if was_retry_wait else "Aborted before it started."
        db.commit()
        db.refresh(run)
    elif run.status == "running":
        scheduler.request_abort("train", run.id, run.pid)
        # The worker turns SIGTERM into the terminal 'aborted' status.
    else:
        raise RunConflict("Only queued, running, or retry-wait runs can be aborted.")
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
    workspace_id = db.scalar(select(models.EvaluationModelWorkspace.id).where(
        models.EvaluationModelWorkspace.training_run_id == run_id
    ))
    if workspace_id is not None:
        raise RunConflict("Delete the model evaluation workspace before removing this training run.")
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
        queue_rank=run.queue_rank,
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
        artifact_signature=run.artifact_signature,
        checkpoint_at=run.checkpoint_at,
        checkpoint_epoch=run.checkpoint_epoch,
        checkpoint_phase=run.checkpoint_phase,
        checkpoint_iteration=run.checkpoint_iteration,
        checkpoint_path=run.checkpoint_path,
        checkpoint_size_bytes=run.checkpoint_size_bytes,
        checkpoint_signature=run.checkpoint_signature,
        checkpoint_warning=run.checkpoint_warning,
        restart_mode=run.restart_mode,
        resume_count=run.resume_count or 0,
        auto_retry_count=run.auto_retry_count or 0,
        next_retry_at=run.next_retry_at,
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
