from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Callable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import data_dir
from app.preprocessing.pipeline import compile_pipeline
from app.schemas import (
    ImageDistributionHourlyPoint,
    ImageDistributionMetricSummary,
    ImageDistributionPeriod,
    ImageDistributionResponse,
    ImageDistributionRunCreate,
    ImageDistributionRunRead,
    PreprocessingGraph,
)
from app.training.data import ResolvedDatasetImage, enumerate_training_dataset_image_records
from app.training.scheduler import next_queue_rank, scheduler


METRIC_VERSION = "image-distribution-v1"
CSV_FIELDS = [
    "image_index",
    "timestamp",
    "relative_path",
    "mean_intensity",
    "spatial_std_intensity",
    "q95_intensity",
    "error",
]
ProgressCallback = Callable[[str, int, int | None, int, int], None]


class AbortedError(Exception):
    pass


def _cache_key(
    training_dataset: models.TrainingDataset,
    pipeline: models.PreprocessingPipeline,
    images: list[ResolvedDatasetImage],
) -> str:
    image_revision = [
        [image.file_path, image.timestamp_parsed.isoformat()]
        for image in images
    ]
    payload = {
        "version": METRIC_VERSION,
        "training_dataset_id": training_dataset.id,
        "training_dataset_updated_at": training_dataset.updated_at.isoformat() if training_dataset.updated_at else None,
        "rules": [
            [rule.id, rule.folder_id, rule.start_timestamp.isoformat(), rule.end_timestamp.isoformat(), rule.stride]
            for rule in training_dataset.rules
        ],
        "pipeline_id": pipeline.id,
        "pipeline_graph": pipeline.graph,
        "images": image_revision,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def cache_path(cache_key: str) -> Path:
    return data_dir() / "image_distribution" / f"{cache_key}.csv"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError("Unsupported cache schema")
        return list(reader)


def _floor_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def _summary(values: list[float]) -> ImageDistributionMetricSummary:
    return ImageDistributionMetricSummary(
        median=float(np.quantile(values, 0.5)),
        q25=float(np.quantile(values, 0.25)),
        q75=float(np.quantile(values, 0.75)),
    )


def _aggregate(rows: list[dict[str, str]]) -> list[ImageDistributionHourlyPoint]:
    grouped: dict[datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"mean_intensity": [], "spatial_std_intensity": [], "q95_intensity": []}
    )
    for row in rows:
        if row.get("error"):
            continue
        try:
            timestamp = datetime.fromisoformat(row["timestamp"])
            values = {key: float(row[key]) for key in grouped[_floor_hour(timestamp)]}
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values.values()):
            continue
        bucket = grouped[_floor_hour(timestamp)]
        for key, value in values.items():
            bucket[key].append(value)

    points: list[ImageDistributionHourlyPoint] = []
    for hour, metrics in sorted(grouped.items()):
        if not metrics["mean_intensity"]:
            continue
        points.append(ImageDistributionHourlyPoint(
            hour=hour,
            image_count=len(metrics["mean_intensity"]),
            mean_intensity=_summary(metrics["mean_intensity"]),
            spatial_std_intensity=_summary(metrics["spatial_std_intensity"]),
            q95_intensity=_summary(metrics["q95_intensity"]),
        ))
    return points


def _training_periods(training_dataset: models.TrainingDataset) -> list[ImageDistributionPeriod]:
    return [ImageDistributionPeriod(
        name=training_dataset.name,
        usage_label=training_dataset.usage_label,
        start=rule.start_timestamp,
        end=rule.end_timestamp,
    ) for rule in sorted(training_dataset.rules, key=lambda item: item.start_timestamp)]


def calculate(
    db: Session,
    training_dataset_id: int,
    preprocessing_pipeline_id: int,
    *,
    progress: ProgressCallback | None = None,
    abort_event: threading.Event | None = None,
) -> ImageDistributionResponse:
    progress = progress or (lambda *_: None)
    abort_event = abort_event or threading.Event()
    progress("resolving_images", 0, None, 0, 0)
    training_dataset = db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset_id)
        .options(
            selectinload(models.TrainingDataset.rules)
            .selectinload(models.TrainingDatasetRule.folder)
            .selectinload(models.DatasetFolder.dataset)
        )
    )
    if training_dataset is None:
        raise ValueError("Train/Test dataset not found.")
    pipeline = db.get(models.PreprocessingPipeline, preprocessing_pipeline_id)
    if pipeline is None:
        raise ValueError("Preprocessing pipeline not found.")

    images = enumerate_training_dataset_image_records(training_dataset)
    if not images:
        raise ValueError("Train/Test dataset selects no images.")
    if abort_event.is_set():
        raise AbortedError()

    progress("checking_cache", 0, len(images), 0, 0)
    key = _cache_key(training_dataset, pipeline, images)
    path = cache_path(key)
    cache_hit = path.is_file()
    if cache_hit:
        try:
            progress("loading_cache", 0, len(images), 0, 0)
            rows = _read_csv(path)
            if len(rows) != len(images):
                raise ValueError("Incomplete cache")
        except (OSError, ValueError, csv.Error):
            cache_hit = False

    if not cache_hit:
        progress("preparing_pipeline", 0, len(images), 0, 0)
        compiled = compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
        rows = []
        successful = 0
        failed = 0
        for index, image in enumerate(images):
            if abort_event.is_set():
                raise AbortedError()
            row = {
                "image_index": str(index),
                "timestamp": image.timestamp_parsed.isoformat(),
                "relative_path": str(Path(image.folder_relative_path) / image.file_name),
                "mean_intensity": "",
                "spatial_std_intensity": "",
                "q95_intensity": "",
                "error": "",
            }
            try:
                values = np.asarray(compiled.run(image.file_path), dtype=np.float64)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    raise ValueError("Preprocessing produced no finite pixels")
                row.update({
                    "mean_intensity": repr(float(np.mean(finite))),
                    "spatial_std_intensity": repr(float(np.std(finite, ddof=0))),
                    "q95_intensity": repr(float(np.quantile(finite, 0.95))),
                })
                successful += 1
            except Exception as exc:  # Preserve partial results and make per-image failures visible in CSV.
                row["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            rows.append(row)
            progress("processing_images", index + 1, len(images), successful, failed)
        progress("writing_csv", len(images), len(images), successful, failed)
        _write_csv(path, rows)

    failed = sum(bool(row.get("error")) for row in rows)
    progress("aggregating_hourly", len(rows), len(rows), len(rows) - failed, failed)
    return ImageDistributionResponse(
        training_dataset_id=training_dataset.id,
        training_dataset_name=training_dataset.name,
        usage_label=training_dataset.usage_label,
        preprocessing_pipeline_id=pipeline.id,
        preprocessing_pipeline_name=pipeline.name,
        cache_key=key,
        cache_hit=cache_hit,
        total_images=len(rows),
        successful_images=len(rows) - failed,
        failed_images=failed,
        hourly=_aggregate(rows),
        periods=_training_periods(training_dataset),
    )


def _serialize_run(run: models.ImageDistributionRun) -> ImageDistributionRunRead:
    return ImageDistributionRunRead.model_validate(run)


def enqueue(db: Session, payload: ImageDistributionRunCreate, *, wake_scheduler: bool = True) -> ImageDistributionRunRead:
    training_dataset = db.get(models.TrainingDataset, payload.training_dataset_id)
    if training_dataset is None:
        raise ValueError("Train/Test dataset not found.")
    pipeline = db.get(models.PreprocessingPipeline, payload.preprocessing_pipeline_id)
    if pipeline is None:
        raise ValueError("Preprocessing pipeline not found.")
    run = models.ImageDistributionRun(
        training_dataset_id=training_dataset.id,
        preprocessing_pipeline_id=pipeline.id,
        training_dataset_name=training_dataset.name,
        usage_label=training_dataset.usage_label,
        preprocessing_pipeline_name=pipeline.name,
        status="queued",
        current_step="queued",
        enqueued_at=models.utc_now(),
        queue_rank=next_queue_rank(db),
        processed_images=0,
        successful_images=0,
        failed_images=0,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    if wake_scheduler:
        scheduler.wake()
    return _serialize_run(run)


def list_runs(db: Session) -> list[ImageDistributionRunRead]:
    runs = db.scalars(
        select(models.ImageDistributionRun).order_by(models.ImageDistributionRun.created_at.desc())
    ).all()
    return [_serialize_run(run) for run in runs]


def get_run(db: Session, run_id: int) -> ImageDistributionRunRead | None:
    run = db.get(models.ImageDistributionRun, run_id)
    return _serialize_run(run) if run is not None else None


def abort_run(db: Session, run_id: int) -> ImageDistributionRunRead | None:
    run = db.get(models.ImageDistributionRun, run_id)
    if run is None:
        return None
    if run.status == "queued":
        run.status = "aborted"
        run.current_step = "aborted"
        run.ended_at = models.utc_now()
        run.error_message = "Aborted before it started."
        db.commit()
        db.refresh(run)
    elif run.status == "running":
        scheduler.request_abort("image_distribution", run.id, run.pid)
    else:
        raise ValueError("Only queued or running jobs can be aborted.")
    return _serialize_run(run)


def delete_run(db: Session, run_id: int) -> bool:
    run = db.get(models.ImageDistributionRun, run_id)
    if run is None:
        return False
    if run.status == "running":
        raise ValueError("Abort the image-distribution analysis before removing it.")
    shutil.rmtree(data_dir() / "image_distribution_runs" / str(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return True


def read_log(db: Session, run_id: int, max_lines: int = 400) -> str | None:
    run = db.get(models.ImageDistributionRun, run_id)
    if run is None:
        return None
    if not run.log_path:
        return ""
    try:
        with open(run.log_path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-max_lines:])
    except FileNotFoundError:
        return ""


def run_scheduled(run_id: int, abort_event: threading.Event | None = None) -> None:
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    from app.database import SessionLocal

    db = SessionLocal()
    last_commit = 0.0
    try:
        run = db.get(models.ImageDistributionRun, run_id)
        if run is None:
            return
        run.status = "running"
        run.current_step = "resolving_images"
        run.started_at = run.started_at or models.utc_now()
        run.device = "CPU"
        run.error_message = None
        db.commit()

        def report(step: str, processed: int, total: int | None, successful: int, failed: int) -> None:
            nonlocal last_commit
            now = time.monotonic()
            terminal_update = total is not None and processed >= total
            if step == "processing_images" and not terminal_update and now - last_commit < 0.5:
                return
            current = db.get(models.ImageDistributionRun, run_id)
            if current is None:
                raise AbortedError()
            current.current_step = step
            current.processed_images = processed
            current.total_images = total
            current.successful_images = successful
            current.failed_images = failed
            db.commit()
            last_commit = now

        try:
            result = calculate(
                db,
                run.training_dataset_id,
                run.preprocessing_pipeline_id,
                progress=report,
                abort_event=abort_event,
            )
            run = db.get(models.ImageDistributionRun, run_id)
            if run is None:
                return
            run.status = "finished"
            run.current_step = "finished"
            run.ended_at = models.utc_now()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            run.total_images = result.total_images
            run.processed_images = result.total_images
            run.successful_images = result.successful_images
            run.failed_images = result.failed_images
            run.cache_key = result.cache_key
            run.cache_hit = result.cache_hit
            run.csv_path = str(cache_path(result.cache_key))
            run.result = result.model_dump(mode="json")
            db.commit()
        except AbortedError:
            db.rollback()
            run = db.get(models.ImageDistributionRun, run_id)
            if run is not None:
                run.status = "aborted"
                run.current_step = "aborted"
                run.ended_at = models.utc_now()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = "Image-distribution analysis aborted by user."
                db.commit()
        except Exception as exc:
            db.rollback()
            run = db.get(models.ImageDistributionRun, run_id)
            if run is not None:
                run.status = "failed"
                run.current_step = "failed"
                run.ended_at = models.utc_now()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = str(exc)
                db.commit()
            raise
    finally:
        db.close()
