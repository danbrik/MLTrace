from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.database import data_dir
from app.preprocessing.pipeline import compile_pipeline
from app.schemas import (
    ImageDistributionHourlyPoint,
    ImageDistributionMetricSummary,
    ImageDistributionPeriod,
    ImageDistributionResponse,
    PreprocessingGraph,
)


METRIC_VERSION = "image-distribution-v1"
CSV_FIELDS = [
    "image_id",
    "timestamp",
    "relative_path",
    "mean_intensity",
    "spatial_std_intensity",
    "q95_intensity",
    "error",
]


def _cache_key(dataset: models.Dataset, pipeline: models.PreprocessingPipeline, images: list[models.DatasetImage]) -> str:
    image_revision = [
        [image.id, image.timestamp_parsed.isoformat(), image.file_size_bytes, image.modified_time.isoformat() if image.modified_time else None]
        for image in images
    ]
    payload = {
        "version": METRIC_VERSION,
        "dataset_id": dataset.id,
        "dataset_updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
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


def _training_periods(db: Session, dataset_id: int) -> list[ImageDistributionPeriod]:
    rows = db.execute(
        select(models.TrainingDatasetRule, models.TrainingDataset)
        .join(models.TrainingDataset, models.TrainingDataset.id == models.TrainingDatasetRule.training_dataset_id)
        .join(models.DatasetFolder, models.DatasetFolder.id == models.TrainingDatasetRule.folder_id)
        .where(models.DatasetFolder.dataset_id == dataset_id)
        .order_by(models.TrainingDatasetRule.start_timestamp)
    ).all()
    return [ImageDistributionPeriod(
        name=training_dataset.name,
        usage_label=training_dataset.usage_label,
        start=rule.start_timestamp,
        end=rule.end_timestamp,
    ) for rule, training_dataset in rows]


def calculate(db: Session, dataset_id: int, preprocessing_pipeline_id: int) -> ImageDistributionResponse:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise ValueError("Dataset not found.")
    if dataset.status != "ready":
        raise ValueError("Dataset must be scanned and ready before analysis.")
    pipeline = db.get(models.PreprocessingPipeline, preprocessing_pipeline_id)
    if pipeline is None:
        raise ValueError("Preprocessing pipeline not found.")

    images = list(db.scalars(
        select(models.DatasetImage)
        .where(models.DatasetImage.dataset_id == dataset_id)
        .order_by(models.DatasetImage.timestamp_parsed, models.DatasetImage.id)
    ))
    if not images:
        raise ValueError("Dataset contains no indexed images.")

    key = _cache_key(dataset, pipeline, images)
    path = cache_path(key)
    cache_hit = path.is_file()
    if cache_hit:
        try:
            rows = _read_csv(path)
            if len(rows) != len(images):
                raise ValueError("Incomplete cache")
        except (OSError, ValueError, csv.Error):
            cache_hit = False

    if not cache_hit:
        compiled = compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
        rows = []
        for image in images:
            row = {
                "image_id": str(image.id),
                "timestamp": image.timestamp_parsed.isoformat(),
                "relative_path": image.relative_path,
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
            except Exception as exc:  # Preserve partial results and make per-image failures visible in CSV.
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        _write_csv(path, rows)

    failed = sum(bool(row.get("error")) for row in rows)
    return ImageDistributionResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        preprocessing_pipeline_id=pipeline.id,
        preprocessing_pipeline_name=pipeline.name,
        cache_key=key,
        cache_hit=cache_hit,
        total_images=len(rows),
        successful_images=len(rows) - failed,
        failed_images=failed,
        hourly=_aggregate(rows),
        periods=_training_periods(db, dataset.id),
    )
