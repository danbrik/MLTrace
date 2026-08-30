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
from sqlalchemy.orm import Session, selectinload

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
from app.training.data import ResolvedDatasetImage, enumerate_training_dataset_image_records


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


def calculate(db: Session, training_dataset_id: int, preprocessing_pipeline_id: int) -> ImageDistributionResponse:
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

    key = _cache_key(training_dataset, pipeline, images)
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
        for index, image in enumerate(images):
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
            except Exception as exc:  # Preserve partial results and make per-image failures visible in CSV.
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        _write_csv(path, rows)

    failed = sum(bool(row.get("error")) for row in rows)
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
