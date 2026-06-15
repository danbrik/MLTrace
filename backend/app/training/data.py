"""Resolve concrete image files selected by train/test dataset rules.

Dataset scanning stores compact folder metadata plus a few representative
``dataset_images`` rows for previews. Training and testing must operate on the
actual TIFF files in the selected time windows, so this resolver enumerates the
folder at runtime, parses timestamps from filenames, applies ranges/stride, and
returns deterministic records without loading image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app import models
from app.scanner import direct_tiff_files, extract_timestamp


@dataclass(frozen=True)
class ResolvedDatasetImage:
    file_path: str
    timestamp_parsed: datetime
    dataset_name: str
    dataset_root_path: str
    folder_id: int
    folder_relative_path: str
    file_name: str


def _sorted_rules(training_dataset: models.TrainingDataset) -> list[models.TrainingDatasetRule]:
    # Same deterministic ordering used by serialize_training_dataset.
    return sorted(
        training_dataset.rules,
        key=lambda rule: (
            rule.start_timestamp,
            rule.end_timestamp,
            rule.folder.dataset.name,
            rule.folder.relative_path,
            rule.id,
        ),
    )


def _folder_path(folder: models.DatasetFolder) -> Path:
    root = Path(folder.dataset.root_path).expanduser()
    if folder.relative_path == ".":
        return root
    return root / folder.relative_path


def enumerate_rule_images(rule: models.TrainingDatasetRule) -> list[ResolvedDatasetImage]:
    """Return concrete files selected by one rule, ordered by timestamp.

    This intentionally does not rely on ``dataset_images`` rows because the fast
    scanner only persists representative rows for large folders.
    """
    folder = rule.folder
    dataset = folder.dataset
    if not dataset.timestamp_regex or not dataset.timestamp_format:
        raise ValueError(f"Dataset '{dataset.name}' has no confirmed timestamp parser.")

    selected: list[ResolvedDatasetImage] = []
    for path in direct_tiff_files(_folder_path(folder)):
        try:
            _, timestamp = extract_timestamp(path.name, dataset.timestamp_regex, dataset.timestamp_format)
        except ValueError as exc:
            raise ValueError(
                f"File '{path.name}' in dataset '{dataset.name}' does not match the confirmed timestamp parser."
            ) from exc
        if rule.start_timestamp <= timestamp <= rule.end_timestamp:
            selected.append(
                ResolvedDatasetImage(
                    file_path=str(path),
                    timestamp_parsed=timestamp,
                    dataset_name=dataset.name,
                    dataset_root_path=dataset.root_path,
                    folder_id=folder.id,
                    folder_relative_path=folder.relative_path,
                    file_name=path.name,
                )
            )

    selected.sort(key=lambda image: (image.timestamp_parsed, image.file_name, image.file_path))
    return selected[:: max(1, rule.stride)]


def enumerate_training_dataset_image_records(
    training_dataset: models.TrainingDataset,
) -> list[ResolvedDatasetImage]:
    records: list[ResolvedDatasetImage] = []
    seen: set[str] = set()
    for rule in _sorted_rules(training_dataset):
        for image in enumerate_rule_images(rule):
            if image.file_path in seen:
                continue
            seen.add(image.file_path)
            records.append(image)
    return records


def enumerate_training_pipeline_image_records(
    pipeline: models.TrainingPipeline,
) -> list[ResolvedDatasetImage]:
    """Return ordered, de-duplicated image records for a training pipeline."""
    records: list[ResolvedDatasetImage] = []
    seen: set[str] = set()

    for entry in sorted(pipeline.entries, key=lambda item: item.position):
        for image in enumerate_training_dataset_image_records(entry.training_dataset):
            if image.file_path in seen:
                continue
            seen.add(image.file_path)
            records.append(image)

    return records


def enumerate_training_pipeline_images(_db, pipeline: models.TrainingPipeline) -> list[str]:
    """Compatibility wrapper returning only file paths for the training engine."""
    return [image.file_path for image in enumerate_training_pipeline_image_records(pipeline)]
