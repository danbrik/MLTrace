"""Resolve concrete image files selected by train/test dataset rules.

Dataset scanning stores compact folder metadata plus a few representative
``dataset_images`` rows for previews. Training and testing must operate on the
actual TIFF files in the selected time windows, so this resolver enumerates the
folder at runtime, parses timestamps from filenames, applies ranges/stride, and
returns deterministic records without loading image pixels.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
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


@dataclass(frozen=True)
class RuleImageCount:
    matching_images: int
    selected_images: int


TimestampIndexEntry = tuple[datetime, str, str]
FolderTimestampCache = dict[int, list[TimestampIndexEntry]]
_HIGH_SORT_SENTINEL = chr(0x10FFFF)


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


def _folder_timestamp_index(
    folder: models.DatasetFolder,
    cache: FolderTimestampCache | None = None,
) -> list[TimestampIndexEntry]:
    """Parse and sort a folder's filename timestamps once per request."""
    if cache is not None and folder.id in cache:
        return cache[folder.id]

    dataset = folder.dataset
    if not dataset.timestamp_regex or not dataset.timestamp_format:
        raise ValueError(f"Dataset '{dataset.name}' has no confirmed timestamp parser.")

    entries: list[TimestampIndexEntry] = []
    for path in direct_tiff_files(_folder_path(folder)):
        try:
            _, timestamp = extract_timestamp(path.name, dataset.timestamp_regex, dataset.timestamp_format)
        except ValueError as exc:
            raise ValueError(
                f"File '{path.name}' in dataset '{dataset.name}' does not match the confirmed timestamp parser."
            ) from exc
        entries.append((timestamp, path.name, str(path)))

    entries.sort()
    if cache is not None:
        cache[folder.id] = entries
    return entries


def _matching_folder_images(
    folder: models.DatasetFolder,
    start_timestamp: datetime,
    end_timestamp: datetime,
) -> list[ResolvedDatasetImage]:
    """Return all files in a timestamp range without opening image pixels."""
    dataset = folder.dataset
    matching: list[ResolvedDatasetImage] = []
    entries = _folder_timestamp_index(folder)
    left = bisect_left(entries, (start_timestamp, "", ""))
    right = bisect_right(entries, (end_timestamp, _HIGH_SORT_SENTINEL, _HIGH_SORT_SENTINEL))
    for timestamp, file_name, file_path in entries[left:right]:
        matching.append(
            ResolvedDatasetImage(
                file_path=file_path,
                timestamp_parsed=timestamp,
                dataset_name=dataset.name,
                dataset_root_path=dataset.root_path,
                folder_id=folder.id,
                folder_relative_path=folder.relative_path,
                file_name=file_name,
            )
        )
    return matching


def count_folder_range_images(
    folder: models.DatasetFolder,
    start_timestamp: datetime,
    end_timestamp: datetime,
    stride: int = 1,
    cache: FolderTimestampCache | None = None,
) -> RuleImageCount:
    """Count matching and selected files exactly from filenames only."""
    stride = max(1, stride)
    if (
        folder.first_timestamp is not None
        and folder.last_timestamp is not None
        and start_timestamp <= folder.first_timestamp
        and end_timestamp >= folder.last_timestamp
    ):
        matching_count = folder.image_count
        selected_count = (matching_count + stride - 1) // stride if matching_count else 0
        return RuleImageCount(matching_images=matching_count, selected_images=selected_count)

    entries = _folder_timestamp_index(folder, cache)
    left = bisect_left(entries, (start_timestamp, "", ""))
    right = bisect_right(entries, (end_timestamp, _HIGH_SORT_SENTINEL, _HIGH_SORT_SENTINEL))
    matching_count = max(0, right - left)
    selected_count = (matching_count + stride - 1) // stride if matching_count else 0
    return RuleImageCount(matching_images=matching_count, selected_images=selected_count)


def enumerate_rule_images(rule: models.TrainingDatasetRule) -> list[ResolvedDatasetImage]:
    """Return concrete files selected by one rule, ordered by timestamp.

    This intentionally does not rely on ``dataset_images`` rows because the fast
    scanner only persists representative rows for large folders.
    """
    selected = _matching_folder_images(rule.folder, rule.start_timestamp, rule.end_timestamp)
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
