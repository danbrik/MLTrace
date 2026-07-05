"""Resolve concrete image files selected by train/test dataset rules.

Dataset scanning stores compact folder metadata plus a few representative
``dataset_images`` rows for previews. Training and testing must operate on the
actual TIFF files in the selected time windows, so this resolver enumerates the
folder at runtime, parses timestamps from filenames, applies ranges/stride, and
returns deterministic records without loading image pixels.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app import models
from app.database import data_dir
from app.scanner import direct_tiff_files, extract_timestamp

logger = logging.getLogger("mltrace.training.data")


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


@dataclass(frozen=True)
class ResolvedClipSample:
    input_frames: tuple[ResolvedDatasetImage, ...]
    future_frames: tuple[ResolvedDatasetImage, ...]
    score_timestamp: datetime
    clip_start: datetime
    clip_end: datetime
    dataset_name: str
    folder_id: int


@dataclass(frozen=True)
class ClipEnumerationSummary:
    clips: tuple[ResolvedClipSample, ...]
    skipped_missing: int
    selected_frame_count: int = 0
    possible_clip_count: int = 0
    sequence_contiguity_mode: str = "ordered_index"


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


def _folder_index_cache_path(folder_id: int) -> Path:
    return data_dir() / "folder_index" / f"{folder_id}.json"


def _folder_dir_signature(folder_path: Path, file_count: int) -> str:
    """Cheap fingerprint that changes when files are added/removed. Combines the
    directory mtime with the file count; avoids ``stat`` on every file."""
    try:
        mtime_ns = folder_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return f"{file_count}:{mtime_ns}"


def _load_folder_index_cache(
    folder_id: int, signature: str, folder_path: Path
) -> list[TimestampIndexEntry] | None:
    path = _folder_index_cache_path(folder_id)
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if payload.get("signature") != signature:
        return None
    try:
        # Stored compactly as [iso, name]; reconstruct the absolute path on load.
        return [
            (datetime.fromisoformat(iso), name, str(folder_path / name))
            for iso, name in payload["entries"]
        ]
    except (KeyError, ValueError, TypeError):
        return None


def _save_folder_index_cache(folder_id: int, signature: str, entries: list[TimestampIndexEntry]) -> None:
    path = _folder_index_cache_path(folder_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {"signature": signature, "entries": [[ts.isoformat(), name] for ts, name, _ in entries]},
                handle,
            )
        tmp.replace(path)
    except OSError as exc:  # noqa: BLE001 - caching is best effort; never fail enumeration
        logger.warning("Could not write folder index cache for folder %s: %s", folder_id, exc)


def _folder_timestamp_index(
    folder: models.DatasetFolder,
    cache: FolderTimestampCache | None = None,
) -> list[TimestampIndexEntry]:
    """Parse and sort a folder's filename timestamps, cached on disk.

    The expensive part is parsing a timestamp out of every filename (regex) and
    sorting — for 150k+ files that recurs on every training/inference/count.
    Results are persisted under ``data_dir()/folder_index/<id>.json`` keyed by a
    cheap directory signature (file count + dir mtime), so subsequent
    enumerations just load the prebuilt index instead of re-parsing.
    """
    if cache is not None and folder.id in cache:
        return cache[folder.id]

    dataset = folder.dataset
    if not dataset.timestamp_regex or not dataset.timestamp_format:
        raise ValueError(f"Dataset '{dataset.name}' has no confirmed timestamp parser.")

    folder_path = _folder_path(folder)
    files = direct_tiff_files(folder_path)
    signature = _folder_dir_signature(folder_path, len(files))

    entries = _load_folder_index_cache(folder.id, signature, folder_path)
    if entries is None:
        entries = []
        for path in files:
            try:
                _, timestamp = extract_timestamp(path.name, dataset.timestamp_regex, dataset.timestamp_format)
            except ValueError as exc:
                raise ValueError(
                    f"File '{path.name}' in dataset '{dataset.name}' does not match the confirmed timestamp parser."
                ) from exc
            entries.append((timestamp, path.name, str(path)))
        entries.sort()
        _save_folder_index_cache(folder.id, signature, entries)

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


def enumerate_training_dataset_image_records_for_range(
    training_dataset: models.TrainingDataset,
    start_timestamp: datetime,
    end_timestamp: datetime,
    *,
    extra_stride: int = 1,
) -> list[ResolvedDatasetImage]:
    """Resolve a clipped, globally stride-sampled selection for inspection.

    The saved rule stride is applied inside each rule first. The additional
    ``extra_stride`` is then applied after de-duplication and global timestamp
    ordering, matching the Inspect page's "make this dataset smaller" behavior.
    """
    records: list[ResolvedDatasetImage] = []
    seen: set[str] = set()
    for rule in _sorted_rules(training_dataset):
        clipped_start = max(rule.start_timestamp, start_timestamp)
        clipped_end = min(rule.end_timestamp, end_timestamp)
        if clipped_end < clipped_start:
            continue
        selected = _matching_folder_images(rule.folder, clipped_start, clipped_end)
        for image in selected[:: max(1, rule.stride)]:
            if image.file_path in seen:
                continue
            seen.add(image.file_path)
            records.append(image)

    records.sort(key=lambda image: (image.timestamp_parsed, image.file_path))
    return records[:: max(1, extra_stride)]


def enumerate_head_records_for_range(
    training_dataset: models.TrainingDataset,
    start_timestamp: datetime,
    end_timestamp: datetime,
    *,
    extra_stride: int = 1,
    limit: int,
) -> list[ResolvedDatasetImage]:
    """Resolve only the first ``limit`` selected records (globally sorted).

    Same selection semantics as :func:`enumerate_training_dataset_image_records_for_range`
    but bounded: it materializes at most ``limit * extra_stride`` records per rule
    (indexing directly into the cached timestamp index without copying the whole
    window), so previews stay fast on huge ranges. The exact total count is not
    needed here — use ``count_folder_range_images`` for that.
    """
    if limit <= 0:
        return []
    extra_stride = max(1, extra_stride)
    per_rule_cap = limit * extra_stride

    candidates: list[ResolvedDatasetImage] = []
    for rule in _sorted_rules(training_dataset):
        clipped_start = max(rule.start_timestamp, start_timestamp)
        clipped_end = min(rule.end_timestamp, end_timestamp)
        if clipped_end < clipped_start:
            continue
        folder = rule.folder
        dataset = folder.dataset
        entries = _folder_timestamp_index(folder)
        left = bisect_left(entries, (clipped_start, "", ""))
        right = bisect_right(entries, (clipped_end, _HIGH_SORT_SENTINEL, _HIGH_SORT_SENTINEL))
        step = max(1, rule.stride)
        index = left
        picked = 0
        while index < right and picked < per_rule_cap:
            timestamp, file_name, file_path = entries[index]
            candidates.append(
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
            picked += 1
            index += step

    candidates.sort(key=lambda image: (image.timestamp_parsed, image.file_path))
    seen: set[str] = set()
    deduped: list[ResolvedDatasetImage] = []
    for image in candidates:
        if image.file_path in seen:
            continue
        seen.add(image.file_path)
        deduped.append(image)
    return deduped[::extra_stride][:limit]


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


def _folder_cadence_seconds(folder: models.DatasetFolder) -> float | None:
    summary = folder.cadence_summary or {}
    for key in ("mean_seconds", "median_seconds"):
        value = summary.get(key)
        if value is not None:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return None


def _timestamps_match_stride(
    first: datetime,
    second: datetime,
    *,
    expected_steps: int,
    cadence_seconds: float | None,
) -> bool:
    if cadence_seconds is None:
        return True
    expected = cadence_seconds * expected_steps
    actual = abs((second - first).total_seconds())
    tolerance = max(0.5, cadence_seconds * 0.25)
    return abs(actual - expected) <= tolerance


def _clip_score_timestamp(
    input_frames: list[ResolvedDatasetImage],
    future_frames: list[ResolvedDatasetImage],
    mode: str,
) -> datetime:
    if mode == "first_future" and future_frames:
        return future_frames[0].timestamp_parsed
    if mode == "center_input":
        return input_frames[len(input_frames) // 2].timestamp_parsed
    return input_frames[-1].timestamp_parsed


def _sequence_contiguity_mode(value: str | None) -> str:
    return "timestamp_cadence" if value == "timestamp_cadence" else "ordered_index"


def enumerate_rule_clip_samples(
    rule: models.TrainingDatasetRule,
    *,
    clip_length: int,
    future_length: int,
    temporal_stride: int = 1,
    future_stride: int = 1,
    missing_frame_policy: str = "skip",
    score_timestamp_mode: str = "last_input",
    sequence_contiguity_mode: str = "ordered_index",
) -> ClipEnumerationSummary:
    """Build clip samples from one rule without opening image pixels.

    Clips are formed from the same selected images that ordinary image
    inference would use. Cadence metadata is used to detect gaps introduced by
    missing files; ``skip`` drops those clips while ``fail`` raises.
    """
    records = enumerate_rule_images(rule)
    clip_length = max(1, int(clip_length))
    future_length = max(0, int(future_length))
    temporal_stride = max(1, int(temporal_stride))
    future_stride = max(1, int(future_stride))
    contiguity_mode = _sequence_contiguity_mode(sequence_contiguity_mode)
    needed_span = (clip_length - 1) * temporal_stride + future_length * future_stride
    if len(records) <= needed_span:
        return ClipEnumerationSummary(
            clips=(),
            skipped_missing=0,
            selected_frame_count=len(records),
            possible_clip_count=0,
            sequence_contiguity_mode=contiguity_mode,
        )

    cadence = _folder_cadence_seconds(rule.folder)
    possible_clip_count = max(0, len(records) - needed_span)
    clips: list[ResolvedClipSample] = []
    skipped = 0
    for start in range(0, possible_clip_count):
        input_indices = [start + index * temporal_stride for index in range(clip_length)]
        last_input_index = input_indices[-1]
        future_indices = [last_input_index + (index + 1) * future_stride for index in range(future_length)]
        indices = input_indices + future_indices
        input_frames = [records[index] for index in input_indices]
        future_frames = [records[index] for index in future_indices]
        if contiguity_mode == "timestamp_cadence":
            valid = True
            for left_index, right_index in zip(indices, indices[1:]):
                if not _timestamps_match_stride(
                    records[left_index].timestamp_parsed,
                    records[right_index].timestamp_parsed,
                    expected_steps=right_index - left_index,
                    cadence_seconds=cadence,
                ):
                    valid = False
                    break
            if not valid:
                skipped += 1
                if missing_frame_policy == "fail":
                    raise ValueError(
                        "Missing frame detected while building sequence clips. "
                        f"Rule folder '{rule.folder.relative_path}' around {records[start].timestamp_parsed.isoformat()}."
                    )
                continue
        all_frames = input_frames + future_frames
        clips.append(
            ResolvedClipSample(
                input_frames=tuple(input_frames),
                future_frames=tuple(future_frames),
                score_timestamp=_clip_score_timestamp(input_frames, future_frames, score_timestamp_mode),
                clip_start=all_frames[0].timestamp_parsed,
                clip_end=all_frames[-1].timestamp_parsed,
                dataset_name=rule.folder.dataset.name,
                folder_id=rule.folder_id,
            )
        )
    return ClipEnumerationSummary(
        clips=tuple(clips),
        skipped_missing=skipped,
        selected_frame_count=len(records),
        possible_clip_count=possible_clip_count,
        sequence_contiguity_mode=contiguity_mode,
    )


def enumerate_training_dataset_clip_samples(
    training_dataset: models.TrainingDataset,
    *,
    clip_length: int,
    future_length: int,
    temporal_stride: int = 1,
    future_stride: int = 1,
    missing_frame_policy: str = "skip",
    score_timestamp_mode: str = "last_input",
    sequence_contiguity_mode: str = "ordered_index",
) -> ClipEnumerationSummary:
    clips: list[ResolvedClipSample] = []
    skipped = 0
    selected_frame_count = 0
    possible_clip_count = 0
    contiguity_mode = _sequence_contiguity_mode(sequence_contiguity_mode)
    for rule in _sorted_rules(training_dataset):
        summary = enumerate_rule_clip_samples(
            rule,
            clip_length=clip_length,
            future_length=future_length,
            temporal_stride=temporal_stride,
            future_stride=future_stride,
            missing_frame_policy=missing_frame_policy,
            score_timestamp_mode=score_timestamp_mode,
            sequence_contiguity_mode=contiguity_mode,
        )
        clips.extend(summary.clips)
        skipped += summary.skipped_missing
        selected_frame_count += summary.selected_frame_count
        possible_clip_count += summary.possible_clip_count
    clips.sort(key=lambda clip: (clip.score_timestamp, clip.dataset_name, clip.folder_id, clip.clip_start))
    return ClipEnumerationSummary(
        clips=tuple(clips),
        skipped_missing=skipped,
        selected_frame_count=selected_frame_count,
        possible_clip_count=possible_clip_count,
        sequence_contiguity_mode=contiguity_mode,
    )


def enumerate_training_pipeline_clip_samples(
    pipeline: models.TrainingPipeline,
    method_config: dict,
) -> ClipEnumerationSummary:
    clips: list[ResolvedClipSample] = []
    skipped = 0
    selected_frame_count = 0
    possible_clip_count = 0
    future_length = int(method_config.get("future_length") or 0) if method_config.get("prediction_branch") else 0
    contiguity_mode = _sequence_contiguity_mode(str(method_config.get("sequence_contiguity_mode") or "ordered_index"))
    for entry in sorted(pipeline.entries, key=lambda item: item.position):
        summary = enumerate_training_dataset_clip_samples(
            entry.training_dataset,
            clip_length=int(method_config.get("clip_length") or 1),
            future_length=future_length,
            temporal_stride=int(method_config.get("temporal_stride") or 1),
            future_stride=int(method_config.get("future_stride") or method_config.get("temporal_stride") or 1),
            missing_frame_policy=str(method_config.get("missing_frame_policy") or "skip"),
            score_timestamp_mode=str(method_config.get("score_timestamp_mode") or "last_input"),
            sequence_contiguity_mode=contiguity_mode,
        )
        clips.extend(summary.clips)
        skipped += summary.skipped_missing
        selected_frame_count += summary.selected_frame_count
        possible_clip_count += summary.possible_clip_count
    clips.sort(key=lambda clip: (clip.score_timestamp, clip.dataset_name, clip.folder_id, clip.clip_start))
    return ClipEnumerationSummary(
        clips=tuple(clips),
        skipped_missing=skipped,
        selected_frame_count=selected_frame_count,
        possible_clip_count=possible_clip_count,
        sequence_contiguity_mode=contiguity_mode,
    )
