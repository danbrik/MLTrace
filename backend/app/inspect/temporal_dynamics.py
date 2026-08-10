"""Training-free temporal dynamics analysis for the Inspect workspace.

The analyzer consumes the same Train/Test dataset rules and compiled
preprocessing pipeline as the other Inspect diagnostics. Frames are decoded
once, normalized to a stable intensity scale, and reused for every requested
lag, the motion signal, autocorrelation, and visual comparison examples.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.inspect.diagnostics import roi_union_mask
from app.metrics.ssim import ssim_distance_map_np
from app.preprocessing.pipeline import absolute_image_to_uint8, compile_pipeline, encode_absolute_image_data_url
from app.schemas import PreprocessingGraph, TemporalDynamicsRequest, TemporalDynamicsResponse
from app.training.data import enumerate_training_dataset_image_records_for_range

_CACHE_MAX_ENTRIES = 8
_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_cache_lock = threading.Lock()
_DECODE_WORKERS = min(8, os.cpu_count() or 1)


def _load_training_dataset(db: Session, training_dataset_id: int) -> models.TrainingDataset | None:
    return db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset_id)
        .options(
            selectinload(models.TrainingDataset.rules)
            .selectinload(models.TrainingDatasetRule.folder)
            .selectinload(models.DatasetFolder.dataset)
        )
    )


def _stable_cache_key(
    payload: TemporalDynamicsRequest,
    training_dataset: models.TrainingDataset,
    preprocessing_pipeline: models.PreprocessingPipeline,
    roi: models.RoiDefinition | None,
    records,
) -> str:
    signature = {
        "request": payload.model_dump(mode="json"),
        "dataset_updated_at": str(training_dataset.updated_at),
        "pipeline_updated_at": str(preprocessing_pipeline.updated_at),
        "pipeline_graph": preprocessing_pipeline.graph,
        "dataset_rules": [
            {
                "folder_id": rule.folder_id,
                "root": rule.folder.dataset.root_path,
                "relative_path": rule.folder.relative_path,
                "start": rule.start_timestamp.isoformat(),
                "end": rule.end_timestamp.isoformat(),
                "stride": rule.stride,
            }
            for rule in training_dataset.rules
        ],
        "roi_updated_at": str(roi.updated_at) if roi else None,
        "records": [
            (record.file_path, record.timestamp_parsed.isoformat())
            for record in records
        ],
    }
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    with _cache_lock:
        value = _cache.get(key)
        if value is None:
            return None
        _cache.move_to_end(key)
        return {**value, "cached": True}


def _cache_put(key: str, value: dict[str, Any]) -> None:
    with _cache_lock:
        _cache[key] = {**value, "cached": False}
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)


def _prepare_frame(compiled, record):
    try:
        processed = compiled.run(record.file_path)
        frame = absolute_image_to_uint8(processed)
        original_shape = frame.shape
        return record, frame, original_shape, None
    except Exception as exc:  # unreadable/missing files are reported and skipped
        return record, None, None, str(exc)


def _effective_cadence_by_folder(
    training_dataset: models.TrainingDataset,
    extra_stride: int = 1,
) -> dict[int, float]:
    candidates: dict[int, list[float]] = defaultdict(list)
    for rule in training_dataset.rules:
        summary = rule.folder.cadence_summary or {}
        raw = summary.get("median_seconds", summary.get("mean_seconds"))
        try:
            cadence = float(raw)
        except (TypeError, ValueError):
            continue
        if cadence > 0:
            candidates[rule.folder_id].append(
                cadence * max(1, int(rule.stride)) * max(1, int(extra_stride))
            )
    return {folder_id: min(values) for folder_id, values in candidates.items() if values}


def _build_segments(records, cadence_by_folder: dict[int, float]):
    by_folder: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_folder[int(record.folder_id)].append(index)

    segments: list[list[int]] = []
    segment_cadences: list[float] = []
    for folder_id, indices in by_folder.items():
        indices.sort(key=lambda index: records[index].timestamp_parsed)
        positive_deltas = [
            (records[right].timestamp_parsed - records[left].timestamp_parsed).total_seconds()
            for left, right in zip(indices, indices[1:])
            if records[right].timestamp_parsed > records[left].timestamp_parsed
        ]
        cadence = cadence_by_folder.get(folder_id)
        if cadence is None:
            # The platform primarily handles ~1 Hz imagery. With too little
            # evidence, defaulting to one second is safer than interpreting a
            # single large missing-frame gap as the normal cadence.
            cadence = float(np.percentile(positive_deltas, 25)) if len(positive_deltas) >= 3 else 1.0
        cadence = max(0.001, float(cadence))
        gap_limit = max(cadence + 0.5, cadence * 1.5)
        current = [indices[0]] if indices else []
        for left, right in zip(indices, indices[1:]):
            delta = (records[right].timestamp_parsed - records[left].timestamp_parsed).total_seconds()
            if delta <= 0 or delta > gap_limit:
                if current:
                    segments.append(current)
                    segment_cadences.append(cadence)
                current = [right]
            else:
                current.append(right)
        if current:
            segments.append(current)
            segment_cadences.append(cadence)
    return segments, segment_cadences


def _pairs_for_lag(records, segments, segment_cadences, lag_seconds: int):
    pairs: list[tuple[int, int, int]] = []
    for segment_id, (indices, cadence) in enumerate(zip(segments, segment_cadences, strict=True)):
        if len(indices) < 2:
            continue
        timestamps = [records[index].timestamp_parsed.timestamp() for index in indices]
        tolerance = max(0.25, min(0.5, cadence * 0.25))
        for local_left, left_index in enumerate(indices):
            target = timestamps[local_left] + float(lag_seconds)
            pos = bisect_left(timestamps, target, lo=local_left + 1)
            candidates = [candidate for candidate in (pos - 1, pos) if local_left < candidate < len(indices)]
            if not candidates:
                continue
            local_right = min(candidates, key=lambda candidate: abs(timestamps[candidate] - target))
            if abs(timestamps[local_right] - target) <= tolerance:
                pairs.append((left_index, indices[local_right], segment_id))
    return pairs


def _distance(left: np.ndarray, right: np.ndarray, metric: str, mask: np.ndarray) -> float:
    left_float = left.astype(np.float32) / 255.0
    right_float = right.astype(np.float32) / 255.0
    if metric == "mse":
        values = np.square(left_float - right_float)
    elif metric == "ssim":
        values, _ = ssim_distance_map_np(left_float, right_float, {"ssim_data_range": 1.0})
    else:
        values = np.abs(left_float - right_float)
    selected = values[mask]
    return float(np.mean(selected)) if selected.size else float("nan")


def _summary(lag_seconds: int, values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return {
        "lag_seconds": lag_seconds,
        "pair_count": int(finite.size),
        "mean": None if finite.size == 0 else float(np.mean(finite)),
        "median": None if finite.size == 0 else float(np.median(finite)),
        "std": None if finite.size == 0 else float(np.std(finite)),
        "p25": None if finite.size == 0 else float(np.percentile(finite, 25)),
        "p75": None if finite.size == 0 else float(np.percentile(finite, 75)),
    }


def _motion_signal(records, frames, segments, mask, progress_step=None, abort_event=None):
    public_rows: list[dict[str, Any]] = []
    internal_rows: list[tuple[int, float, float]] = []
    for segment_id, indices in enumerate(segments):
        for left, right in zip(indices, indices[1:]):
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("aborted")
            interval = (records[right].timestamp_parsed - records[left].timestamp_parsed).total_seconds()
            value = _distance(frames[left], frames[right], "mae", mask)
            public_rows.append({
                "timestamp": records[right].timestamp_parsed,
                "difference": value,
                "interval_seconds": interval,
                "segment_id": segment_id,
            })
            internal_rows.append((segment_id, records[right].timestamp_parsed.timestamp(), value))
            if progress_step is not None:
                progress_step()
    order = np.argsort([row["timestamp"].timestamp() for row in public_rows]) if public_rows else []
    public_rows = [public_rows[int(index)] for index in order]
    return public_rows, internal_rows


def _autocorrelation(motion_rows, max_lag: int, progress_step=None, abort_event=None):
    by_segment: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for segment_id, timestamp, value in motion_rows:
        by_segment[segment_id].append((timestamp, value))
    for rows in by_segment.values():
        rows.sort()
    output = []
    for lag in range(1, max_lag + 1):
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("aborted")
        left_values: list[float] = []
        right_values: list[float] = []
        for rows in by_segment.values():
            timestamps = [row[0] for row in rows]
            tolerance = 0.5
            for index, (_, left_value) in enumerate(rows):
                target = timestamps[index] + lag
                pos = bisect_left(timestamps, target, lo=index + 1)
                candidates = [candidate for candidate in (pos - 1, pos) if index < candidate < len(rows)]
                if not candidates:
                    continue
                right_index = min(candidates, key=lambda candidate: abs(timestamps[candidate] - target))
                if abs(timestamps[right_index] - target) <= tolerance:
                    left_values.append(left_value)
                    right_values.append(rows[right_index][1])
        correlation = None
        if len(left_values) >= 2:
            left_array = np.asarray(left_values, dtype=np.float64)
            right_array = np.asarray(right_values, dtype=np.float64)
            left_centered = left_array - left_array.mean()
            right_centered = right_array - right_array.mean()
            denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
            if denominator > 1e-12:
                correlation = float(np.clip(np.sum(left_centered * right_centered) / denominator, -1.0, 1.0))
        output.append({"lag_seconds": lag, "autocorrelation": correlation, "pair_count": len(left_values)})
        if progress_step is not None:
            progress_step()
    return output


def _plateau_lag(lag_statistics: list[dict[str, Any]]) -> int | None:
    valid = [row for row in lag_statistics if row["mean"] is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return int(valid[0]["lag_seconds"])
    tail = valid[-min(3, len(valid)):]
    asymptote = float(np.median([row["mean"] for row in tail]))
    baseline = float(valid[0]["mean"])
    if asymptote <= baseline:
        return int(valid[0]["lag_seconds"])
    threshold = baseline + 0.9 * (asymptote - baseline)
    for row in valid:
        if float(row["mean"]) >= threshold:
            return int(row["lag_seconds"])
    return int(valid[-1]["lag_seconds"])


def _recommendation(relevant_seconds: int) -> tuple[int, int, int]:
    sequence_length = 16
    strides = (1, 2, 4, 8)
    stride = min(strides, key=lambda value: (abs(sequence_length * value - relevant_seconds), value))
    return sequence_length, stride, sequence_length * stride


def analyze_temporal_dynamics_records(
    *,
    training_dataset: models.TrainingDataset,
    pipeline: models.PreprocessingPipeline,
    roi: models.RoiDefinition | None,
    records,
    compiled,
    reference_timestamp,
    lags_seconds: list[int],
    distance_metric: str,
    stride: int = 1,
    autocorrelation_max_lag_seconds: int = 128,
    autocorrelation_threshold: float = 0.2,
    progress_callback=None,
    abort_event=None,
) -> TemporalDynamicsResponse:
    """Analyze an already resolved Inspect range and report live work progress."""
    if len(records) < 2:
        raise ValueError("At least two images are required in the selected temporal analysis window.")
    lags_seconds = sorted(set(int(value) for value in lags_seconds if int(value) > 0))
    if not lags_seconds:
        raise ValueError("At least one positive temporal lag is required.")

    provisional_total = max(1, len(records) * (len(lags_seconds) + 2) + autocorrelation_max_lag_seconds)
    done = 0

    def report(total: int = provisional_total) -> None:
        if progress_callback is not None:
            progress_callback(done, max(1, total))

    loaded = []
    workers = min(_DECODE_WORKERS, len(records))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start_index in range(0, len(records), workers):
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("aborted")
            chunk = records[start_index : start_index + workers]
            loaded.extend(pool.map(lambda record: _prepare_frame(compiled, record), chunk))
            done += len(chunk)
            report()

    skipped = sum(1 for _, frame, _, _ in loaded if frame is None)
    valid = [(record, frame, shape) for record, frame, shape, _ in loaded if frame is not None]
    if len(valid) < 2:
        first_error = next((error for _, frame, _, error in loaded if frame is None and error), None)
        detail = f" First preprocessing error: {first_error}" if first_error else ""
        raise ValueError(f"Fewer than two readable images remain after preprocessing.{detail}")
    source_shapes = {shape for _, _, shape in valid}
    if len(source_shapes) != 1:
        raise ValueError("Preprocessing output size changed inside the selected analysis window.")
    source_height, source_width = next(iter(source_shapes))[:2]
    records = [item[0] for item in valid]
    frames = [item[1] for item in valid]

    mask, roi_meta = roi_union_mask(source_width, source_height, roi)
    if not np.any(mask):
        raise ValueError("The selected ROI has no pixels in the preprocessing output.")

    cadence_by_folder = _effective_cadence_by_folder(training_dataset, stride)
    segments, segment_cadences = _build_segments(records, cadence_by_folder)
    lag_pairs = {
        lag: _pairs_for_lag(records, segments, segment_cadences, lag)
        for lag in lags_seconds
    }
    motion_pair_count = sum(max(0, len(indices) - 1) for indices in segments)
    total = len(loaded) + sum(len(pairs) for pairs in lag_pairs.values()) + motion_pair_count + autocorrelation_max_lag_seconds
    done = len(loaded)
    report(total)

    def progress_step() -> None:
        nonlocal done
        done += 1
        report(total)

    lag_statistics: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for lag in lags_seconds:
        pairs = lag_pairs[lag]
        values: list[float] = []
        for left, right, _ in pairs:
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("aborted")
            values.append(_distance(frames[left], frames[right], distance_metric, mask))
            progress_step()
        lag_statistics.append(_summary(lag, values))
        if pairs:
            example_index = min(
                range(len(pairs)),
                key=lambda index: abs(
                    (records[pairs[index][0]].timestamp_parsed - reference_timestamp).total_seconds()
                ),
            )
            left, right, _ = pairs[example_index]
            display_mask = mask[..., None] if frames[left].ndim == 3 else mask
            left_display = np.where(display_mask, frames[left], 0).astype(np.uint8)
            right_display = np.where(display_mask, frames[right], 0).astype(np.uint8)
            difference_display = np.where(
                display_mask,
                np.abs(frames[left].astype(np.int16) - frames[right].astype(np.int16)),
                0,
            ).astype(np.uint8)
            examples.append({
                "lag_seconds": lag,
                "reference_timestamp": records[left].timestamp_parsed,
                "comparison_timestamp": records[right].timestamp_parsed,
                "actual_lag_seconds": (records[right].timestamp_parsed - records[left].timestamp_parsed).total_seconds(),
                "difference": values[example_index],
                "reference_image_data_url": encode_absolute_image_data_url(left_display),
                "comparison_image_data_url": encode_absolute_image_data_url(right_display),
                "difference_image_data_url": encode_absolute_image_data_url(difference_display),
            })

    motion_signal, internal_motion = _motion_signal(
        records, frames, segments, mask, progress_step, abort_event
    )
    autocorrelation = _autocorrelation(
        internal_motion,
        autocorrelation_max_lag_seconds,
        progress_step,
        abort_event,
    )
    correlation_length = next(
        (
            int(row["lag_seconds"])
            for row in autocorrelation
            if row["autocorrelation"] is not None
            and float(row["autocorrelation"]) < autocorrelation_threshold
        ),
        None,
    )
    plateau_lag = _plateau_lag(lag_statistics)
    relevant_candidates = [value for value in (plateau_lag, correlation_length) if value is not None]
    relevant_seconds = max(relevant_candidates) if relevant_candidates else max(lags_seconds)
    sequence_length, temporal_stride, covered_window = _recommendation(relevant_seconds)
    labels = {
        "mae": "Mean absolute difference (normalized intensity)",
        "mse": "Mean squared difference (normalized intensity)",
        "ssim": "Mean SSIM distance (1 - SSIM)",
    }
    report(total)
    return TemporalDynamicsResponse.model_validate({
        "training_dataset_id": training_dataset.id,
        "preprocessing_pipeline_id": pipeline.id,
        "training_dataset_name": training_dataset.name,
        "preprocessing_pipeline_name": pipeline.name,
        "roi_id": roi.id if roi else None,
        "roi_name": roi_meta["name"] if roi_meta else None,
        "reference_timestamp": reference_timestamp,
        "start_timestamp": min(record.timestamp_parsed for record in records),
        "end_timestamp": max(record.timestamp_parsed for record in records),
        "distance_metric": distance_metric,
        "distance_label": labels[distance_metric],
        "image_width": source_width,
        "image_height": source_height,
        "stride": max(1, int(stride)),
        "loaded_frame_count": len(frames),
        "skipped_frame_count": skipped,
        "contiguous_segment_count": len(segments),
        "lag_statistics": lag_statistics,
        "motion_signal": motion_signal,
        "autocorrelation": autocorrelation,
        "autocorrelation_threshold": autocorrelation_threshold,
        "estimated_correlation_length_seconds": correlation_length,
        "estimated_lag_plateau_seconds": plateau_lag,
        "estimated_relevant_time_scale_seconds": relevant_seconds,
        "recommended_sequence_length": sequence_length,
        "recommended_temporal_stride": temporal_stride,
        "covered_time_window_seconds": covered_window,
        "comparison_examples": examples,
        "cached": False,
    })


def analyze_temporal_dynamics(db: Session, payload: TemporalDynamicsRequest) -> TemporalDynamicsResponse:
    training_dataset = _load_training_dataset(db, int(payload.training_dataset_id))
    if training_dataset is None:
        raise ValueError(f"Train/Test Dataset does not exist: {payload.training_dataset_id}")
    pipeline = db.get(models.PreprocessingPipeline, int(payload.preprocessing_pipeline_id))
    if pipeline is None:
        raise ValueError(f"Preprocessing pipeline does not exist: {payload.preprocessing_pipeline_id}")
    roi = db.get(models.RoiDefinition, payload.roi_id) if payload.roi_id is not None else None
    if payload.roi_id is not None and roi is None:
        raise ValueError(f"ROI does not exist: {payload.roi_id}")

    half_window = timedelta(seconds=payload.analysis_window_seconds / 2.0)
    start = payload.reference_timestamp - half_window
    end = payload.reference_timestamp + half_window
    records = enumerate_training_dataset_image_records_for_range(
        training_dataset, start, end, extra_stride=max(1, payload.stride)
    )
    if len(records) < 2:
        raise ValueError("At least two images are required in the selected temporal analysis window.")
    key = _stable_cache_key(payload, training_dataset, pipeline, roi, records)
    cached = _cache_get(key)
    if cached is not None:
        return TemporalDynamicsResponse.model_validate(cached)

    compiled = compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
    result = analyze_temporal_dynamics_records(
        training_dataset=training_dataset,
        pipeline=pipeline,
        roi=roi,
        records=records,
        compiled=compiled,
        reference_timestamp=payload.reference_timestamp,
        lags_seconds=payload.lags_seconds,
        distance_metric=payload.distance_metric,
        stride=payload.stride,
        autocorrelation_max_lag_seconds=payload.autocorrelation_max_lag_seconds,
        autocorrelation_threshold=payload.autocorrelation_threshold,
    )
    serialized = result.model_dump(mode="python")
    _cache_put(key, serialized)
    return result
