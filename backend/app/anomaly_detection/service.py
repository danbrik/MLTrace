from __future__ import annotations

import math
import statistics
import threading
import time
from bisect import bisect_left, bisect_right, insort
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.schemas import (
    AnomalyDetectionConfig,
    AnomalyDetectionEventRead,
    AnomalyDetectionProgressRead,
    AnomalyDetectionRunCreate,
    AnomalyDetectionRunRead,
    AnomalyDetectionRunSummary,
    AnomalyDetectionSeriesPoint,
    AnomalyDetectionThresholdPreviewRead,
    AnomalyDetectionThresholdPreviewRequest,
)


ALGORITHM_VERSIONS = {
    "robust_zscore": "robust_zscore_v1",
    "robust_cusum": "robust_cusum_v1",
    "event_threshold": "event_threshold_v1",
}

_PROGRESS_TTL_SECONDS = 10 * 60
_PROGRESS_PHASE_RANGES = {
    "loading": (0.0, 10.0),
    "smoothing": (10.0, 25.0),
    "detecting": (25.0, 90.0),
    "saving": (90.0, 95.0),
    "plotting": (95.0, 100.0),
    "complete": (100.0, 100.0),
}
_PROGRESS_LOCK = threading.Lock()
_PROGRESS: dict[tuple[str, str], dict] = {}

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class SignalPoint:
    timestamp: datetime
    score: float


@dataclass
class DetectionEvent:
    warning_start: datetime
    confirmed_at: datetime | None
    end_timestamp: datetime
    end_reason: str
    peak_timestamp: datetime
    max_score: float
    max_robust_z: float | None
    duration_seconds: float | None = None
    max_smoothed_score: float | None = None
    mean_smoothed_score: float | None = None
    threshold: float | None = None


@dataclass
class DetectionOutput:
    series: list[AnomalyDetectionSeriesPoint]
    events: list[DetectionEvent]


class _FenwickMultiset:
    """Exact order-statistics multiset over coordinate-compressed floats."""

    def __init__(self, coordinates: list[float]) -> None:
        self.coordinates = coordinates
        self._indices = {value: index for index, value in enumerate(coordinates)}
        self._tree = [0] * (len(coordinates) + 1)
        # CPython performs list shifts in optimized C. Keeping this exact
        # ordered view makes the two-array MAD selection substantially faster
        # than repeated tree descent for the usual bounded baseline window.
        self._ordered: list[float] = []
        self.size = 0

    def add(self, value: float, delta: int) -> None:
        index = self._indices[value]
        tree_index = index + 1
        while tree_index < len(self._tree):
            self._tree[tree_index] += delta
            tree_index += tree_index & -tree_index
        if delta > 0:
            insort(self._ordered, value)
        elif delta < 0:
            self._ordered.pop(bisect_left(self._ordered, value))
        self.size += delta

    def count_before(self, coordinate_index: int) -> int:
        total = 0
        tree_index = coordinate_index
        while tree_index > 0:
            total += self._tree[tree_index]
            tree_index -= tree_index & -tree_index
        return total

    def value_at(self, order: int) -> float:
        if order < 0 or order >= self.size:
            raise IndexError("Order statistic is outside the multiset.")
        tree_index = 0
        bit = 1 << (len(self._tree).bit_length() - 1)
        remaining = order + 1
        while bit:
            candidate = tree_index + bit
            if candidate < len(self._tree) and self._tree[candidate] < remaining:
                tree_index = candidate
                remaining -= self._tree[candidate]
            bit >>= 1
        return self.coordinates[tree_index]

    def median(self) -> float | None:
        if self.size == 0:
            return None
        middle = self.size // 2
        if self.size % 2:
            return self._ordered[middle]
        return (self._ordered[middle - 1] + self._ordered[middle]) / 2.0

    def median_absolute_deviation(self, median: float) -> float | None:
        if self.size == 0:
            return None
        left_size = bisect_left(self._ordered, median)
        right_size = self.size - left_size

        def left_value(index: int) -> float:
            return median - self._ordered[left_size - 1 - index]

        def right_value(index: int) -> float:
            return self._ordered[left_size + index] - median

        def kth_distance(order: int) -> float:
            # Select from two virtual sorted arrays without materializing all
            # absolute deviations for every source point.
            left_partition_min = max(0, order + 1 - right_size)
            left_partition_max = min(order + 1, left_size)
            while left_partition_min <= left_partition_max:
                left_count = (left_partition_min + left_partition_max) // 2
                right_count = order + 1 - left_count
                left_before = left_value(left_count - 1) if left_count else -math.inf
                left_after = left_value(left_count) if left_count < left_size else math.inf
                right_before = right_value(right_count - 1) if right_count else -math.inf
                right_after = right_value(right_count) if right_count < right_size else math.inf
                if left_before <= right_after and right_before <= left_after:
                    return max(left_before, right_before)
                if left_before > right_after:
                    left_partition_max = left_count - 1
                else:
                    left_partition_min = left_count + 1
            raise RuntimeError("Could not select an absolute-deviation order statistic.")

        middle = self.size // 2
        if self.size % 2:
            return kth_distance(middle)
        return (kth_distance(middle - 1) + kth_distance(middle)) / 2.0


def _project_key(db: Session) -> str:
    return str(db.info.get("database_url") or db.get_bind().url)


def _cleanup_progress(now: float) -> None:
    expired = [key for key, entry in _PROGRESS.items() if now - entry["updated_monotonic"] > _PROGRESS_TTL_SECONDS]
    for key in expired:
        _PROGRESS.pop(key, None)


def _set_progress(
    db: Session,
    token: str | None,
    phase: str,
    completed: int,
    total: int,
    message: str,
    *,
    status: str = "running",
    error: str | None = None,
) -> None:
    if not token:
        return
    now = time.monotonic()
    phase_start, phase_end = _PROGRESS_PHASE_RANGES[phase]
    fraction = min(1.0, max(0.0, completed / total)) if total > 0 else 0.0
    percent = 100.0 if phase == "complete" else phase_start + (phase_end - phase_start) * fraction
    entry = {
        "progress_token": token,
        "phase": phase,
        "status": status,
        "completed": max(0, completed),
        "total": max(0, total),
        "percent": percent,
        "message": message,
        "error": error,
        "updated_at": datetime.now().astimezone(),
        "updated_monotonic": now,
    }
    with _PROGRESS_LOCK:
        _cleanup_progress(now)
        _PROGRESS[(_project_key(db), token)] = entry


def get_progress(db: Session, token: str) -> AnomalyDetectionProgressRead | None:
    now = time.monotonic()
    with _PROGRESS_LOCK:
        _cleanup_progress(now)
        entry = _PROGRESS.get((_project_key(db), token))
        if entry is None:
            return None
        payload = {key: value for key, value in entry.items() if key != "updated_monotonic"}
    return AnomalyDetectionProgressRead.model_validate(payload)


def _progress_callback(db: Session, token: str | None) -> ProgressCallback:
    return lambda phase, completed, total, message: _set_progress(
        db, token, phase, completed, total, message
    )


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _median_positive_delta_seconds(points: list[SignalPoint]) -> float:
    deltas = [
        (current.timestamp - previous.timestamp).total_seconds()
        for previous, current in zip(points, points[1:])
        if current.timestamp > previous.timestamp
    ]
    return statistics.median(deltas) if deltas else 1.0


def _smooth_event_scores(
    points: list[SignalPoint],
    config: AnomalyDetectionConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[SignalPoint], list[float], list[bool]]:
    ordered = sorted(points, key=lambda item: item.timestamp)
    if not ordered:
        return [], [], []
    cadence = _median_positive_delta_seconds(ordered)
    gap_seconds = max(config.event_minimum_gap_seconds, cadence * config.gap_multiplier)
    window_seconds = config.event_smoothing_window_seconds
    window: deque[SignalPoint] = deque()
    median_values = _FenwickMultiset(sorted({point.score for point in ordered}))
    moving_sum = 0.0
    smoothed_values: list[float] = []
    gap_flags: list[bool] = []
    previous_timestamp: datetime | None = None
    total = len(ordered)
    stride = max(1, total // 100)
    if progress_callback is not None:
        progress_callback("smoothing", 0, total, "Smoothing the score series")

    for index, point in enumerate(ordered):
        dt = 0.0 if previous_timestamp is None else max(
            0.0, (point.timestamp - previous_timestamp).total_seconds()
        )
        is_gap = previous_timestamp is not None and dt > gap_seconds
        if is_gap:
            while window:
                expired = window.popleft()
                moving_sum -= expired.score
                median_values.add(expired.score, -1)

        cutoff = point.timestamp - timedelta(seconds=window_seconds)
        while window and window[0].timestamp <= cutoff:
            expired = window.popleft()
            moving_sum -= expired.score
            median_values.add(expired.score, -1)
        window.append(point)
        moving_sum += point.score
        median_values.add(point.score, 1)

        if not config.event_smoothing_enabled:
            smoothed = point.score
        elif config.event_smoothing_method == "median":
            smoothed = median_values.median()
            if smoothed is None:  # pragma: no cover - current point is always present
                smoothed = point.score
        else:
            smoothed = moving_sum / len(window)
        smoothed_values.append(smoothed)
        gap_flags.append(is_gap)
        previous_timestamp = point.timestamp
        if progress_callback is not None and ((index + 1) % stride == 0 or index + 1 == total):
            progress_callback("smoothing", index + 1, total, "Smoothing the score series")
    return ordered, smoothed_values, gap_flags


def _merge_event_threshold_events(
    closed: list[tuple[DetectionEvent, int]],
    ordered: list[SignalPoint],
    smoothed_values: list[float],
    threshold: float,
    merge_gap_seconds: float,
) -> list[DetectionEvent]:
    if not closed:
        return []
    grouped: list[tuple[DetectionEvent, int]] = []
    for event, segment in closed:
        if grouped:
            previous, previous_segment = grouped[-1]
            gap = (event.warning_start - previous.end_timestamp).total_seconds()
            if segment == previous_segment and gap < merge_gap_seconds:
                previous.end_timestamp = event.end_timestamp
                previous.end_reason = event.end_reason
                continue
        grouped.append((event, segment))

    timestamps = [point.timestamp for point in ordered]
    result: list[DetectionEvent] = []
    for event, _segment in grouped:
        start_index = bisect_left(timestamps, event.warning_start)
        end_index = bisect_right(timestamps, event.end_timestamp)
        event_points = ordered[start_index:end_index]
        event_smoothed = smoothed_values[start_index:end_index]
        if event_points:
            peak_offset = max(range(len(event_points)), key=lambda index: event_points[index].score)
            event.peak_timestamp = event_points[peak_offset].timestamp
            event.max_score = event_points[peak_offset].score
        if event_smoothed:
            event.max_smoothed_score = max(event_smoothed)
            event.mean_smoothed_score = statistics.fmean(event_smoothed)
        event.duration_seconds = max(
            0.0, (event.end_timestamp - event.warning_start).total_seconds()
        )
        event.threshold = threshold
        result.append(event)
    return result


def _detect_event_threshold(
    points: list[SignalPoint],
    config: AnomalyDetectionConfig,
    threshold: float,
    progress_callback: ProgressCallback | None = None,
) -> DetectionOutput:
    ordered, smoothed_values, gap_flags = _smooth_event_scores(points, config, progress_callback)
    if not ordered:
        return DetectionOutput(series=[], events=[])

    threshold_off = threshold * config.threshold_off_factor
    candidates: deque[bool] = deque()
    candidate_count = 0
    active: DetectionEvent | None = None
    recovery_started: datetime | None = None
    closed: list[tuple[DetectionEvent, int]] = []
    segment = 0
    series: list[AnomalyDetectionSeriesPoint] = []
    total = len(ordered)
    stride = max(1, total // 100)

    def clear_candidates() -> None:
        nonlocal candidate_count
        candidates.clear()
        candidate_count = 0

    def close_active(at: datetime, reason: str) -> None:
        nonlocal active, recovery_started
        if active is not None:
            active.end_timestamp = at
            active.end_reason = reason
            closed.append((active, segment))
        active = None
        recovery_started = None
        if reason in {"recovered", "data_gap"}:
            clear_candidates()

    if progress_callback is not None:
        progress_callback("detecting", 0, total, "Applying K-out-of-N event detection")
    for index, point in enumerate(ordered):
        smoothed = smoothed_values[index]
        if gap_flags[index]:
            close_active(ordered[index - 1].timestamp, "data_gap")
            segment += 1
            clear_candidates()

        candidate = smoothed > threshold
        if len(candidates) == config.persistence_n:
            candidate_count -= int(candidates.popleft())
        candidates.append(candidate)
        candidate_count += int(candidate)

        if active is None and candidate_count >= config.persistence_k:
            active = DetectionEvent(
                warning_start=point.timestamp,
                confirmed_at=point.timestamp,
                end_timestamp=point.timestamp,
                end_reason="range_end",
                peak_timestamp=point.timestamp,
                max_score=point.score,
                max_robust_z=None,
                threshold=threshold,
            )

        if active is not None:
            active.end_timestamp = point.timestamp
            if smoothed <= threshold_off:
                recovery_started = recovery_started or point.timestamp
                if (point.timestamp - recovery_started).total_seconds() >= config.normal_close_seconds:
                    close_active(point.timestamp, "recovered")
            else:
                recovery_started = None

        series.append(AnomalyDetectionSeriesPoint(
            timestamp=point.timestamp,
            score=point.score,
            smoothed=smoothed,
            baseline=None,
            warning_threshold=None,
            high_threshold=None,
            robust_z=None,
            cusum=0.0,
            state="confirmed" if active is not None else "normal",
            threshold_on=threshold,
            threshold_off=threshold_off,
            candidate=candidate,
            persistence_count=candidate_count,
        ))
        if progress_callback is not None and ((index + 1) % stride == 0 or index + 1 == total):
            progress_callback("detecting", index + 1, total, "Applying K-out-of-N event detection")

    if active is not None:
        close_active(ordered[-1].timestamp, "range_end")
    events = _merge_event_threshold_events(
        closed,
        ordered,
        smoothed_values,
        threshold,
        config.merge_gap_seconds,
    )
    return DetectionOutput(series=series, events=events)


def detect(
    points: list[SignalPoint],
    config: AnomalyDetectionConfig,
    progress_callback: ProgressCallback | None = None,
    *,
    resolved_threshold: float | None = None,
) -> DetectionOutput:
    if config.algorithm == "event_threshold":
        threshold = config.manual_threshold if config.threshold_mode == "manual" else resolved_threshold
        if threshold is None or not math.isfinite(threshold):
            raise ValueError("A finite resolved threshold is required for event-threshold detection.")
        return _detect_event_threshold(points, config, float(threshold), progress_callback)
    return _detect_robust(points, config, progress_callback)


def _detect_robust(
    points: list[SignalPoint],
    config: AnomalyDetectionConfig,
    progress_callback: ProgressCallback | None = None,
) -> DetectionOutput:
    """Run the version-1 causal detector.

    The baseline clock advances only while the detector is normal. Warning and
    confirmed intervals therefore cannot teach their elevated scores back into
    the rolling median/MAD baseline.
    """
    if not points:
        return DetectionOutput(series=[], events=[])

    ordered = sorted(points, key=lambda item: item.timestamp)
    cadence = _median_positive_delta_seconds(ordered)
    gap_seconds = max(config.minimum_gap_minutes * 60.0, cadence * config.gap_multiplier)
    half_life_seconds = config.smoothing_half_life_minutes * 60.0
    baseline_window_seconds = config.baseline_window_minutes * 60.0
    warmup_seconds = config.warmup_minutes * 60.0
    confirmation_seconds = config.confirmation_minutes * 60.0
    recovery_seconds = config.recovery_minutes * 60.0

    total_points = len(ordered)
    progress_stride = max(1, total_points // 100)
    smoothed_values: list[float] = []
    elapsed_seconds: list[float] = []
    gap_flags: list[bool] = []
    smoothed = ordered[0].score
    previous_timestamp: datetime | None = None
    if progress_callback is not None:
        progress_callback("smoothing", 0, total_points, "Smoothing the score series")
    for index, point in enumerate(ordered):
        dt = 0.0 if previous_timestamp is None else max(
            0.0, (point.timestamp - previous_timestamp).total_seconds()
        )
        is_gap = previous_timestamp is not None and dt > gap_seconds
        if is_gap or previous_timestamp is None:
            smoothed = point.score
            effective_dt = 0.0
        else:
            effective_dt = dt
            alpha = 1.0 - math.exp(-math.log(2.0) * dt / half_life_seconds) if dt > 0 else 0.0
            smoothed = alpha * point.score + (1.0 - alpha) * smoothed
        smoothed_values.append(smoothed)
        elapsed_seconds.append(effective_dt)
        gap_flags.append(is_gap)
        previous_timestamp = point.timestamp
        if progress_callback is not None and (
            (index + 1) % progress_stride == 0 or index + 1 == total_points
        ):
            progress_callback("smoothing", index + 1, total_points, "Smoothing the score series")

    baseline_buffer: deque[tuple[float, float]] = deque()
    baseline_values = _FenwickMultiset(sorted(set(smoothed_values)))
    normal_clock = 0.0
    first_baseline_clock: float | None = None
    state = "normal"
    cusum = 0.0
    active: DetectionEvent | None = None
    recovery_started: datetime | None = None
    output_events: list[DetectionEvent] = []
    series: list[AnomalyDetectionSeriesPoint] = []

    def close_active(at: datetime, reason: str) -> None:
        nonlocal active, state, cusum, recovery_started
        if active is not None:
            active.end_timestamp = at
            active.end_reason = reason
            output_events.append(active)
        active = None
        state = "normal"
        cusum = 0.0
        recovery_started = None

    if progress_callback is not None:
        progress_callback("detecting", 0, total_points, "Detecting warnings and anomalies")
    for index, point in enumerate(ordered):
        dt = elapsed_seconds[index]
        is_gap = gap_flags[index]
        smoothed = smoothed_values[index]
        if is_gap:
            close_active(ordered[index - 1].timestamp, "data_gap")

        if state == "normal":
            normal_clock += dt
            cutoff = normal_clock - baseline_window_seconds
            while baseline_buffer and baseline_buffer[0][0] < cutoff:
                _, expired_value = baseline_buffer.popleft()
                baseline_values.add(expired_value, -1)

        baseline = baseline_values.median()
        mad = baseline_values.median_absolute_deviation(baseline) if baseline is not None else None
        scale = None
        if baseline is not None and mad is not None:
            scale = max(1.4826 * mad, abs(baseline) * 1e-6, 1e-12)
        baseline_span = 0.0 if first_baseline_clock is None else normal_clock - first_baseline_clock
        ready = baseline_values.size >= config.minimum_warmup_points and baseline_span >= warmup_seconds

        robust_z = (smoothed - baseline) / scale if ready and baseline is not None and scale is not None else None
        warning_threshold = baseline + config.warning_z * scale if ready and baseline is not None and scale is not None else None
        high_threshold = baseline + config.high_z * scale if ready and baseline is not None and scale is not None else None

        if ready and robust_z is not None:
            evidence_minutes = max(0.0, dt) / 60.0
            if config.algorithm == "robust_cusum":
                cusum = max(0.0, cusum + (robust_z - config.cusum_drift) * evidence_minutes)
            else:
                cusum = 0.0
            if state == "normal" and robust_z >= config.warning_z:
                state = "warning"
                active = DetectionEvent(
                    warning_start=point.timestamp,
                    confirmed_at=None,
                    end_timestamp=point.timestamp,
                    end_reason="range_end",
                    peak_timestamp=point.timestamp,
                    max_score=point.score,
                    max_robust_z=robust_z,
                )
            if active is not None:
                if point.score > active.max_score:
                    active.max_score = point.score
                    active.peak_timestamp = point.timestamp
                active.max_robust_z = max(active.max_robust_z, robust_z)
                warning_age = (point.timestamp - active.warning_start).total_seconds()
                high_evidence = robust_z >= config.high_z
                if config.algorithm == "robust_cusum":
                    high_evidence = high_evidence or cusum >= config.cusum_threshold
                if (
                    state == "warning"
                    and warning_age >= confirmation_seconds
                    and robust_z >= config.warning_z
                    and high_evidence
                ):
                    state = "confirmed"
                    active.confirmed_at = point.timestamp

                if robust_z < config.recovery_z:
                    recovery_started = recovery_started or point.timestamp
                    if (point.timestamp - recovery_started).total_seconds() >= recovery_seconds:
                        close_active(point.timestamp, "recovered")
                else:
                    recovery_started = None
        else:
            cusum = 0.0

        display_state = "warmup" if not ready else state
        series.append(AnomalyDetectionSeriesPoint(
            timestamp=point.timestamp,
            score=point.score,
            smoothed=smoothed,
            baseline=baseline if ready else None,
            warning_threshold=warning_threshold,
            high_threshold=high_threshold,
            robust_z=robust_z,
            cusum=cusum,
            state=display_state,
        ))

        if state == "normal":
            if first_baseline_clock is None:
                first_baseline_clock = normal_clock
            baseline_buffer.append((normal_clock, smoothed))
            baseline_values.add(smoothed, 1)
        if progress_callback is not None and (
            (index + 1) % progress_stride == 0 or index + 1 == total_points
        ):
            progress_callback("detecting", index + 1, total_points, "Detecting warnings and anomalies")

    if active is not None:
        close_active(ordered[-1].timestamp, "range_end")
    return DetectionOutput(series=series, events=output_events)


def _score_column(score_series: str):
    if score_series == "score":
        return models.TestingRunResult.score
    if score_series == "full_mse":
        return models.TestingRunResult.full_mse
    if score_series == "roi_mse":
        return models.TestingRunResult.roi_mse
    raise ValueError(f"Unsupported score series: {score_series}")


def _load_points(
    db: Session,
    testing_run_id: int,
    score_series: str,
    start_timestamp: datetime,
    end_timestamp: datetime,
    preroll_minutes: float,
) -> list[SignalPoint]:
    column = _score_column(score_series)
    query_start = start_timestamp - timedelta(minutes=preroll_minutes)
    rows = db.execute(
        select(models.TestingRunResult.timestamp, column)
        .where(
            models.TestingRunResult.testing_run_id == testing_run_id,
            models.TestingRunResult.timestamp >= query_start,
            models.TestingRunResult.timestamp <= end_timestamp,
            column.is_not(None),
        )
        .order_by(models.TestingRunResult.timestamp, models.TestingRunResult.position)
    ).all()
    return [SignalPoint(timestamp=timestamp, score=float(score)) for timestamp, score in rows if _finite(score)]


def _validate_threshold_reference(
    target: models.TestingRun,
    reference: models.TestingRun,
    score_series: str,
) -> None:
    if reference.id == target.id:
        raise ValueError("The validation inference must differ from the analyzed inference.")
    if reference.status != "finished":
        raise ValueError("The validation inference must be finished.")
    if reference.training_run_id != target.training_run_id:
        raise ValueError("The validation inference must use the same trained model.")
    if (reference.inference_config or {}) != (target.inference_config or {}):
        raise ValueError("The validation inference must use the same scoring configuration.")
    if score_series in {"score", "roi_mse"} and (reference.roi_geometry or None) != (target.roi_geometry or None):
        raise ValueError("The validation inference must use the same ROI geometry for this score series.")
    if score_series == "roi_mse" and reference.roi_mse_mean is None:
        raise ValueError("The validation inference has no ROI MSE values.")


def _calculate_quantile_threshold(
    db: Session,
    testing_run: models.TestingRun,
    score_series: str,
    start_timestamp: datetime,
    end_timestamp: datetime,
    *,
    smoothing_enabled: bool,
    smoothing_method: str,
    smoothing_window_seconds: float,
    gap_multiplier: float,
    minimum_gap_seconds: float,
    quantile: float,
) -> tuple[float, int]:
    points = _load_points(
        db,
        testing_run.id,
        score_series,
        start_timestamp,
        end_timestamp,
        0.0,
    )
    if not points:
        raise ValueError("No finite validation scores exist in the selected threshold range.")
    smoothing_config = AnomalyDetectionConfig(
        algorithm="event_threshold",
        threshold_mode="quantile",
        event_smoothing_enabled=smoothing_enabled,
        event_smoothing_method=smoothing_method,
        event_smoothing_window_seconds=smoothing_window_seconds,
        gap_multiplier=gap_multiplier,
        event_minimum_gap_seconds=minimum_gap_seconds,
    )
    _ordered, smoothed, _gaps = _smooth_event_scores(points, smoothing_config)
    threshold = float(np.quantile(np.asarray(smoothed, dtype=np.float64), quantile, method="linear"))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("The calculated validation threshold must be finite and non-negative.")
    return threshold, len(smoothed)


def preview_threshold(
    db: Session,
    payload: AnomalyDetectionThresholdPreviewRequest,
) -> AnomalyDetectionThresholdPreviewRead:
    testing_run = db.get(models.TestingRun, payload.testing_run_id)
    if testing_run is None:
        raise ValueError("Validation inference run not found.")
    if testing_run.status != "finished":
        raise ValueError("Only finished inference runs can provide validation scores.")
    threshold, point_count = _calculate_quantile_threshold(
        db,
        testing_run,
        payload.score_series,
        payload.start_timestamp,
        payload.end_timestamp,
        smoothing_enabled=payload.smoothing_enabled,
        smoothing_method=payload.smoothing_method,
        smoothing_window_seconds=payload.smoothing_window_seconds,
        gap_multiplier=payload.gap_multiplier,
        minimum_gap_seconds=payload.minimum_gap_seconds,
        quantile=payload.quantile,
    )
    return AnomalyDetectionThresholdPreviewRead(
        testing_run_id=testing_run.id,
        testing_run_name=testing_run.name,
        score_series=payload.score_series,
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        point_count=point_count,
        quantile=payload.quantile,
        threshold=threshold,
    )


def _visible_output(output: DetectionOutput, start: datetime, end: datetime) -> DetectionOutput:
    timestamps = [point.timestamp for point in output.series]
    visible_start = bisect_left(timestamps, start)
    visible_end = bisect_right(timestamps, end)
    series = output.series[visible_start:visible_end]
    events: list[DetectionEvent] = []
    for event in output.events:
        if event.end_timestamp < start or event.warning_start > end:
            continue
        warning_start = max(start, event.warning_start)
        end_timestamp = min(end, event.end_timestamp)
        confirmed_at = event.confirmed_at
        if confirmed_at is not None:
            if confirmed_at < start:
                confirmed_at = start
            elif confirmed_at > end_timestamp:
                confirmed_at = None
        event_start = bisect_left(timestamps, warning_start)
        event_end = bisect_right(timestamps, end_timestamp)
        visible_event_points = output.series[event_start:event_end]
        peak_point = max(visible_event_points, key=lambda point: point.score) if visible_event_points else None
        visible_smoothed = [point.smoothed for point in visible_event_points]
        visible_robust_z = [point.robust_z for point in visible_event_points if point.robust_z is not None]
        events.append(DetectionEvent(
            warning_start=warning_start,
            confirmed_at=confirmed_at,
            end_timestamp=end_timestamp,
            end_reason=event.end_reason if event.end_timestamp <= end else "range_end",
            peak_timestamp=peak_point.timestamp if peak_point else min(max(event.peak_timestamp, warning_start), end_timestamp),
            max_score=peak_point.score if peak_point else event.max_score,
            max_robust_z=max(visible_robust_z, default=event.max_robust_z),
            duration_seconds=max(0.0, (end_timestamp - warning_start).total_seconds()),
            max_smoothed_score=max(visible_smoothed, default=event.max_smoothed_score),
            mean_smoothed_score=(statistics.fmean(visible_smoothed) if visible_smoothed else event.mean_smoothed_score),
            threshold=event.threshold,
        ))
    return DetectionOutput(series=series, events=events)


def _compute_for_run(
    db: Session,
    run: models.AnomalyDetectionRun,
    progress_token: str | None = None,
) -> DetectionOutput:
    config = AnomalyDetectionConfig.model_validate(run.config)
    _set_progress(db, progress_token, "loading", 0, 0, "Loading the full-resolution score series")
    points = _load_points(
        db,
        run.testing_run_id,
        run.score_series,
        run.start_timestamp,
        run.end_timestamp,
        config.preroll_minutes,
    )
    _set_progress(
        db,
        progress_token,
        "loading",
        len(points),
        len(points),
        f"Loaded {len(points):,} score points",
    )
    detected = detect(
        points,
        config,
        _progress_callback(db, progress_token),
        resolved_threshold=run.resolved_threshold,
    )
    return _visible_output(detected, run.start_timestamp, run.end_timestamp)


def _decimate_series(
    points: list[AnomalyDetectionSeriesPoint], max_points: int | None
) -> tuple[list[AnomalyDetectionSeriesPoint], bool]:
    if not max_points or len(points) <= max_points:
        return points, False
    if max_points < 4:
        return [points[0], points[-1]][:max_points], True
    bucket_count = max(1, (max_points - 2) // 2)
    interior = points[1:-1]
    selected: dict[int, AnomalyDetectionSeriesPoint] = {0: points[0], len(points) - 1: points[-1]}
    for bucket in range(bucket_count):
        start = bucket * len(interior) // bucket_count
        end = (bucket + 1) * len(interior) // bucket_count
        if start >= end:
            continue
        indexed = list(enumerate(interior[start:end], start=start + 1))
        for index, point in (min(indexed, key=lambda item: item[1].score), max(indexed, key=lambda item: item[1].score)):
            selected[index] = point
    ordered = [selected[index] for index in sorted(selected)]
    if len(ordered) > max_points:
        step = (len(ordered) - 1) / (max_points - 1)
        indices = sorted({round(index * step) for index in range(max_points)})
        ordered = [ordered[index] for index in indices]
    return ordered, True


def list_runs(db: Session) -> list[AnomalyDetectionRunSummary]:
    runs = db.scalars(select(models.AnomalyDetectionRun).order_by(models.AnomalyDetectionRun.created_at.desc())).all()
    return [AnomalyDetectionRunSummary.model_validate(run) for run in runs]


def _run_read(
    run: models.AnomalyDetectionRun,
    output: DetectionOutput,
    max_points: int | None,
) -> AnomalyDetectionRunRead:
    visible_series, decimated = _decimate_series(output.series, max_points)
    summary = AnomalyDetectionRunSummary.model_validate(run)
    return AnomalyDetectionRunRead(
        **summary.model_dump(),
        events=[AnomalyDetectionEventRead.model_validate(event) for event in run.events],
        series=visible_series,
        total=len(output.series),
        decimated=decimated,
    )


def create_run(db: Session, payload: AnomalyDetectionRunCreate) -> AnomalyDetectionRunRead:
    token = payload.progress_token
    _set_progress(db, token, "loading", 0, 0, "Starting anomaly detection")
    try:
        testing_run = db.get(models.TestingRun, payload.testing_run_id)
        if testing_run is None:
            raise ValueError("Inference run not found.")
        if testing_run.status != "finished":
            raise ValueError("Only finished inference runs can be analyzed.")

        name = payload.name.strip()
        if not name:
            raise ValueError("Run name is required.")
        threshold_testing_run: models.TestingRun | None = None
        resolved_threshold: float | None = None
        if payload.config.algorithm == "event_threshold":
            if payload.config.threshold_mode == "manual":
                resolved_threshold = payload.config.manual_threshold
            else:
                threshold_testing_run = db.get(models.TestingRun, payload.threshold_testing_run_id)
                if threshold_testing_run is None:
                    raise ValueError("Validation inference run not found.")
                _validate_threshold_reference(testing_run, threshold_testing_run, payload.score_series)
                resolved_threshold, _point_count = _calculate_quantile_threshold(
                    db,
                    threshold_testing_run,
                    payload.score_series,
                    payload.threshold_start_timestamp,
                    payload.threshold_end_timestamp,
                    smoothing_enabled=payload.config.event_smoothing_enabled,
                    smoothing_method=payload.config.event_smoothing_method,
                    smoothing_window_seconds=payload.config.event_smoothing_window_seconds,
                    gap_multiplier=payload.config.gap_multiplier,
                    minimum_gap_seconds=payload.config.event_minimum_gap_seconds,
                    quantile=payload.config.threshold_quantile,
                )
        run = models.AnomalyDetectionRun(
            name=name,
            testing_run_id=testing_run.id,
            testing_run_name=testing_run.name,
            threshold_testing_run_id=threshold_testing_run.id if threshold_testing_run else None,
            threshold_testing_run_name=threshold_testing_run.name if threshold_testing_run else None,
            threshold_start_timestamp=(payload.threshold_start_timestamp if threshold_testing_run else None),
            threshold_end_timestamp=(payload.threshold_end_timestamp if threshold_testing_run else None),
            resolved_threshold=resolved_threshold,
            score_series=payload.score_series,
            start_timestamp=payload.start_timestamp,
            end_timestamp=payload.end_timestamp,
            algorithm_version=ALGORITHM_VERSIONS[payload.config.algorithm],
            config=payload.config.model_dump(mode="json"),
        )
        db.add(run)
        db.flush()
        output = _compute_for_run(db, run, token)
        if not output.series:
            series_label = payload.score_series.replace("_", " ")
            raise ValueError(f"No {series_label} values exist in the selected time range.")
        run.point_count = len(output.series)
        run.warning_count = len(output.events)
        run.anomaly_count = sum(event.confirmed_at is not None for event in output.events)
        _set_progress(db, token, "saving", 0, max(1, len(output.events)), "Saving detected events")
        for index, event in enumerate(output.events):
            run.events.append(models.AnomalyDetectionEvent(**event.__dict__))
            _set_progress(
                db,
                token,
                "saving",
                index + 1,
                max(1, len(output.events)),
                "Saving detected events",
            )
        if not output.events:
            _set_progress(db, token, "saving", 1, 1, "Saving detection run")
        db.commit()
        _set_progress(db, token, "plotting", 0, len(output.series), "Preparing the plot series")
        result = _run_read(run, output, 8000)
        _set_progress(
            db,
            token,
            "plotting",
            len(output.series),
            len(output.series),
            "Prepared the plot series",
        )
        _set_progress(db, token, "complete", 1, 1, "Anomaly detection complete", status="complete")
        return result
    except Exception as exc:
        db.rollback()
        _set_progress(
            db,
            token,
            "complete",
            0,
            1,
            "Anomaly detection failed",
            status="error",
            error=str(exc),
        )
        raise


def get_run(
    db: Session,
    run_id: int,
    *,
    max_points: int | None = 8000,
    progress_token: str | None = None,
) -> AnomalyDetectionRunRead | None:
    _set_progress(db, progress_token, "loading", 0, 0, "Loading saved detection run")
    try:
        run = db.scalar(
            select(models.AnomalyDetectionRun)
            .where(models.AnomalyDetectionRun.id == run_id)
            .options(selectinload(models.AnomalyDetectionRun.events))
        )
        if run is None:
            _set_progress(
                db,
                progress_token,
                "complete",
                0,
                1,
                "Anomaly detection run not found",
                status="error",
                error="Anomaly detection run not found.",
            )
            return None
        output = _compute_for_run(db, run, progress_token)
        _set_progress(db, progress_token, "plotting", 0, len(output.series), "Reducing the plot series")
        result = _run_read(run, output, max_points)
        _set_progress(
            db,
            progress_token,
            "complete",
            1,
            1,
            "Saved detection run loaded",
            status="complete",
        )
        return result
    except Exception as exc:
        _set_progress(
            db,
            progress_token,
            "complete",
            0,
            1,
            "Could not load saved detection run",
            status="error",
            error=str(exc),
        )
        raise


def delete_run(db: Session, run_id: int) -> bool:
    run = db.get(models.AnomalyDetectionRun, run_id)
    if run is None:
        return False
    db.delete(run)
    db.commit()
    return True


def delete_runs_for_testing_run(db: Session, testing_run_id: int) -> None:
    db.execute(
        delete(models.AnomalyDetectionRun).where(
            or_(
                models.AnomalyDetectionRun.testing_run_id == testing_run_id,
                models.AnomalyDetectionRun.threshold_testing_run_id == testing_run_id,
            )
        )
    )
