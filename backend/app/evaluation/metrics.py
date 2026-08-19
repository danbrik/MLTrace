"""Pure calculations for the three single-model evaluation stages.

The module deliberately does not know about SQLAlchemy or API schemas.  It
accepts either the lightweight dataclasses declared here or mapping-shaped
records, and returns immutable result objects with JSON-safe ``to_dict``
methods. Normal windows and drift bins use half-open semantics
(``[start, end)``); labeled event and exclusion boundaries are inclusive.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta
from typing import Any, TypeAlias, cast

import numpy as np


DEFAULT_EPSILON = 1e-12
OPERATING_QUANTILES = (0.99, 0.995, 0.999, 0.9995, 0.9999)


class EvaluationMetricError(ValueError):
    """A deterministic validation/calculation failure for one metric stage."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class ScorePoint:
    timestamp: datetime
    position: int
    score: float
    continuity_segment: int = 0


@dataclass(frozen=True, slots=True)
class TimeRange:
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class EvaluationEvent:
    event_id: str
    start: datetime
    end: datetime
    name: str = ""
    category: str = ""


@dataclass(frozen=True, slots=True)
class MetricWarning:
    code: str
    message: str
    context: Mapping[str, Any]


class _JsonResult:
    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _json_safe(asdict(self)))


@dataclass(frozen=True, slots=True)
class SeparationEventResult(_JsonResult):
    event_id: str
    name: str
    category: str
    event_start: datetime
    event_end: datetime
    normal_start: datetime
    normal_end: datetime
    event_point_count: int
    normal_point_count: int
    normal_median: float
    normal_mad: float
    robust_scale: float
    separation: float
    separation_p95: float


@dataclass(frozen=True, slots=True)
class SeparationResult(_JsonResult):
    sep_median: float
    sep_min: float
    events: tuple[SeparationEventResult, ...]
    warnings: tuple[MetricWarning, ...]


@dataclass(frozen=True, slots=True)
class DriftWindowResult(_JsonResult):
    index: int
    start: datetime
    end: datetime
    status: str
    exclusion_reason: str | None
    point_count: int
    wasserstein_1: float | None
    normalized_drift: float | None


@dataclass(frozen=True, slots=True)
class DriftResult(_JsonResult):
    d_mean: float
    d_max: float
    reference_iqr: float
    reference_point_count: int
    valid_window_count: int
    discarded_window_count: int
    windows: tuple[DriftWindowResult, ...]
    warnings: tuple[MetricWarning, ...]


@dataclass(frozen=True, slots=True)
class DetectionEventResult(_JsonResult):
    event_id: str
    name: str
    category: str
    event_start: datetime
    event_end: datetime
    detection_window_start: datetime
    detection_window_end: datetime
    detected: bool
    first_detection: datetime | None
    delay_seconds: float | None


@dataclass(frozen=True, slots=True)
class FalseAlarmResult(_JsonResult):
    start: datetime
    end: datetime
    point_count: int
    continuity_segment: int


@dataclass(frozen=True, slots=True)
class DetectionOperatingPointResult(_JsonResult):
    quantile: float
    threshold: float
    event_recall: float
    median_delay_seconds: float | None
    frame_fpr: float
    far_t0: float
    event_count: int
    detected_event_count: int
    missed_event_count: int
    normal_frame_count: int
    false_positive_frame_count: int
    false_alarm_event_count: int
    normal_observation_seconds: float
    events: tuple[DetectionEventResult, ...]
    false_alarms: tuple[FalseAlarmResult, ...]


@dataclass(frozen=True, slots=True)
class DetectionResult(_JsonResult):
    calibration_point_count: int
    standard_duration_seconds: float
    anticipation_seconds: float
    operating_points: tuple[DetectionOperatingPointResult, ...]
    warnings: tuple[MetricWarning, ...]


ScorePointLike: TypeAlias = ScorePoint | Mapping[str, Any]
TimeRangeLike: TypeAlias = TimeRange | Mapping[str, Any]
EvaluationEventLike: TypeAlias = EvaluationEvent | Mapping[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _error(code: str, message: str, **details: Any) -> EvaluationMetricError:
    return EvaluationMetricError(code, message, details=_json_safe(details))


def _coerce_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise _error("invalid_timestamp", f"{field} is not a valid ISO timestamp", field=field) from exc
    if not isinstance(value, datetime):
        raise _error("invalid_timestamp", f"{field} must be a datetime", field=field)
    if value.tzinfo is not None and value.utcoffset() is not None:
        raise _error(
            "timezone_aware_timestamp",
            f"{field} must use dataset-local, timezone-naive time semantics",
            field=field,
        )
    return value


def _coerce_range(value: TimeRangeLike, field: str) -> TimeRange:
    if isinstance(value, TimeRange):
        result = TimeRange(
            start=_coerce_datetime(value.start, f"{field}.start"),
            end=_coerce_datetime(value.end, f"{field}.end"),
        )
    elif isinstance(value, Mapping):
        start = value.get("start", value.get("start_timestamp"))
        end = value.get("end", value.get("end_timestamp"))
        result = TimeRange(
            start=_coerce_datetime(start, f"{field}.start"),
            end=_coerce_datetime(end, f"{field}.end"),
        )
    else:
        raise _error("invalid_time_range", f"{field} must be a TimeRange or mapping", field=field)
    if result.start >= result.end:
        raise _error(
            "invalid_time_range",
            f"{field} start must be before end",
            field=field,
            start=result.start,
            end=result.end,
        )
    return result


def _coerce_point(value: ScorePointLike, index: int, field: str) -> ScorePoint:
    if isinstance(value, ScorePoint):
        point = ScorePoint(
            timestamp=_coerce_datetime(value.timestamp, f"{field}[{index}].timestamp"),
            position=int(value.position),
            score=float(value.score),
            continuity_segment=int(value.continuity_segment),
        )
    elif isinstance(value, Mapping):
        if "score" in value:
            score = value["score"]
        elif "value" in value:
            score = value["value"]
        else:
            raise _error(
                "missing_score_series",
                f"{field}[{index}] has no score value",
                field=field,
                index=index,
            )
        if score is None:
            raise _error(
                "missing_score_series",
                f"{field}[{index}] has no score value",
                field=field,
                index=index,
            )
        try:
            point = ScorePoint(
                timestamp=_coerce_datetime(value.get("timestamp"), f"{field}[{index}].timestamp"),
                position=int(value.get("position", index)),
                score=float(score),
                continuity_segment=int(value.get("continuity_segment", value.get("segment", 0))),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, EvaluationMetricError):
                raise
            raise _error(
                "invalid_score_point",
                f"{field}[{index}] is not a valid score point",
                field=field,
                index=index,
            ) from exc
    else:
        raise _error(
            "invalid_score_point",
            f"{field}[{index}] must be a ScorePoint or mapping",
            field=field,
            index=index,
        )
    if not math.isfinite(point.score):
        raise _error(
            "non_finite_score",
            f"{field}[{index}] has a non-finite score",
            field=field,
            index=index,
            position=point.position,
        )
    return point


def _ordered_points(values: Sequence[ScorePointLike], field: str) -> list[ScorePoint]:
    points = [_coerce_point(value, index, field) for index, value in enumerate(values)]
    points.sort(key=lambda point: (point.timestamp, point.position))
    return points


def _coerce_event(value: EvaluationEventLike, index: int) -> EvaluationEvent:
    if isinstance(value, EvaluationEvent):
        event = EvaluationEvent(
            event_id=str(value.event_id),
            start=_coerce_datetime(value.start, f"events[{index}].start"),
            end=_coerce_datetime(value.end, f"events[{index}].end"),
            name=str(value.name),
            category=str(value.category),
        )
    elif isinstance(value, Mapping):
        event_id = value.get("event_id", value.get("id"))
        if event_id is None or str(event_id).strip() == "":
            raise _error("missing_event_id", f"events[{index}] has no event id", index=index)
        event = EvaluationEvent(
            event_id=str(event_id),
            start=_coerce_datetime(
                value.get("start", value.get("start_timestamp")),
                f"events[{index}].start",
            ),
            end=_coerce_datetime(
                value.get("end", value.get("end_timestamp")),
                f"events[{index}].end",
            ),
            name=str(value.get("name") or ""),
            category=str(value.get("category") or ""),
        )
    else:
        raise _error(
            "invalid_event",
            f"events[{index}] must be an EvaluationEvent or mapping",
            index=index,
        )
    if event.start >= event.end:
        raise _error(
            "invalid_event_range",
            f"Event {event.event_id!r} start must be before end",
            event_id=event.event_id,
        )
    return event


def _events_for_range(
    values: Sequence[EvaluationEventLike],
    evaluation_range: TimeRange,
    *,
    required: bool,
) -> list[EvaluationEvent]:
    events = [_coerce_event(value, index) for index, value in enumerate(values)]
    ids = [event.event_id for event in events]
    duplicates = sorted({event_id for event_id in ids if ids.count(event_id) > 1})
    if duplicates:
        raise _error("duplicate_event_id", "Event ids must be unique", event_ids=duplicates)

    selected: list[EvaluationEvent] = []
    for event in events:
        if not _closed_ranges_overlap(
            event.start,
            event.end,
            evaluation_range.start,
            evaluation_range.end,
        ):
            continue
        if event.start < evaluation_range.start or event.end > evaluation_range.end:
            raise _error(
                "partial_event",
                f"Event {event.event_id!r} is only partially inside the evaluation range",
                event_id=event.event_id,
            )
        selected.append(event)
    selected.sort(key=lambda event: (event.start, event.end, event.event_id))
    for previous, current in zip(selected, selected[1:]):
        if current.start <= previous.end:
            raise _error(
                "overlapping_events",
                "Target event intervals must not overlap",
                event_ids=[previous.event_id, current.event_id],
            )
    if required and not selected:
        raise _error("no_target_events", "At least one target event is required in the evaluation range")
    return selected


def _coerce_exclusions(values: Sequence[TimeRangeLike]) -> list[TimeRange]:
    exclusions = [_coerce_range(value, f"exclusions[{index}]") for index, value in enumerate(values)]
    exclusions.sort(key=lambda item: (item.start, item.end))
    return exclusions


def _validate_event_exclusions(
    events: Sequence[EvaluationEvent],
    exclusions: Sequence[TimeRange],
) -> None:
    for event in events:
        if any(_closed_ranges_overlap(event.start, event.end, item.start, item.end) for item in exclusions):
            raise _error(
                "event_overlaps_exclusion",
                f"Event {event.event_id!r} overlaps an excluded/invalid interval",
                event_id=event.event_id,
            )


def _positive_duration(value: float, field: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error("invalid_duration", f"{field} must be a finite duration in seconds", field=field) from exc
    lower_bound_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not lower_bound_ok:
        comparison = "non-negative" if allow_zero else "positive"
        raise _error("invalid_duration", f"{field} must be a finite {comparison} duration", field=field)
    return result


def _positive_epsilon(value: float) -> float:
    epsilon = _positive_duration(value, "epsilon")
    return epsilon


def _slice_points(
    points: Sequence[ScorePoint],
    timestamps: Sequence[datetime],
    time_range: TimeRange,
    *,
    inclusive_end: bool,
) -> list[ScorePoint]:
    start_index = bisect_left(timestamps, time_range.start)
    if inclusive_end:
        end_index = bisect_right(timestamps, time_range.end)
    else:
        end_index = bisect_left(timestamps, time_range.end)
    return list(points[start_index:end_index])


def _closed_ranges_overlap(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> bool:
    return a_start <= b_end and b_start <= a_end


def _range_overlaps_events(time_range: TimeRange, events: Sequence[EvaluationEvent]) -> bool:
    # ``time_range`` is half-open while event boundaries are inclusive.
    return any(time_range.start <= event.end and event.start < time_range.end for event in events)


def _range_overlaps_ranges(time_range: TimeRange, ranges: Sequence[TimeRange]) -> bool:
    # ``time_range`` is half-open while exclusion boundaries are inclusive.
    return any(time_range.start <= item.end and item.start < time_range.end for item in ranges)


def _continuity_gaps(points: Sequence[ScorePoint]) -> list[tuple[datetime, datetime]]:
    return [
        (previous.timestamp, current.timestamp)
        for previous, current in zip(points, points[1:])
        if previous.continuity_segment != current.continuity_segment
    ]


def _range_crosses_continuity_gap(
    time_range: TimeRange,
    gaps: Sequence[tuple[datetime, datetime]],
) -> bool:
    for previous, current in gaps:
        if current <= time_range.start:
            continue
        if previous >= time_range.end:
            break
        # The open gap (previous, current) intersects the half-open range. A
        # boundary exactly at a range edge is harmless.
        if time_range.start < current and time_range.end > previous:
            return True
    return False


def _merge_closed_ranges(ranges: Sequence[TimeRange]) -> list[TimeRange]:
    merged: list[TimeRange] = []
    for item in sorted(ranges, key=lambda value: (value.start, value.end)):
        if merged and item.start <= merged[-1].end:
            merged[-1] = TimeRange(merged[-1].start, max(merged[-1].end, item.end))
        else:
            merged.append(item)
    return merged


def calculate_separation(
    points: Sequence[ScorePointLike],
    events: Sequence[EvaluationEventLike],
    *,
    evaluation_range: TimeRangeLike,
    normal_window_duration_seconds: float,
    normal_window_buffer_seconds: float = 0.0,
    exclusions: Sequence[TimeRangeLike] = (),
    normal_window_overrides: Mapping[str, TimeRangeLike] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> SeparationResult:
    """Calculate event-local robust separation (stage A).

    Every selected event must have a valid normal window and at least one
    score in both its event and normal interval.  Invalid events fail the whole
    stage rather than being silently omitted.
    """

    evaluation = _coerce_range(evaluation_range, "evaluation_range")
    duration = _positive_duration(normal_window_duration_seconds, "normal_window_duration_seconds")
    buffer_seconds = _positive_duration(
        normal_window_buffer_seconds,
        "normal_window_buffer_seconds",
        allow_zero=True,
    )
    epsilon = _positive_epsilon(epsilon)
    ordered = _ordered_points(points, "points")
    ordered_timestamps = [point.timestamp for point in ordered]
    gaps = _continuity_gaps(ordered)
    evaluation_points = _slice_points(
        ordered,
        ordered_timestamps,
        evaluation,
        inclusive_end=True,
    )
    if not evaluation_points:
        raise _error("missing_score_series", "No score points exist in the evaluation range")

    selected_events = _events_for_range(events, evaluation, required=True)
    excluded_ranges = _coerce_exclusions(exclusions)
    _validate_event_exclusions(selected_events, excluded_ranges)

    override_values = dict(normal_window_overrides or {})
    unknown_overrides = sorted(set(override_values) - {event.event_id for event in selected_events})
    if unknown_overrides:
        raise _error(
            "unknown_normal_window_override",
            "Normal-window overrides reference unknown or unselected events",
            event_ids=unknown_overrides,
        )

    results: list[SeparationEventResult] = []
    warnings: list[MetricWarning] = []
    invalid_events: list[dict[str, Any]] = []
    duration_delta = timedelta(seconds=duration)
    buffer_delta = timedelta(seconds=buffer_seconds)

    for event in selected_events:
        if event.event_id in override_values:
            normal_range = _coerce_range(
                override_values[event.event_id],
                f"normal_window_overrides[{event.event_id!r}]",
            )
        else:
            normal_end = event.start - buffer_delta
            normal_range = TimeRange(start=normal_end - duration_delta, end=normal_end)

        reasons: list[str] = []
        if normal_range.start < evaluation.start or normal_range.end > evaluation.end:
            reasons.append("outside_evaluation_range")
        if _range_overlaps_events(normal_range, selected_events):
            reasons.append("target_event")
        if _range_overlaps_ranges(normal_range, excluded_ranges):
            reasons.append("excluded_interval")
        if _range_crosses_continuity_gap(normal_range, gaps):
            reasons.append("continuity_gap")

        normal_points = _slice_points(
            ordered,
            ordered_timestamps,
            normal_range,
            inclusive_end=False,
        )
        event_range = TimeRange(event.start, event.end)
        event_points = _slice_points(
            ordered,
            ordered_timestamps,
            event_range,
            inclusive_end=True,
        )
        if not normal_points:
            reasons.append("no_normal_scores")
        if not event_points:
            reasons.append("no_event_scores")
        if reasons:
            invalid_events.append({
                "event_id": event.event_id,
                "normal_start": normal_range.start,
                "normal_end": normal_range.end,
                "reasons": sorted(set(reasons)),
            })
            continue

        normal_scores = np.asarray([point.score for point in normal_points], dtype=np.float64)
        event_scores = np.asarray([point.score for point in event_points], dtype=np.float64)
        normal_median = float(np.median(normal_scores))
        normal_mad = float(np.median(np.abs(normal_scores - normal_median)))
        robust_scale = 1.4826 * normal_mad + epsilon
        standardized = (event_scores - normal_median) / robust_scale
        separation = float(np.median(standardized))
        separation_p95 = float(np.quantile(standardized, 0.95, method="linear"))
        if normal_mad <= epsilon:
            warnings.append(MetricWarning(
                code="near_zero_normal_mad",
                message="The local normal MAD is near zero; separation is epsilon-dominated.",
                context={"event_id": event.event_id, "normal_mad": normal_mad},
            ))
        results.append(SeparationEventResult(
            event_id=event.event_id,
            name=event.name,
            category=event.category,
            event_start=event.start,
            event_end=event.end,
            normal_start=normal_range.start,
            normal_end=normal_range.end,
            event_point_count=len(event_points),
            normal_point_count=len(normal_points),
            normal_median=normal_median,
            normal_mad=normal_mad,
            robust_scale=robust_scale,
            separation=separation,
            separation_p95=separation_p95,
        ))

    if invalid_events:
        raise _error(
            "invalid_separation_events",
            "Every selected event needs a valid, gap-free normal window and score data",
            events=invalid_events,
        )

    separations = [item.separation for item in results]
    return SeparationResult(
        sep_median=float(statistics.median(separations)),
        sep_min=float(min(separations)),
        events=tuple(results),
        warnings=tuple(warnings),
    )


def empirical_wasserstein_1(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float:
    """Return the exact 1D empirical W1 distance without SciPy.

    The calculation integrates the absolute difference of the two empirical
    CDFs. It supports unequal sample sizes and repeated values.
    """

    left = np.sort(np.asarray(first, dtype=np.float64).reshape(-1))
    right = np.sort(np.asarray(second, dtype=np.float64).reshape(-1))
    if left.size == 0 or right.size == 0:
        raise _error("empty_distribution", "Wasserstein distance requires two non-empty distributions")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise _error("non_finite_score", "Wasserstein distance requires finite values")

    left_index = 0
    right_index = 0
    left_cdf = 0.0
    right_cdf = 0.0
    previous = float(min(left[0], right[0]))
    distance = 0.0

    while left_index < left.size or right_index < right.size:
        left_value = float(left[left_index]) if left_index < left.size else math.inf
        right_value = float(right[right_index]) if right_index < right.size else math.inf
        current = min(left_value, right_value)
        distance += abs(left_cdf - right_cdf) * (current - previous)
        while left_index < left.size and float(left[left_index]) == current:
            left_index += 1
        while right_index < right.size and float(right[right_index]) == current:
            right_index += 1
        left_cdf = left_index / left.size
        right_cdf = right_index / right.size
        previous = current

    return float(distance)


def calculate_drift(
    evaluation_points: Sequence[ScorePointLike],
    reference_points: Sequence[ScorePointLike],
    *,
    evaluation_range: TimeRangeLike,
    reference_range: TimeRangeLike,
    window_duration_seconds: float,
    events: Sequence[EvaluationEventLike] = (),
    exclusions: Sequence[TimeRangeLike] = (),
    epsilon: float = DEFAULT_EPSILON,
) -> DriftResult:
    """Calculate fixed-window normalized distribution drift (stage B)."""

    evaluation = _coerce_range(evaluation_range, "evaluation_range")
    reference = _coerce_range(reference_range, "reference_range")
    window_seconds = _positive_duration(window_duration_seconds, "window_duration_seconds")
    epsilon = _positive_epsilon(epsilon)
    ordered = _ordered_points(evaluation_points, "evaluation_points")
    reference_ordered = _ordered_points(reference_points, "reference_points")
    ordered_timestamps = [point.timestamp for point in ordered]
    reference_timestamps = [point.timestamp for point in reference_ordered]
    gaps = _continuity_gaps(ordered)
    scoped_points = _slice_points(
        ordered,
        ordered_timestamps,
        evaluation,
        inclusive_end=False,
    )
    scoped_reference = _slice_points(
        reference_ordered,
        reference_timestamps,
        reference,
        inclusive_end=True,
    )
    if not scoped_points:
        raise _error("missing_score_series", "No score points exist in the evaluation range")
    if not scoped_reference:
        raise _error("missing_reference_scores", "No score points exist in the reference range")

    selected_events = _events_for_range(events, evaluation, required=False)
    excluded_ranges = _coerce_exclusions(exclusions)

    reference_scores = np.asarray([point.score for point in scoped_reference], dtype=np.float64)
    q25, q75 = np.quantile(reference_scores, [0.25, 0.75], method="linear")
    reference_iqr = float(q75 - q25)
    denominator = reference_iqr + epsilon
    warnings: list[MetricWarning] = []
    if reference_iqr <= epsilon:
        warnings.append(MetricWarning(
            code="near_zero_reference_iqr",
            message="The reference IQR is near zero; normalized drift is epsilon-dominated.",
            context={"reference_iqr": reference_iqr},
        ))

    windows: list[DriftWindowResult] = []
    valid_values: list[float] = []
    window_delta = timedelta(seconds=window_seconds)
    cursor = evaluation.start
    index = 0
    while cursor < evaluation.end:
        window_end = cursor + window_delta
        if window_end > evaluation.end:
            windows.append(DriftWindowResult(
                index=index,
                start=cursor,
                end=evaluation.end,
                status="discarded",
                exclusion_reason="incomplete_window",
                point_count=len(_slice_points(
                    ordered,
                    ordered_timestamps,
                    TimeRange(cursor, evaluation.end),
                    inclusive_end=False,
                )),
                wasserstein_1=None,
                normalized_drift=None,
            ))
            break

        window_range = TimeRange(cursor, window_end)
        window_points = _slice_points(
            ordered,
            ordered_timestamps,
            window_range,
            inclusive_end=False,
        )
        reason: str | None = None
        if _range_overlaps_events(window_range, selected_events):
            reason = "target_event"
        elif _range_overlaps_ranges(window_range, excluded_ranges):
            reason = "excluded_interval"
        elif _range_crosses_continuity_gap(window_range, gaps):
            reason = "continuity_gap"
        elif not window_points:
            reason = "no_scores"

        if reason is not None:
            windows.append(DriftWindowResult(
                index=index,
                start=cursor,
                end=window_end,
                status="discarded",
                exclusion_reason=reason,
                point_count=len(window_points),
                wasserstein_1=None,
                normalized_drift=None,
            ))
        else:
            scores = [point.score for point in window_points]
            wasserstein = empirical_wasserstein_1(reference_scores, scores)
            normalized = wasserstein / denominator
            windows.append(DriftWindowResult(
                index=index,
                start=cursor,
                end=window_end,
                status="valid",
                exclusion_reason=None,
                point_count=len(window_points),
                wasserstein_1=wasserstein,
                normalized_drift=normalized,
            ))
            valid_values.append(normalized)
        cursor = window_end
        index += 1

    if not valid_values:
        raise _error(
            "no_valid_drift_windows",
            "The evaluation range contains no complete, valid normal drift window",
            windows=[item.to_dict() for item in windows],
        )

    return DriftResult(
        d_mean=float(statistics.fmean(valid_values)),
        d_max=float(max(valid_values)),
        reference_iqr=reference_iqr,
        reference_point_count=len(scoped_reference),
        valid_window_count=len(valid_values),
        discarded_window_count=len(windows) - len(valid_values),
        windows=tuple(windows),
        warnings=tuple(warnings),
    )


def calculate_detection(
    evaluation_points: Sequence[ScorePointLike],
    calibration_points: Sequence[ScorePointLike],
    events: Sequence[EvaluationEventLike],
    *,
    evaluation_range: TimeRangeLike,
    calibration_range: TimeRangeLike,
    standard_duration_seconds: float,
    exclusions: Sequence[TimeRangeLike] = (),
    anticipation_seconds: float = 0.0,
) -> DetectionResult:
    """Calculate all five fixed threshold operating points (stage C)."""

    evaluation = _coerce_range(evaluation_range, "evaluation_range")
    calibration = _coerce_range(calibration_range, "calibration_range")
    standard_seconds = _positive_duration(standard_duration_seconds, "standard_duration_seconds")
    anticipation = _positive_duration(anticipation_seconds, "anticipation_seconds", allow_zero=True)
    ordered = _ordered_points(evaluation_points, "evaluation_points")
    calibration_ordered = _ordered_points(calibration_points, "calibration_points")
    ordered_timestamps = [point.timestamp for point in ordered]
    calibration_timestamps = [point.timestamp for point in calibration_ordered]
    scoped_points = _slice_points(
        ordered,
        ordered_timestamps,
        evaluation,
        inclusive_end=True,
    )
    scoped_calibration = _slice_points(
        calibration_ordered,
        calibration_timestamps,
        calibration,
        inclusive_end=True,
    )
    if not scoped_points:
        raise _error("missing_score_series", "No score points exist in the evaluation range")
    if not scoped_calibration:
        raise _error("missing_calibration_scores", "No score points exist in the calibration range")

    selected_events = _events_for_range(events, evaluation, required=True)
    excluded_ranges = _coerce_exclusions(exclusions)
    anticipation_delta = timedelta(seconds=anticipation)
    detection_ranges: list[TimeRange] = []
    for event in selected_events:
        start = event.start - anticipation_delta
        if start < evaluation.start:
            raise _error(
                "anticipation_outside_evaluation",
                f"The anticipation window for event {event.event_id!r} starts before the evaluation range",
                event_id=event.event_id,
            )
        detection_ranges.append(TimeRange(start=start, end=event.end))
    for index in range(1, len(detection_ranges)):
        if detection_ranges[index].start <= detection_ranges[index - 1].end:
            raise _error(
                "overlapping_anticipation_windows",
                "Anticipation-extended event windows must not overlap",
                event_ids=[selected_events[index - 1].event_id, selected_events[index].event_id],
            )
    blocked_ranges = _merge_closed_ranges([*detection_ranges, *excluded_ranges])
    normal_flags: list[bool] = []
    blocked_index = 0
    for point in scoped_points:
        while (
            blocked_index < len(blocked_ranges)
            and blocked_ranges[blocked_index].end < point.timestamp
        ):
            blocked_index += 1
        blocked = (
            blocked_index < len(blocked_ranges)
            and blocked_ranges[blocked_index].start <= point.timestamp
            and point.timestamp <= blocked_ranges[blocked_index].end
        )
        normal_flags.append(not blocked)
    normal_frame_count = sum(normal_flags)
    if normal_frame_count == 0:
        raise _error("no_normal_frames", "No valid normal frames remain for false-positive evaluation")

    normal_observation_seconds = 0.0
    blocked_index = 0
    for index in range(1, len(scoped_points)):
        previous = scoped_points[index - 1]
        current = scoped_points[index]
        if not (normal_flags[index - 1] and normal_flags[index]):
            continue
        if previous.continuity_segment != current.continuity_segment:
            continue
        while (
            blocked_index < len(blocked_ranges)
            and blocked_ranges[blocked_index].end <= previous.timestamp
        ):
            blocked_index += 1
        if (
            blocked_index < len(blocked_ranges)
            and blocked_ranges[blocked_index].start < current.timestamp
            and blocked_ranges[blocked_index].end > previous.timestamp
        ):
            continue
        delta = (current.timestamp - previous.timestamp).total_seconds()
        if delta > 0:
            normal_observation_seconds += delta
    if normal_observation_seconds <= 0:
        raise _error(
            "no_normal_observation_duration",
            "Valid normal points provide no positive contiguous observation duration",
            normal_frame_count=normal_frame_count,
        )

    calibration_scores = np.asarray([point.score for point in scoped_calibration], dtype=np.float64)
    scoped_timestamps = [point.timestamp for point in scoped_points]
    warnings: list[MetricWarning] = []
    operating_points: list[DetectionOperatingPointResult] = []
    for quantile in OPERATING_QUANTILES:
        threshold = float(np.quantile(calibration_scores, quantile, method="linear"))
        expected_tail_count = len(scoped_calibration) * (1.0 - quantile)
        if expected_tail_count < 1.0:
            warnings.append(MetricWarning(
                code="underpopulated_calibration_tail",
                message="The calibration sample has fewer than one expected observation above this quantile.",
                context={
                    "quantile": quantile,
                    "calibration_point_count": len(scoped_calibration),
                    "expected_tail_count": expected_tail_count,
                },
            ))

        positive_flags = [point.score > threshold for point in scoped_points]
        event_results: list[DetectionEventResult] = []
        delays: list[float] = []
        for event, detection_range in zip(selected_events, detection_ranges, strict=True):
            start_index = bisect_left(scoped_timestamps, detection_range.start)
            end_index = bisect_right(scoped_timestamps, detection_range.end)
            first_detection = next(
                (
                    scoped_points[index].timestamp
                    for index in range(start_index, end_index)
                    if positive_flags[index]
                ),
                None,
            )
            delay = (
                (first_detection - event.start).total_seconds()
                if first_detection is not None
                else None
            )
            if delay is not None:
                delays.append(delay)
            event_results.append(DetectionEventResult(
                event_id=event.event_id,
                name=event.name,
                category=event.category,
                event_start=event.start,
                event_end=event.end,
                detection_window_start=detection_range.start,
                detection_window_end=detection_range.end,
                detected=first_detection is not None,
                first_detection=first_detection,
                delay_seconds=delay,
            ))

        detected_count = len(delays)
        false_positive_count = sum(
            1 for positive, normal in zip(positive_flags, normal_flags, strict=True)
            if positive and normal
        )
        false_alarms = _false_alarm_events(
            scoped_points,
            positive_flags,
            normal_flags,
            blocked_ranges,
        )
        operating_points.append(DetectionOperatingPointResult(
            quantile=quantile,
            threshold=threshold,
            event_recall=detected_count / len(selected_events),
            median_delay_seconds=float(statistics.median(delays)) if delays else None,
            frame_fpr=false_positive_count / normal_frame_count,
            far_t0=standard_seconds * len(false_alarms) / normal_observation_seconds,
            event_count=len(selected_events),
            detected_event_count=detected_count,
            missed_event_count=len(selected_events) - detected_count,
            normal_frame_count=normal_frame_count,
            false_positive_frame_count=false_positive_count,
            false_alarm_event_count=len(false_alarms),
            normal_observation_seconds=normal_observation_seconds,
            events=tuple(event_results),
            false_alarms=tuple(false_alarms),
        ))

    return DetectionResult(
        calibration_point_count=len(scoped_calibration),
        standard_duration_seconds=standard_seconds,
        anticipation_seconds=anticipation,
        operating_points=tuple(operating_points),
        warnings=tuple(warnings),
    )


def _false_alarm_events(
    points: Sequence[ScorePoint],
    positive_flags: Sequence[bool],
    normal_flags: Sequence[bool],
    blocked_ranges: Sequence[TimeRange],
) -> list[FalseAlarmResult]:
    results: list[FalseAlarmResult] = []
    active_start: datetime | None = None
    active_end: datetime | None = None
    active_count = 0
    active_segment = 0
    previous: ScorePoint | None = None
    blocked_index = 0

    def finish_active() -> None:
        nonlocal active_start, active_end, active_count
        if active_start is not None and active_end is not None:
            results.append(FalseAlarmResult(
                start=active_start,
                end=active_end,
                point_count=active_count,
                continuity_segment=active_segment,
            ))
        active_start = None
        active_end = None
        active_count = 0

    for point, positive, normal in zip(points, positive_flags, normal_flags, strict=True):
        if previous is not None:
            while (
                blocked_index < len(blocked_ranges)
                and blocked_ranges[blocked_index].end <= previous.timestamp
            ):
                blocked_index += 1
            crosses_blocked_range = (
                blocked_index < len(blocked_ranges)
                and blocked_ranges[blocked_index].start < point.timestamp
                and blocked_ranges[blocked_index].end > previous.timestamp
            )
            if (
                previous.continuity_segment != point.continuity_segment
                or crosses_blocked_range
            ):
                finish_active()
        if not normal or not positive:
            finish_active()
        elif active_start is None:
            active_start = point.timestamp
            active_end = point.timestamp
            active_count = 1
            active_segment = point.continuity_segment
        else:
            active_end = point.timestamp
            active_count += 1
        previous = point
    finish_active()
    return results
