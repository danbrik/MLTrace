from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.schemas import (
    AnomalyDetectionConfig,
    AnomalyDetectionEventRead,
    AnomalyDetectionRunCreate,
    AnomalyDetectionRunRead,
    AnomalyDetectionRunSummary,
    AnomalyDetectionSeriesPoint,
)


ALGORITHM_VERSIONS = {
    "robust_zscore": "robust_zscore_v1",
    "robust_cusum": "robust_cusum_v1",
}


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
    max_robust_z: float


@dataclass
class DetectionOutput:
    series: list[AnomalyDetectionSeriesPoint]
    events: list[DetectionEvent]


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _median_positive_delta_seconds(points: list[SignalPoint]) -> float:
    deltas = [
        (current.timestamp - previous.timestamp).total_seconds()
        for previous, current in zip(points, points[1:])
        if current.timestamp > previous.timestamp
    ]
    return statistics.median(deltas) if deltas else 1.0


def detect(points: list[SignalPoint], config: AnomalyDetectionConfig) -> DetectionOutput:
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

    baseline_buffer: list[tuple[float, float]] = []
    normal_clock = 0.0
    first_baseline_clock: float | None = None
    state = "normal"
    smoothed = ordered[0].score
    cusum = 0.0
    active: DetectionEvent | None = None
    recovery_started: datetime | None = None
    output_events: list[DetectionEvent] = []
    series: list[AnomalyDetectionSeriesPoint] = []
    previous_timestamp: datetime | None = None

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

    for point in ordered:
        dt = 0.0 if previous_timestamp is None else max(0.0, (point.timestamp - previous_timestamp).total_seconds())
        is_gap = previous_timestamp is not None and dt > gap_seconds
        if is_gap:
            close_active(previous_timestamp, "data_gap")
            smoothed = point.score
            dt = 0.0
        elif previous_timestamp is None:
            smoothed = point.score
        else:
            alpha = 1.0 - math.exp(-math.log(2.0) * dt / half_life_seconds) if dt > 0 else 0.0
            smoothed = alpha * point.score + (1.0 - alpha) * smoothed

        if state == "normal":
            normal_clock += dt
            cutoff = normal_clock - baseline_window_seconds
            baseline_buffer = [(clock, value) for clock, value in baseline_buffer if clock >= cutoff]

        values = [value for _, value in baseline_buffer]
        baseline = statistics.median(values) if values else None
        mad = statistics.median([abs(value - baseline) for value in values]) if baseline is not None else None
        scale = None
        if baseline is not None and mad is not None:
            scale = max(1.4826 * mad, abs(baseline) * 1e-6, 1e-12)
        baseline_span = 0.0 if first_baseline_clock is None else normal_clock - first_baseline_clock
        ready = len(values) >= config.minimum_warmup_points and baseline_span >= warmup_seconds

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
        previous_timestamp = point.timestamp

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


def _visible_output(output: DetectionOutput, start: datetime, end: datetime) -> DetectionOutput:
    series = [point for point in output.series if start <= point.timestamp <= end]
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
        visible_event_points = [
            point for point in output.series if warning_start <= point.timestamp <= end_timestamp
        ]
        peak_point = max(visible_event_points, key=lambda point: point.score) if visible_event_points else None
        events.append(DetectionEvent(
            warning_start=warning_start,
            confirmed_at=confirmed_at,
            end_timestamp=end_timestamp,
            end_reason=event.end_reason if event.end_timestamp <= end else "range_end",
            peak_timestamp=peak_point.timestamp if peak_point else min(max(event.peak_timestamp, warning_start), end_timestamp),
            max_score=peak_point.score if peak_point else event.max_score,
            max_robust_z=max(
                (point.robust_z for point in visible_event_points if point.robust_z is not None),
                default=event.max_robust_z,
            ),
        ))
    return DetectionOutput(series=series, events=events)


def _compute_for_run(db: Session, run: models.AnomalyDetectionRun) -> DetectionOutput:
    config = AnomalyDetectionConfig.model_validate(run.config)
    points = _load_points(
        db,
        run.testing_run_id,
        run.score_series,
        run.start_timestamp,
        run.end_timestamp,
        config.preroll_minutes,
    )
    return _visible_output(detect(points, config), run.start_timestamp, run.end_timestamp)


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


def create_run(db: Session, payload: AnomalyDetectionRunCreate) -> AnomalyDetectionRunRead:
    testing_run = db.get(models.TestingRun, payload.testing_run_id)
    if testing_run is None:
        raise ValueError("Inference run not found.")
    if testing_run.status != "finished":
        raise ValueError("Only finished inference runs can be analyzed.")

    name = payload.name.strip()
    if not name:
        raise ValueError("Run name is required.")
    run = models.AnomalyDetectionRun(
        name=name,
        testing_run_id=testing_run.id,
        testing_run_name=testing_run.name,
        score_series=payload.score_series,
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        algorithm_version=ALGORITHM_VERSIONS[payload.config.algorithm],
        config=payload.config.model_dump(mode="json"),
    )
    db.add(run)
    db.flush()
    output = _compute_for_run(db, run)
    if not output.series:
        db.rollback()
        series_label = payload.score_series.replace("_", " ")
        raise ValueError(f"No {series_label} values exist in the selected time range.")
    run.point_count = len(output.series)
    run.warning_count = len(output.events)
    run.anomaly_count = sum(event.confirmed_at is not None for event in output.events)
    for event in output.events:
        run.events.append(models.AnomalyDetectionEvent(**event.__dict__))
    db.commit()
    return get_run(db, run.id)  # type: ignore[return-value]


def get_run(db: Session, run_id: int, *, max_points: int | None = 8000) -> AnomalyDetectionRunRead | None:
    run = db.scalar(
        select(models.AnomalyDetectionRun)
        .where(models.AnomalyDetectionRun.id == run_id)
        .options(selectinload(models.AnomalyDetectionRun.events))
    )
    if run is None:
        return None
    output = _compute_for_run(db, run)
    visible_series, decimated = _decimate_series(output.series, max_points)
    summary = AnomalyDetectionRunSummary.model_validate(run)
    return AnomalyDetectionRunRead(
        **summary.model_dump(),
        events=[AnomalyDetectionEventRead.model_validate(event) for event in run.events],
        series=visible_series,
        total=len(output.series),
        decimated=decimated,
    )


def delete_run(db: Session, run_id: int) -> bool:
    run = db.get(models.AnomalyDetectionRun, run_id)
    if run is None:
        return False
    db.delete(run)
    db.commit()
    return True


def delete_runs_for_testing_run(db: Session, testing_run_id: int) -> None:
    db.execute(delete(models.AnomalyDetectionRun).where(models.AnomalyDetectionRun.testing_run_id == testing_run_id))
