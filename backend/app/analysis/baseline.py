from __future__ import annotations

from bisect import bisect_left
from datetime import UTC, datetime
import hashlib
import json
import math
import statistics
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.schemas import (
    BaselineAnomalyEventRead,
    BaselineNormalizationRequest,
    BaselineNormalizationResponse,
    BaselineRegionStatisticsRead,
    BaselineSeriesPointRead,
    BaselineStatisticsRead,
    BaselineThresholdStatisticsRead,
    BaselineTraceResultRead,
)


def _number(params: dict[str, Any], key: str, fallback: float) -> float:
    value = params.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return fallback


def _string(params: dict[str, Any], key: str, fallback: str) -> str:
    value = params.get(key)
    return value if isinstance(value, str) else fallback


def _boolean(params: dict[str, Any], key: str, fallback: bool) -> bool:
    value = params.get(key)
    return value if isinstance(value, bool) else fallback


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def _ewma(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    alpha = min(1.0, max(0.0, alpha))
    output = [values[0]]
    for value in values[1:]:
        output.append(alpha * value + (1.0 - alpha) * output[-1])
    return output


def _derivative(values: list[float], times: list[float], time_normalized: bool) -> list[float]:
    output = [0.0]
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        seconds = (times[index] - times[index - 1]) / 1000.0
        if not math.isfinite(seconds) or seconds <= 0:
            seconds = 1.0
        output.append(delta / seconds if time_normalized else delta)
    return output


def _window_start(times: list[float], index: int, params: dict[str, Any], sample_key: str = "windowSamples", minute_key: str = "windowMinutes") -> int:
    if _string(params, "windowMode", "samples") == "minutes":
        cutoff = times[index] - max(0.0, _number(params, minute_key, 3.0)) * 60_000.0
        return bisect_left(times, cutoff, 0, index + 1)
    samples = max(1, int(_number(params, sample_key, 12.0)))
    return max(0, index - samples + 1)


def _rolling(values: list[float], times: list[float], params: dict[str, Any], reducer: Callable[[list[float]], float], sample_key: str = "windowSamples", minute_key: str = "windowMinutes") -> list[float]:
    return [reducer(values[_window_start(times, index, params, sample_key, minute_key): index + 1]) for index in range(len(values))]


def _robust_z(values: list[float], times: list[float], params: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    smooth = _ewma(values, _number(params, "alpha", 0.2))
    baseline = _rolling(smooth, times, params, _median)
    mad: list[float] = []
    for index, center in enumerate(baseline):
        start = _window_start(times, index, params)
        mad.append(_median([abs(value - center) for value in smooth[start:index + 1]]))
    epsilon = _number(params, "epsilon", 1e-12)
    z = [(value - baseline[index]) / (1.4826 * mad[index] + epsilon) for index, value in enumerate(smooth)]
    return z, baseline, mad


def _positive_exceedance(values: list[float], times: list[float], params: dict[str, Any]) -> list[float]:
    z, _, _ = _robust_z(values, times, params)
    threshold = _number(params, "threshold", _number(params, "zThreshold", 1.0))
    return [max(0.0, value - threshold) for value in z]


def _rolling_area(values: list[float], times: list[float], params: dict[str, Any], sample_key: str = "windowSamples", minute_key: str = "windowMinutes") -> list[float]:
    exceedance = _positive_exceedance(values, times, params)
    return [sum(exceedance[_window_start(times, index, params, sample_key, minute_key): index + 1]) for index in range(len(values))]


def compute_analytics(kind: str, params: dict[str, Any], values: list[float], timestamps: list[datetime]) -> list[float | None]:
    times = [timestamp.timestamp() * 1000.0 for timestamp in timestamps]
    alpha = _number(params, "alpha", 0.2)
    base = values if _string(params, "source", "smoothed") == "raw" else _ewma(values, alpha)
    derivative = _derivative(base, times, _boolean(params, "timeNormalized", False))

    def finite(output: list[float]) -> list[float | None]:
        return [value if math.isfinite(value) else None for value in output]

    if kind == "raw":
        return finite(values)
    if kind == "ewma":
        return finite(_ewma(values, alpha))
    if kind == "derivative":
        return finite(derivative)
    if kind == "smoothed_derivative":
        return finite(_ewma(derivative, _number(params, "beta", 0.2)))
    if kind == "second_derivative":
        return finite(_derivative(_ewma(derivative, _number(params, "beta", 0.2)), times, _boolean(params, "timeNormalized", False)))
    if kind == "rolling_slope":
        output: list[float] = []
        for index, value in enumerate(base):
            start = _window_start(times, index, params)
            denominator = max(1.0, (times[index] - times[start]) / 1000.0) if _boolean(params, "timeNormalized", False) else max(1, index - start)
            output.append((value - base[start]) / denominator)
        return finite(output)
    if kind == "rolling_median":
        return finite(_rolling(_ewma(values, alpha), times, params, _median))
    if kind in {"rolling_mad", "robust_z"}:
        z, _, mad = _robust_z(values, times, params)
        return finite(mad if kind == "rolling_mad" else z)
    if kind == "positive_exceedance":
        return finite(_positive_exceedance(values, times, params))
    if kind == "rolling_area":
        return finite(_rolling_area(values, times, params))
    if kind == "rolling_mean":
        return finite(_rolling(_ewma(values, alpha), times, params, _mean))
    if kind == "rolling_max":
        return finite(_rolling(_ewma(values, alpha), times, params, max))
    if kind == "drawdown":
        smooth = _ewma(values, alpha)
        maxima = _rolling(smooth, times, params, max)
        epsilon = _number(params, "epsilon", 1e-12)
        output = [maxima[index] - value for index, value in enumerate(smooth)]
        if _string(params, "mode", "relative") == "relative":
            output = [value / (maxima[index] + epsilon) for index, value in enumerate(output)]
        return finite(output)
    if kind in {"positive_slope_count", "positive_slope_fraction"}:
        threshold = _number(params, "slopeThreshold", 0.0)
        output = []
        for index in range(len(values)):
            start = _window_start(times, index, params)
            window = derivative[start:index + 1]
            count = sum(value > threshold for value in window)
            output.append(count / max(1, len(window)) if kind == "positive_slope_fraction" else float(count))
        return finite(output)
    if kind == "rising_streak":
        threshold = _number(params, "slopeThreshold", 0.0)
        streak = 0
        output = []
        for value in derivative:
            streak = streak + 1 if value > threshold else 0
            output.append(float(streak))
        return output
    if kind in {"cusum", "page_hinkley"}:
        z, _, _ = _robust_z(values, times, params)
        output = []
        accumulator = 0.0
        running_mean = 0.0
        for index, value in enumerate(z):
            if kind == "cusum":
                accumulator = max(0.0, accumulator + value - _number(params, "k", 1.0))
            else:
                running_mean += (value - running_mean) / (index + 1)
                accumulator = max(0.0, accumulator + value - running_mean - _number(params, "delta", 0.2))
            output.append(accumulator)
        return finite(output)
    if kind == "evidence_score":
        z, _, _ = _robust_z(values, times, params)
        smooth_d = _ewma(derivative, _number(params, "beta", 0.2))
        draw_params = {**params, "mode": "relative"}
        drawdown = [value or 0.0 for value in compute_analytics("drawdown", draw_params, values, timestamps)]
        evidence = 0.0
        output = []
        for index, value in enumerate(z):
            z_threshold = _number(params, "zThreshold", 1.0)
            slope_threshold = _number(params, "slopeThreshold", 0.0)
            positive = (_number(params, "w1", 1.0) * max(0.0, value - z_threshold)
                        + _number(params, "w2", 1.0) * max(0.0, smooth_d[index] - slope_threshold)
                        + _number(params, "w3", 0.2) * (1.0 if smooth_d[index] > slope_threshold else 0.0))
            negative = (_number(params, "v1", 1.0) * max(0.0, -smooth_d[index])
                        + _number(params, "v2", 0.5) * (1.0 if value < z_threshold else 0.0)
                        + _number(params, "v3", 1.0) * drawdown[index])
            evidence = max(0.0, evidence + positive - negative)
            output.append(evidence)
        return finite(output)
    if kind == "slope_height_ratio":
        z, _, _ = _robust_z(values, times, params)
        epsilon = _number(params, "epsilon", 1e-12)
        return finite([value / (abs(z[index]) + epsilon) for index, value in enumerate(derivative)])
    if kind == "energy_ratio":
        short = _rolling_area(values, times, params)
        long = _rolling_area(values, times, params, "longWindowSamples", "longWindowMinutes")
        epsilon = _number(params, "epsilon", 1e-12)
        return finite([value / (long[index] + epsilon) for index, value in enumerate(short)])
    if kind in {"snr_db", "snr_ratio"}:
        epsilon = _number(params, "epsilon", 1e-12)
        output: list[float | None] = []
        for index in range(len(values)):
            start = _window_start(times, index, params)
            window = [value for value in values[start:index + 1] if math.isfinite(value)]
            if len(window) < 2:
                output.append(None)
                continue
            ratio = abs(_mean(window)) / (_std(window) + epsilon)
            value = ratio if kind == "snr_ratio" else 20.0 * math.log10(ratio) if ratio > 0 else -math.inf
            output.append(value if math.isfinite(value) else None)
        return output
    if kind in {"rolling_std", "rolling_cv"}:
        smooth = _ewma(values, alpha)
        means = _rolling(smooth, times, params, _mean)
        stds = _rolling(smooth, times, params, _std)
        if kind == "rolling_std":
            return finite(stds)
        epsilon = _number(params, "epsilon", 1e-12)
        return finite([value / (means[index] + epsilon) for index, value in enumerate(stds)])
    if kind == "time_since_onset":
        z, _, _ = _robust_z(values, times, params)
        onset: float | None = None
        output = []
        for index, value in enumerate(z):
            if onset is None and value > _number(params, "onsetThreshold", 1.0) and derivative[index] > _number(params, "slopeThreshold", 0.0):
                onset = times[index]
            if onset is not None and value < _number(params, "resetThreshold", 0.5):
                onset = None
            output.append(0.0 if onset is None else (times[index] - onset) / 1000.0)
        return finite(output)
    if kind == "state_machine":
        z, _, _ = _robust_z(values, times, params)
        cusum = [value or 0.0 for value in compute_analytics("cusum", params, values, timestamps)]
        state = 0.0
        output = []
        for index, value in enumerate(z):
            if cusum[index] >= _number(params, "hHigh", 10.0):
                state = 3.0
            elif cusum[index] >= _number(params, "hLow", 5.0):
                state = 2.0
            elif value > _number(params, "lowThreshold", 1.0) and derivative[index] > _number(params, "slopeThreshold", 0.0):
                state = 1.0
            elif state > 0 and value < _number(params, "offThreshold", 0.5):
                state = 0.0
            output.append(state)
        return output
    return finite(values)


def _score(row: models.TestingRunResult, series: str) -> float:
    metadata = row.result_metadata or {}
    fast = metadata.get("fast_anogan")
    if isinstance(fast, dict):
        mapping = {"fast_residual": "residual_score", "fast_feature": "feature_score", "fast_combined": "combined_score"}
        value = fast.get(mapping.get(series, ""))
        if isinstance(value, (int, float)):
            return float(value)
    if series == "reconstruction":
        value = metadata.get("reconstruction_score")
        return float(value) if isinstance(value, (int, float)) else float(row.full_mse)
    if series == "prediction":
        value = metadata.get("prediction_score")
        return float(value) if isinstance(value, (int, float)) else float(row.roi_mse if row.roi_mse is not None else row.score)
    if series.startswith("future+"):
        try:
            horizon = int(series.split("+", 1)[1])
        except ValueError:
            horizon = -1
        for item in metadata.get("future_scores", []):
            if isinstance(item, dict) and item.get("horizon") == horizon and isinstance(item.get("score"), (int, float)):
                return float(item["score"])
    return float(row.score if row.score is not None else row.roi_mse if row.roi_mse is not None else row.full_mse)


def _moving_average(values: list[float], size: int) -> list[float]:
    if size <= 1:
        return values[:]
    output: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= size:
            running -= values[index - size]
        output.append(running / min(size, index + 1))
    return output


def _in_regions(timestamp: datetime, regions: list[Any]) -> bool:
    return any(_naive(region.start) <= timestamp <= _naive(region.end) for region in regions)


def _naive(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _optional_stat(values: list[float], reducer: Callable[[list[float]], float]) -> float | None:
    return reducer(values) if values else None


def _longest_above(timestamps: list[datetime], z_values: list[float], threshold: float) -> float:
    positive_deltas = sorted((timestamps[index] - timestamps[index - 1]).total_seconds() for index in range(1, len(timestamps)) if timestamps[index] > timestamps[index - 1])
    typical = statistics.median(positive_deltas) if positive_deltas else 0.0
    gap_limit = max(15.0, 5.0 * typical)
    longest = 0.0
    start: datetime | None = None
    previous: datetime | None = None
    for timestamp, value in zip(timestamps, z_values, strict=True):
        if value > threshold:
            if start is None or (previous is not None and (timestamp - previous).total_seconds() > gap_limit):
                start = timestamp
            longest = max(longest, (timestamp - start).total_seconds())
            previous = timestamp
        else:
            start = None
            previous = None
    return longest


def _anomaly_events(
    timestamps: list[datetime],
    z_values: list[float | None],
    thresholds: list[float],
    persistence_samples: int,
) -> list[BaselineAnomalyEventRead]:
    positive_deltas = sorted(
        (timestamps[index] - timestamps[index - 1]).total_seconds()
        for index in range(1, len(timestamps))
        if timestamps[index] > timestamps[index - 1]
    )
    typical = statistics.median(positive_deltas) if positive_deltas else 0.0
    gap_limit = max(15.0, 5.0 * typical)
    events: list[BaselineAnomalyEventRead] = []
    for threshold in thresholds:
        sequence_start: int | None = None
        for index in range(len(timestamps) + 1):
            value = z_values[index] if index < len(z_values) else None
            gap = (
                index > 0
                and index < len(timestamps)
                and (timestamps[index] - timestamps[index - 1]).total_seconds() > gap_limit
            )
            above = value is not None and math.isfinite(value) and value > threshold and not gap
            if above and sequence_start is None:
                sequence_start = index
            if above:
                continue
            if sequence_start is not None:
                sequence_end = index - 1
                count = sequence_end - sequence_start + 1
                if count >= persistence_samples:
                    events.append(BaselineAnomalyEventRead(
                        threshold=threshold,
                        start=timestamps[sequence_start],
                        end=timestamps[sequence_end],
                        sample_count=count,
                    ))
                sequence_start = None
            if gap and value is not None and math.isfinite(value) and value > threshold:
                sequence_start = index
    return events


def _decimate(points: list[BaselineSeriesPointRead], max_points: int) -> tuple[list[BaselineSeriesPointRead], bool]:
    if len(points) <= max_points:
        return points, False
    if max_points <= 0:
        return [], True
    if max_points == 1:
        return [points[0]], True
    step = (len(points) - 1) / (max_points - 1)
    indices = sorted({round(index * step) for index in range(max_points)} | {0, len(points) - 1})
    return [points[index] for index in indices], True


def _region_point_budgets(point_counts: list[int], max_points: int) -> list[int]:
    """Distribute one trace's plot budget proportionally across its regions."""
    budgets = [0] * len(point_counts)
    nonempty = [index for index, count in enumerate(point_counts) if count > 0]
    if not nonempty:
        return budgets
    if max_points < len(nonempty):
        for index in sorted(nonempty, key=lambda item: point_counts[item], reverse=True)[:max_points]:
            budgets[index] = 1
        return budgets
    # A region needs both endpoints when possible. For exceptionally many
    # regions, retaining one point per region is more useful than omitting a
    # selected region from the response entirely.
    minimum = 2 if max_points >= 2 * len(nonempty) else 1
    for index in nonempty:
        budgets[index] = min(minimum, point_counts[index])
    remaining = max(0, max_points - sum(budgets))
    unmet = [max(0, point_counts[index] - budgets[index]) for index in range(len(point_counts))]
    total_unmet = sum(unmet)
    if remaining == 0 or total_unmet == 0:
        return budgets
    shares = [remaining * count / total_unmet for count in unmet]
    for index, share in enumerate(shares):
        addition = min(unmet[index], math.floor(share))
        budgets[index] += addition
        remaining -= addition
    for index in sorted(range(len(point_counts)), key=lambda item: shares[item] - math.floor(shares[item]), reverse=True):
        if remaining <= 0:
            break
        if budgets[index] < point_counts[index]:
            budgets[index] += 1
            remaining -= 1
    return budgets


def calculate(db: Session, payload: BaselineNormalizationRequest) -> BaselineNormalizationResponse:
    trace_results: list[BaselineTraceResultRead] = []
    for trace in payload.traces:
        run = db.get(models.TestingRun, trace.testing_run_id)
        if run is None or run.status != "finished":
            raise ValueError(f"Finished testing run #{trace.testing_run_id} was not found.")
        trace_start = _naive(trace.start)
        trace_end = _naive(trace.end)
        rows = list(db.scalars(
            select(models.TestingRunResult)
            .where(models.TestingRunResult.testing_run_id == trace.testing_run_id)
            .where(models.TestingRunResult.timestamp >= trace_start)
            .where(models.TestingRunResult.timestamp <= trace_end)
            .order_by(models.TestingRunResult.timestamp, models.TestingRunResult.position)
        ))[::payload.sampling]
        if not rows:
            raise ValueError(f"Testing run #{trace.testing_run_id} has no results in the selected plot range.")

        timestamps = [row.timestamp for row in rows]
        raw = [_score(row, payload.score_series) for row in rows]
        signal: list[float | None] = _moving_average(raw, payload.moving_average) if not payload.analytics_pipeline else raw[:]
        if payload.stage_index >= 0:
            current = raw[:]
            for method in payload.analytics_pipeline[:payload.stage_index + 1]:
                output = compute_analytics(method.kind, method.params, current, timestamps)
                current = [value if value is not None else math.nan for value in output]
            signal = output

        baseline_values = [float(value) for timestamp, value in zip(timestamps, signal, strict=True) if value is not None and math.isfinite(value) and _in_regions(timestamp, payload.baseline_regions)]
        if len(baseline_values) < 2:
            raise ValueError(f"Trace '{trace.label}' has fewer than two valid baseline samples.")
        mean = _mean(baseline_values)
        std = _std(baseline_values)
        median = _median(baseline_values)
        mad = _median([abs(value - median) for value in baseline_values])
        center = mean if payload.normalization == "classic" else median
        raw_scale = std if payload.normalization == "classic" else 1.4826 * mad
        scale = max(raw_scale, 1e-12, abs(center) * 1e-12)
        z: list[float | None] = [None if value is None or not math.isfinite(value) else (value - center) / scale for value in signal]
        baseline = BaselineStatisticsRead(sample_count=len(baseline_values), mean=mean, std=std, median=median, mad=mad, center=center, scale=scale)

        region_payloads: list[dict[str, Any]] = []
        for region in payload.analysis_regions:
            region_start = _naive(region.start)
            region_end = _naive(region.end)
            region_indices = [index for index, timestamp in enumerate(timestamps) if region_start <= timestamp <= region_end]
            indices = [
                index for index in region_indices
                if signal[index] is not None
                and z[index] is not None
                and math.isfinite(float(signal[index]))
                and math.isfinite(float(z[index]))
            ]
            raw_values = [raw[index] for index in indices if math.isfinite(raw[index])]
            signal_values = [float(signal[index]) for index in indices if signal[index] is not None and math.isfinite(float(signal[index]))]
            z_values = [float(z[index]) for index in indices if z[index] is not None and math.isfinite(float(z[index]))]
            region_times = [timestamps[index] for index in indices]
            threshold_results = []
            for threshold in payload.thresholds:
                count = sum(value > threshold for value in z_values)
                threshold_results.append(BaselineThresholdStatisticsRead(
                    threshold=threshold,
                    sample_count=count,
                    sample_fraction=count / len(z_values) if z_values else 0.0,
                    longest_seconds=_longest_above(region_times, z_values, threshold),
                ))
            region_points = [BaselineSeriesPointRead(
                timestamp=timestamps[index],
                raw=raw[index] if math.isfinite(raw[index]) else None,
                signal=(float(signal[index]) if signal[index] is not None and math.isfinite(float(signal[index])) else None),
                z=(float(z[index]) if z[index] is not None and math.isfinite(float(z[index])) else None),
            ) for index in region_indices]
            region_payloads.append({
                "region_id": region.id,
                "region_name": region.name,
                "start": region_start,
                "end": region_end,
                "sample_count": len(signal_values),
                "raw_mean": _optional_stat(raw_values, _mean),
                "raw_max": _optional_stat(raw_values, max),
                "signal_mean": _optional_stat(signal_values, _mean),
                "signal_max": _optional_stat(signal_values, max),
                "signal_std": _optional_stat(signal_values, _std),
                "z_mean": _optional_stat(z_values, _mean),
                "z_median": _optional_stat(z_values, _median),
                "z_max": _optional_stat(z_values, max),
                "thresholds": threshold_results,
                "full_series": region_points,
                "events": _anomaly_events(
                    [point.timestamp for point in region_points],
                    [point.z for point in region_points],
                    payload.thresholds,
                    payload.persistence_samples,
                ),
            })

        budgets = _region_point_budgets(
            [len(region["full_series"]) for region in region_payloads], payload.max_points
        )
        region_results: list[BaselineRegionStatisticsRead] = []
        for region, budget in zip(region_payloads, budgets, strict=True):
            full_series = region.pop("full_series")
            visible, decimated = _decimate(full_series, budget) if full_series else ([], False)
            region_results.append(BaselineRegionStatisticsRead(
                **region,
                series=visible,
                total_points=len(full_series),
                decimated=decimated,
            ))
        fingerprint_payload = {
            "run": run.id,
            "updated": run.updated_at.isoformat() if run.updated_at else None,
            "count": len(rows),
            "first": rows[0].id,
            "last": rows[-1].id,
            "first_timestamp": rows[0].timestamp.isoformat(),
            "last_timestamp": rows[-1].timestamp.isoformat(),
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
        trace_results.append(BaselineTraceResultRead(
            testing_run_id=run.id,
            label=trace.label,
            color=trace.color,
            fingerprint=fingerprint,
            baseline=baseline,
            regions=region_results,
            series=[],
            events=[],
            total_points=sum(region.total_points for region in region_results),
            decimated=any(region.decimated for region in region_results),
        ))
    return BaselineNormalizationResponse(
        computed_at=datetime.now(UTC),
        normalization=payload.normalization,
        thresholds=payload.thresholds,
        persistence_samples=payload.persistence_samples,
        traces=trace_results,
    )
