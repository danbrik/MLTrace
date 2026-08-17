from datetime import datetime, timedelta
import math
import random
import statistics

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.anomaly_detection.service import SignalPoint, detect
from app.anomaly_detection import service as anomaly_service
from app.database import Base, get_db
from app.main import app
from app.schemas import AnomalyDetectionConfig
from app.schemas import AnomalyDetectionCalibrationRequest
from app.schemas import AnomalyDetectionRunCreate
from app.schemas import AnomalyDetectionThresholdPreviewRequest
from app.testing import service as testing_service


BASE = datetime(2026, 1, 1)


def _points(values: list[float], *, seconds: int = 60) -> list[SignalPoint]:
    return [SignalPoint(BASE + timedelta(seconds=index * seconds), value) for index, value in enumerate(values)]


def _fast_config(**overrides) -> AnomalyDetectionConfig:
    values = {
        "smoothing_half_life_minutes": 0.1,
        "baseline_window_minutes": 10,
        "warmup_minutes": 2,
        "minimum_warmup_points": 3,
        "warning_z": 3,
        "high_z": 5,
        "cusum_drift": 1,
        "cusum_threshold": 10,
        "confirmation_minutes": 2,
        "recovery_z": 1,
        "recovery_minutes": 2,
        "preroll_minutes": 10,
        "minimum_gap_minutes": 15,
    }
    values.update(overrides)
    return AnomalyDetectionConfig(**values)


def _event_config(**overrides) -> AnomalyDetectionConfig:
    values = {
        "algorithm": "event_threshold",
        "event_smoothing_enabled": False,
        "event_smoothing_method": "median",
        "event_smoothing_window_seconds": 5,
        "threshold_mode": "manual",
        "manual_threshold": 1.0,
        "persistence_k": 2,
        "persistence_n": 3,
        "threshold_off_factor": 0.8,
        "normal_close_seconds": 2,
        "merge_gap_seconds": 0,
        "event_minimum_gap_seconds": 10,
        "gap_multiplier": 5,
        "preroll_minutes": 0,
    }
    values.update(overrides)
    return AnomalyDetectionConfig(**values)


def _sigma_config(**overrides) -> AnomalyDetectionConfig:
    values = {
        "algorithm": "rolling_sigma",
        "baseline_window_minutes": 10,
        "warmup_minutes": 0,
        "minimum_warmup_points": 3,
        "sigma_threshold": 3,
        "preroll_minutes": 10,
        "gap_multiplier": 5,
        "minimum_gap_minutes": 10,
    }
    values.update(overrides)
    return AnomalyDetectionConfig(**values)


def test_constant_signal_is_stable_when_mad_is_zero() -> None:
    output = detect(_points([1.0] * 40), _fast_config())
    assert output.events == []
    ready = [point for point in output.series if point.state != "warmup"]
    assert ready
    assert all(point.robust_z == 0 for point in ready)
    assert all(point.warning_threshold is not None for point in ready)


def test_scale_floor_limits_small_residual_after_zero_mad_baseline() -> None:
    output = detect(_points([1.0] * 12 + [1.0001]), _fast_config())
    point = output.series[-1]

    assert point.mad == 0
    assert point.scale == pytest.approx(0.001)
    assert point.robust_z == pytest.approx(0.1, rel=0.02)
    assert point.state == "normal"


def test_cusum_increment_is_capped_and_frozen_after_confirmation() -> None:
    output = detect(
        _points([1.0] * 12 + [2.0] * 4),
        _fast_config(
            confirmation_mode="samples",
            confirmation_samples=1,
            cusum_z_cap=6,
        ),
    )
    confirmed = [point for point in output.series if point.state == "confirmed"]

    assert confirmed
    assert confirmed[0].cusum_increment == pytest.approx(5.0)
    assert all(point.cusum_increment == 0 for point in confirmed[1:])
    assert len({point.cusum for point in confirmed}) == 1


def test_brief_peak_remains_unconfirmed() -> None:
    values = [1.0] * 12 + [3.0] + [1.0] * 12
    output = detect(_points(values), _fast_config(confirmation_minutes=3))
    assert len(output.events) == 1
    assert output.events[0].confirmed_at is None
    assert output.events[0].end_reason == "recovered"


def test_robust_zscore_does_not_accumulate_cusum() -> None:
    values = [1.0] * 12 + [3.0] * 10 + [1.0] * 5
    output = detect(_points(values), _fast_config(algorithm="robust_zscore"))
    assert len(output.events) == 1
    assert output.events[0].confirmed_at is not None
    assert all(point.cusum == 0 for point in output.series)


def test_robust_zscore_can_confirm_after_consecutive_samples() -> None:
    values = [1.0] * 12 + [3.0] * 4 + [1.0] * 5
    output = detect(
        _points(values),
        _fast_config(
            algorithm="robust_zscore",
            confirmation_mode="samples",
            confirmation_samples=3,
        ),
    )

    assert len(output.events) == 1
    assert output.events[0].confirmed_at == output.events[0].warning_start + timedelta(minutes=2)


def test_sustained_shift_confirms_and_frozen_baseline_does_not_follow() -> None:
    values = [1.0] * 12 + [3.0] * 10 + [1.0] * 5
    output = detect(_points(values), _fast_config())
    assert len(output.events) == 1
    event = output.events[0]
    assert event.confirmed_at is not None
    assert event.confirmed_at >= event.warning_start + timedelta(minutes=2)
    event_points = [point for point in output.series if event.warning_start <= point.timestamp <= event.end_timestamp]
    baselines = {point.baseline for point in event_points if point.baseline is not None}
    assert baselines == {1.0}
    assert event.end_reason == "recovered"


def test_fallback_recovery_closes_event_below_warning() -> None:
    values = [1.0] * 12 + [1.01] * 2 + [1.002] * 4
    output = detect(
        _points(values),
        _fast_config(
            confirmation_mode="samples",
            confirmation_samples=1,
            fallback_recovery_minutes=2,
        ),
    )

    assert len(output.events) == 1
    assert output.events[0].confirmed_at is not None
    assert output.events[0].end_reason == "recovered"
    assert output.series[-1].state == "normal"


def test_large_gap_closes_active_event() -> None:
    points = _points([1.0] * 8 + [3.0] * 4)
    points.append(SignalPoint(points[-1].timestamp + timedelta(hours=1), 1.0))
    output = detect(points, _fast_config(confirmation_minutes=1))
    assert output.events
    assert output.events[0].end_reason == "data_gap"
    assert output.series[-1].state == "warmup"
    assert output.series[-1].baseline is None
    assert output.series[-1].mad is None


def test_event_threshold_median_smoothing_removes_single_peak() -> None:
    output = detect(
        _points([0.02, 0.02, 0.09, 0.02, 0.02], seconds=1),
        _event_config(
            event_smoothing_enabled=True,
            event_smoothing_method="median",
            persistence_k=1,
            persistence_n=1,
            manual_threshold=0.05,
        ),
    )
    assert [point.score for point in output.series] == [0.02, 0.02, 0.09, 0.02, 0.02]
    assert [point.smoothed for point in output.series] == [0.02] * 5
    assert output.events == []


def test_event_threshold_moving_average_uses_causal_time_window() -> None:
    points = [
        SignalPoint(BASE, 1.0),
        SignalPoint(BASE + timedelta(seconds=2), 3.0),
        SignalPoint(BASE + timedelta(seconds=5), 9.0),
    ]
    output = detect(
        points,
        _event_config(
            event_smoothing_enabled=True,
            event_smoothing_method="moving_average",
            manual_threshold=100,
        ),
    )
    assert [point.smoothed for point in output.series] == [1.0, 2.0, 6.0]


def test_event_threshold_triggers_at_kth_candidate_and_recovers_with_hysteresis() -> None:
    values = [0.2, 1.2, 0.9, 1.3, 0.9, 0.7, 0.75, 0.7]
    output = detect(_points(values, seconds=1), _event_config())
    assert len(output.events) == 1
    event = output.events[0]
    assert event.warning_start == BASE + timedelta(seconds=3)
    assert event.confirmed_at == event.warning_start
    assert event.end_timestamp == BASE + timedelta(seconds=7)
    assert event.end_reason == "recovered"
    assert event.duration_seconds == 4
    assert event.max_score == 1.3
    assert event.max_smoothed_score == 1.3
    assert event.threshold == 1.0


def test_event_threshold_merges_close_events_but_not_across_data_gap() -> None:
    config = _event_config(
        persistence_k=1,
        persistence_n=1,
        normal_close_seconds=1,
        merge_gap_seconds=5,
    )
    merged = detect(_points([1.2, 0.2, 0.2, 1.5, 0.2, 0.2], seconds=1), config)
    assert len(merged.events) == 1
    assert merged.events[0].warning_start == BASE
    assert merged.events[0].end_timestamp == BASE + timedelta(seconds=5)
    assert merged.events[0].max_smoothed_score == 1.5
    assert merged.events[0].mean_smoothed_score == statistics.fmean([1.2, 0.2, 0.2, 1.5, 0.2, 0.2])

    points = _points([1.2, 0.2], seconds=1)
    points.extend([
        SignalPoint(BASE + timedelta(seconds=20), 1.5),
        SignalPoint(BASE + timedelta(seconds=21), 0.2),
    ])
    separated = detect(points, _event_config(
        persistence_k=1,
        persistence_n=1,
        normal_close_seconds=30,
        merge_gap_seconds=60,
        event_minimum_gap_seconds=5,
    ))
    assert len(separated.events) == 2
    assert separated.events[0].end_reason == "data_gap"


def test_rolling_sigma_uses_raw_score_and_triggers_immediately() -> None:
    output = detect(_points([1.0, 2.0, 3.0, 10.0, 2.0], seconds=1), _sigma_config())

    assert [point.smoothed for point in output.series] == [point.score for point in output.series]
    anomaly_point = output.series[3]
    assert anomaly_point.baseline == pytest.approx(2.0)
    assert anomaly_point.baseline_std == pytest.approx(statistics.pstdev([1.0, 2.0, 3.0]))
    assert anomaly_point.high_threshold == pytest.approx(
        2.0 + 3 * statistics.pstdev([1.0, 2.0, 3.0])
    )
    assert anomaly_point.state == "confirmed"
    assert len(output.events) == 1
    assert output.events[0].warning_start == BASE + timedelta(seconds=3)
    assert output.events[0].confirmed_at == output.events[0].warning_start
    assert output.events[0].end_reason == "recovered"


def test_rolling_sigma_requires_consecutive_samples_before_confirmation() -> None:
    output = detect(
        _points([1.0, 2.0, 3.0, 10.0, 2.0, 10.0, 11.0, 12.0, 2.0], seconds=1),
        _sigma_config(confirmation_mode="samples", confirmation_samples=3),
    )

    assert len(output.events) == 1
    event = output.events[0]
    assert event.warning_start == BASE + timedelta(seconds=5)
    assert event.confirmed_at == BASE + timedelta(seconds=7)
    assert output.series[3].state == "warning"
    assert output.series[4].state == "normal"
    assert [point.state for point in output.series[5:8]] == ["warning", "warning", "confirmed"]


def test_rolling_sigma_requires_continuous_minutes_before_confirmation() -> None:
    output = detect(
        _points([1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 2.0], seconds=30),
        _sigma_config(confirmation_mode="minutes", confirmation_minutes=1),
    )

    assert len(output.events) == 1
    assert output.events[0].warning_start == BASE + timedelta(seconds=90)
    assert output.events[0].confirmed_at == BASE + timedelta(seconds=150)


def test_rolling_sigma_freezes_baseline_during_anomaly() -> None:
    output = detect(
        _points([1.0, 1.1, 0.9, 4.0, 5.0, 6.0, 1.0], seconds=1),
        _sigma_config(),
    )

    anomaly_points = output.series[3:6]
    assert all(point.state == "confirmed" for point in anomaly_points)
    assert {round(point.baseline or 0.0, 10) for point in anomaly_points} == {1.0}
    assert len(output.events) == 1
    assert output.events[0].max_score == 6.0
    assert output.events[0].peak_timestamp == BASE + timedelta(seconds=5)


def test_rolling_sigma_constant_baseline_is_numerically_stable() -> None:
    output = detect(_points([1.0] * 20, seconds=1), _sigma_config())
    assert output.events == []
    ready = [point for point in output.series if point.state != "warmup"]
    assert ready
    assert all(point.robust_z == 0 for point in ready)
    assert all(point.baseline_std == pytest.approx(1e-6) for point in ready)


def test_rolling_sigma_data_gap_resets_baseline_and_warmup() -> None:
    points = _points([1.0, 1.1, 0.9, 4.0], seconds=1)
    points.extend([
        SignalPoint(BASE + timedelta(hours=1), 1.0),
        SignalPoint(BASE + timedelta(hours=1, seconds=1), 1.0),
    ])
    output = detect(points, _sigma_config(minimum_gap_minutes=1))

    assert output.events[0].end_reason == "data_gap"
    assert output.series[-2].state == "warmup"
    assert output.series[-1].state == "warmup"


def test_incremental_order_statistics_match_exact_median_and_mad() -> None:
    random.seed(11)
    values = [round(random.uniform(-3, 5), 3) for _ in range(200)]
    multiset = anomaly_service._FenwickMultiset(sorted(set(values)))
    active: list[float] = []
    for index, value in enumerate(values):
        multiset.add(value, 1)
        active.append(value)
        if index >= 37:
            expired = values[index - 37]
            multiset.add(expired, -1)
            active.remove(expired)
        expected_median = statistics.median(active)
        expected_mad = statistics.median(abs(item - expected_median) for item in active)
        assert multiset.median() == expected_median
        assert multiset.median_absolute_deviation(expected_median) == expected_mad


def test_detector_reports_real_smoothing_and_detection_progress() -> None:
    updates: list[tuple[str, int, int]] = []
    points = _points([1.0] * 20 + [3.0] * 8 + [1.0] * 5)
    detect(points, _fast_config(), lambda phase, completed, total, _message: updates.append((phase, completed, total)))
    assert updates[0] == ("smoothing", 0, len(points))
    assert ("smoothing", len(points), len(points)) in updates
    assert ("detecting", 0, len(points)) in updates
    assert updates[-1] == ("detecting", len(points), len(points))


def test_optimized_detector_matches_naive_exact_reference(monkeypatch) -> None:
    class NaiveExactMultiset:
        def __init__(self, _coordinates) -> None:
            self.values: list[float] = []

        @property
        def size(self) -> int:
            return len(self.values)

        def add(self, value: float, delta: int) -> None:
            if delta > 0:
                self.values.append(value)
            else:
                self.values.remove(value)

        def median(self) -> float | None:
            return statistics.median(self.values) if self.values else None

        def median_absolute_deviation(self, median: float) -> float | None:
            return statistics.median(abs(value - median) for value in self.values) if self.values else None

    random.seed(23)
    timestamps = [BASE]
    for index in range(1, 180):
        seconds = random.choice([0, 20, 60, 90])
        if index == 115:
            seconds = 3600
        timestamps.append(timestamps[-1] + timedelta(seconds=seconds))
    values = [1.0 + random.uniform(-0.08, 0.08) for _ in timestamps]
    for index in range(55, 90):
        values[index] += 1.5
    points = [SignalPoint(timestamp, value) for timestamp, value in zip(timestamps, values)]

    for algorithm in ("robust_zscore", "robust_cusum"):
        config = _fast_config(algorithm=algorithm, minimum_gap_minutes=10)
        optimized = detect(points, config)
        with monkeypatch.context() as context:
            context.setattr(anomaly_service, "_FenwickMultiset", NaiveExactMultiset)
            reference = detect(points, config)
        assert [point.model_dump() for point in optimized.series] == [
            point.model_dump() for point in reference.series
        ]
        assert [event.__dict__ for event in optimized.events] == [
            event.__dict__ for event in reference.events
        ]


def _seed_testing_run(
    db: Session,
    count: int = 80,
    *,
    name: str = "Crucible inference",
    values: list[float] | None = None,
    training_run_id: int = 1,
    inference_config: dict | None = None,
) -> models.TestingRun:
    run = models.TestingRun(
        name=name,
        training_run_id=training_run_id,
        training_dataset_id=1,
        status="finished",
        training_run_name="model",
        training_pipeline_name="pipeline",
        training_dataset_name="dataset",
        preprocessing_pipeline_name="preprocessing",
        method_type="autoencoder",
        method_family="autoencoder",
        training_mode="gradient",
        artifact_kind="weights",
        artifact_path="/tmp/model.pt",
        inference_config=inference_config,
    )
    db.add(run)
    db.flush()
    rows = []
    for index in range(count):
        value = values[index] if values is not None else (3.0 if 35 <= index < 55 else 1.0)
        rows.append({
            "testing_run_id": run.id,
            "position": index,
            "image_path": f"/tmp/{index}.tif",
            "timestamp": BASE + timedelta(minutes=index),
            "score": value,
            "full_mse": value,
            "roi_mse": None,
            "width": 8,
            "height": 8,
        })
    db.bulk_insert_mappings(models.TestingRunResult, rows)
    db.commit()
    return run


def test_quantile_preview_uses_only_selected_smoothed_validation_range() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        reference = _seed_testing_run(
            db,
            count=5,
            name="Normal validation",
            values=[100.0, 1.0, 9.0, 1.0, 100.0],
        )
        preview = anomaly_service.preview_threshold(db, AnomalyDetectionThresholdPreviewRequest(
            testing_run_id=reference.id,
            score_series="score",
            start_timestamp=BASE + timedelta(minutes=1),
            end_timestamp=BASE + timedelta(minutes=3),
            smoothing_enabled=True,
            smoothing_method="median",
            smoothing_window_seconds=300,
            quantile=0.5,
        ))
        assert preview.point_count == 3
        assert preview.threshold == 1.0


def test_calibration_profiles_are_monotonic_and_finite() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    values = [1.0 + 0.01 * math.sin(index / 7) + (index % 11) * 0.0002 for index in range(500)]
    with SessionLocal() as db:
        run = _seed_testing_run(db, count=len(values), values=values)
        config = _fast_config(algorithm="robust_cusum", baseline_window_minutes=30)
        results = [
            anomaly_service.preview_calibration(db, AnomalyDetectionCalibrationRequest(
                testing_run_id=run.id,
                start_timestamp=BASE,
                end_timestamp=BASE + timedelta(minutes=len(values) - 1),
                algorithm="robust_cusum",
                profile=profile,
                config=config,
            ))
            for profile in ("sensitive", "balanced", "conservative")
        ]

    for result in results:
        recommendation = result.recommendation
        assert math.isfinite(recommendation.minimum_scale_relative)
        assert math.isfinite(recommendation.minimum_scale_absolute)
        assert math.isfinite(recommendation.warning_z)
        assert math.isfinite(recommendation.high_z)
        assert recommendation.cusum_drift is not None and math.isfinite(recommendation.cusum_drift)
        assert recommendation.cusum_threshold is not None and math.isfinite(recommendation.cusum_threshold)
        assert result.metrics.confirmed_event_count == 0
    assert [result.recommendation.warning_z for result in results] == sorted(
        result.recommendation.warning_z for result in results
    )
    assert [result.recommendation.high_z for result in results] == sorted(
        result.recommendation.high_z for result in results
    )
    assert [result.recommendation.cusum_threshold for result in results] == sorted(
        result.recommendation.cusum_threshold for result in results
    )


def test_calibration_constant_signal_retains_floors_and_reports_low_confidence() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        run = _seed_testing_run(db, count=80, values=[1.0] * 80)
        config = _fast_config(algorithm="robust_zscore")
        result = anomaly_service.preview_calibration(db, AnomalyDetectionCalibrationRequest(
            testing_run_id=run.id,
            start_timestamp=BASE,
            end_timestamp=BASE + timedelta(minutes=79),
            algorithm="robust_zscore",
            profile="balanced",
            config=config,
        ))

    assert result.recommendation.minimum_scale_relative == config.minimum_scale_relative
    assert result.recommendation.minimum_scale_absolute == config.minimum_scale_absolute
    assert result.recommendation.warning_z == config.warning_z
    assert result.recommendation.high_z == config.high_z
    assert result.recommendation.cusum_drift is None
    assert result.recommendation.cusum_threshold is None
    assert result.confidence == "low"
    assert result.metrics.confirmed_event_count == 0
    assert any("MAD was zero" in warning for warning in result.warnings)


def test_calibration_near_zero_signal_produces_finite_scale_recommendations() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    values = [1e-10 + (index % 7) * 1e-12 for index in range(100)]
    with SessionLocal() as db:
        run = _seed_testing_run(db, count=len(values), values=values)
        result = anomaly_service.preview_calibration(db, AnomalyDetectionCalibrationRequest(
            testing_run_id=run.id,
            start_timestamp=BASE,
            end_timestamp=BASE + timedelta(minutes=len(values) - 1),
            algorithm="robust_cusum",
            profile="sensitive",
            config=_fast_config(),
        ))

    assert math.isfinite(result.recommendation.minimum_scale_absolute)
    assert math.isfinite(result.recommendation.minimum_scale_relative)
    assert result.recommendation.minimum_scale_absolute >= 0
    assert result.recommendation.minimum_scale_relative >= 0


def test_calibration_resets_after_gap_and_rejects_too_few_points() -> None:
    config = _fast_config(baseline_window_minutes=5, warmup_minutes=2, minimum_warmup_points=3)
    points = _points([1.0 + (index % 3) * 0.001 for index in range(30)])
    points[15:] = [
        SignalPoint(point.timestamp + timedelta(hours=1), point.score)
        for point in points[15:]
    ]
    samples, gap_count = anomaly_service._calibration_trace(points, config)
    assert gap_count == 1
    assert samples
    assert len({sample.segment for sample in samples}) == 2

    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        run = _seed_testing_run(db, count=8, values=[1.0] * 8)
        with pytest.raises(ValueError, match="at least 9 finite points"):
            anomaly_service.preview_calibration(db, AnomalyDetectionCalibrationRequest(
                testing_run_id=run.id,
                start_timestamp=BASE,
                end_timestamp=BASE + timedelta(minutes=7),
                algorithm="robust_cusum",
                config=config,
            ))


def test_event_threshold_run_persists_reference_and_resolved_threshold() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        reference = _seed_testing_run(
            db,
            count=20,
            name="Normal validation",
            values=[1.0] * 20,
            inference_config={"error_metric": "mse"},
        )
        target = _seed_testing_run(
            db,
            count=20,
            name="Target inference",
            values=[1.0] * 10 + [3.0] * 10,
            inference_config={"error_metric": "mse"},
        )
        detection = anomaly_service.create_run(db, AnomalyDetectionRunCreate(
            name="Quantile event detector",
            testing_run_id=target.id,
            score_series="score",
            start_timestamp=BASE,
            end_timestamp=BASE + timedelta(minutes=19),
            threshold_testing_run_id=reference.id,
            threshold_start_timestamp=BASE,
            threshold_end_timestamp=BASE + timedelta(minutes=19),
            config=_event_config(
                threshold_mode="quantile",
                manual_threshold=None,
                threshold_quantile=0.9999,
                persistence_k=2,
                persistence_n=3,
            ),
        ))
        assert detection.algorithm_version == "event_threshold_v1"
        assert detection.threshold_testing_run_id == reference.id
        assert detection.threshold_testing_run_name == "Normal validation"
        assert detection.resolved_threshold == 1.0
        assert detection.anomaly_count == 1
        assert detection.events[0].max_smoothed_score == 3.0
        assert detection.events[0].threshold == 1.0
        assert detection.series[-1].threshold_off == 0.8


def test_quantile_reference_must_have_matching_model_and_scoring() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        target = _seed_testing_run(db, count=5, inference_config={"error_metric": "mse"})
        wrong_model = _seed_testing_run(
            db,
            count=5,
            name="Wrong model",
            training_run_id=2,
            inference_config={"error_metric": "mse"},
        )
        wrong_scoring = _seed_testing_run(
            db,
            count=5,
            name="Wrong scoring",
            inference_config={"error_metric": "mae"},
        )
        wrong_roi = _seed_testing_run(
            db,
            count=5,
            name="Wrong ROI",
            inference_config={"error_metric": "mse"},
        )
        target.roi_geometry = {"x": 0, "y": 0, "width": 10, "height": 10}
        wrong_roi.roi_geometry = {"x": 1, "y": 0, "width": 10, "height": 10}
        db.commit()
        payload = {
            "name": "Invalid reference",
            "testing_run_id": target.id,
            "start_timestamp": BASE,
            "end_timestamp": BASE + timedelta(minutes=4),
            "threshold_start_timestamp": BASE,
            "threshold_end_timestamp": BASE + timedelta(minutes=4),
            "config": _event_config(threshold_mode="quantile", manual_threshold=None),
        }
        for reference, message in (
            (wrong_model, "same trained model"),
            (wrong_scoring, "same scoring configuration"),
            (wrong_roi, "same ROI geometry"),
        ):
            with pytest.raises(ValueError, match=message):
                anomaly_service.create_run(db, AnomalyDetectionRunCreate(
                    **payload,
                    threshold_testing_run_id=reference.id,
                ))


def test_anomaly_detection_api_crud_and_decimation(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        run = _seed_testing_run(db)
        testing_run_id = run.id

    def override_get_db():
        with SessionLocal() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            progress_token = "test-create-progress"
            created = client.post("/api/anomaly-detection-runs", json={
                "name": "Sensitive detector",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _fast_config().model_dump(),
                "progress_token": progress_token,
            })
            assert created.status_code == 200, created.text
            body = created.json()
            assert body["point_count"] == 80
            assert body["anomaly_count"] == 1
            assert body["algorithm_version"] == "robust_cusum_v3"
            calibration = client.post("/api/anomaly-detection-calibration-preview", json={
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "algorithm": "robust_cusum",
                "profile": "balanced",
                "config": _fast_config().model_dump(),
            })
            assert calibration.status_code == 200, calibration.text
            calibration_body = calibration.json()
            assert calibration_body["profile"] == "balanced"
            assert calibration_body["metrics"]["point_count"] == 80
            assert calibration_body["recommendation"]["cusum_threshold"] is not None
            loaded_end_timestamps: list[datetime] = []
            original_load_points = anomaly_service._load_points

            def tracked_load_points(
                db: Session,
                testing_run_id: int,
                score_series: str,
                start_timestamp: datetime,
                end_timestamp: datetime,
                preroll_minutes: float,
            ):
                loaded_end_timestamps.append(end_timestamp)
                return original_load_points(
                    db,
                    testing_run_id,
                    score_series,
                    start_timestamp,
                    end_timestamp,
                    preroll_minutes,
                )

            monkeypatch.setattr(anomaly_service, "_load_points", tracked_load_points)
            diagnostics = client.get(
                f"/api/anomaly-detection-runs/{body['id']}/diagnostics",
                params={"anchor": (BASE + timedelta(minutes=35)).isoformat(), "count": 3},
            )
            assert diagnostics.status_code == 200, diagnostics.text
            diagnostic_points = diagnostics.json()
            assert len(diagnostic_points) == 3
            assert diagnostic_points[0]["timestamp"].startswith((BASE + timedelta(minutes=35)).isoformat())
            assert diagnostic_points[0]["mad"] is not None
            assert diagnostic_points[0]["scale"] is not None
            assert diagnostic_points[0]["cusum_increment"] is not None
            assert loaded_end_timestamps == [BASE + timedelta(minutes=37)]
            assert body["config"]["algorithm"] == "robust_cusum"
            run_id = body["id"]
            progress = client.get(f"/api/anomaly-detection-progress/{progress_token}")
            assert progress.status_code == 200
            assert progress.json()["status"] == "complete"
            assert progress.json()["percent"] == 100

            listed = client.get("/api/anomaly-detection-runs")
            assert listed.status_code == 200
            assert [item["id"] for item in listed.json()] == [run_id]

            detail_token = "test-detail-progress"
            detail = client.get(
                f"/api/anomaly-detection-runs/{run_id}?max_points=20&progress_token={detail_token}"
            )
            assert detail.status_code == 200
            assert detail.json()["decimated"] is True
            assert len(detail.json()["series"]) <= 20
            assert detail.json()["events"][0]["confirmed_at"] is not None
            detail_progress = client.get(f"/api/anomaly-detection-progress/{detail_token}")
            assert detail_progress.status_code == 200
            assert detail_progress.json()["status"] == "complete"

            threshold_preview = client.post("/api/anomaly-detection-threshold-preview", json={
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "smoothing_enabled": False,
                "smoothing_method": "median",
                "smoothing_window_seconds": 5,
                "quantile": 0.5,
            })
            assert threshold_preview.status_code == 200, threshold_preview.text
            assert threshold_preview.json()["point_count"] == 80
            assert threshold_preview.json()["threshold"] == 1.0

            event_run = client.post("/api/anomaly-detection-runs", json={
                "name": "K-out-of-N detector",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _event_config(
                    persistence_k=2,
                    persistence_n=3,
                    manual_threshold=2.0,
                ).model_dump(),
            })
            assert event_run.status_code == 200, event_run.text
            event_body = event_run.json()
            assert event_body["algorithm_version"] == "event_threshold_v1"
            assert event_body["resolved_threshold"] == 2.0
            assert event_body["anomaly_count"] == 1
            assert event_body["series"][0]["threshold_on"] == 2.0
            assert event_body["events"][0]["max_smoothed_score"] == 3.0
            assert client.delete(f"/api/anomaly-detection-runs/{event_body['id']}").status_code == 204

            sigma_run = client.post("/api/anomaly-detection-runs", json={
                "name": "Rolling 3-sigma detector",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _sigma_config().model_dump(),
            })
            assert sigma_run.status_code == 200, sigma_run.text
            sigma_body = sigma_run.json()
            assert sigma_body["algorithm_version"] == "rolling_sigma_v2"
            assert sigma_body["anomaly_count"] == 1
            assert sigma_body["series"][35]["smoothed"] == sigma_body["series"][35]["score"]
            assert sigma_body["series"][35]["baseline"] == 1.0
            assert sigma_body["series"][35]["baseline_std"] == pytest.approx(1e-6)
            assert client.delete(f"/api/anomaly-detection-runs/{sigma_body['id']}").status_code == 204

            zscore = client.post("/api/anomaly-detection-runs", json={
                "name": "Robust z-score detector",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _fast_config(algorithm="robust_zscore").model_dump(),
            })
            assert zscore.status_code == 200, zscore.text
            assert zscore.json()["algorithm_version"] == "robust_zscore_v3"
            assert zscore.json()["config"]["algorithm"] == "robust_zscore"
            assert all(point["cusum"] == 0 for point in zscore.json()["series"])
            assert client.delete(f"/api/anomaly-detection-runs/{zscore.json()['id']}").status_code == 204

            with_preroll = client.post("/api/anomaly-detection-runs", json={
                "name": "Selected range with pre-roll",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": (BASE + timedelta(minutes=30)).isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _fast_config(preroll_minutes=30).model_dump(),
            })
            assert with_preroll.status_code == 200, with_preroll.text
            preroll_body = with_preroll.json()
            assert preroll_body["series"][0]["timestamp"].startswith("2026-01-01T00:30")
            assert preroll_body["series"][0]["state"] == "normal"
            assert all(not point["timestamp"].startswith("2026-01-01T00:29") for point in preroll_body["series"])
            assert client.delete(f"/api/anomaly-detection-runs/{preroll_body['id']}").status_code == 204

            deleted = client.delete(f"/api/anomaly-detection-runs/{run_id}")
            assert deleted.status_code == 204
            assert client.get(f"/api/anomaly-detection-runs/{run_id}").status_code == 404
            assert client.get("/api/anomaly-detection-progress/unknown-token").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_create_run_executes_detector_once(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    calls = 0
    original_detect = anomaly_service.detect

    def counted_detect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_detect(*args, **kwargs)

    monkeypatch.setattr(anomaly_service, "detect", counted_detect)
    with SessionLocal() as db:
        testing_run = _seed_testing_run(db)
        anomaly_service.create_run(db, AnomalyDetectionRunCreate(
            name="Single pass detector",
            testing_run_id=testing_run.id,
            start_timestamp=BASE,
            end_timestamp=BASE + timedelta(minutes=79),
            config=_fast_config(),
        ))
    assert calls == 1


def test_progress_registry_is_project_isolated() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as first, SessionLocal() as second:
        first.info["database_url"] = "sqlite:///project-one.db"
        second.info["database_url"] = "sqlite:///project-two.db"
        anomaly_service._set_progress(first, "same-token", "loading", 1, 2, "Project one")
        assert anomaly_service.get_progress(first, "same-token") is not None
        assert anomaly_service.get_progress(second, "same-token") is None


def test_restarting_inference_invalidates_saved_detections(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        testing_run = _seed_testing_run(db)
        detection = anomaly_service.create_run(db, AnomalyDetectionRunCreate(
            name="Derived detector",
            testing_run_id=testing_run.id,
            start_timestamp=BASE,
            end_timestamp=BASE + timedelta(minutes=79),
            config=_fast_config(),
        ))
        assert db.get(models.AnomalyDetectionRun, detection.id) is not None
        monkeypatch.setattr(testing_service.scheduler, "wake", lambda: None)
        restarted = testing_service.restart_testing_run(db, testing_run.id)
        assert restarted is not None
        assert restarted.status == "queued"
        assert db.get(models.AnomalyDetectionRun, detection.id) is None


def test_restarting_threshold_reference_invalidates_saved_detection(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        reference = _seed_testing_run(db, count=10, name="Normal reference", values=[1.0] * 10)
        target = _seed_testing_run(db, count=10, name="Target", values=[1.0] * 5 + [3.0] * 5)
        detection = anomaly_service.create_run(db, AnomalyDetectionRunCreate(
            name="Referenced detector",
            testing_run_id=target.id,
            start_timestamp=BASE,
            end_timestamp=BASE + timedelta(minutes=9),
            threshold_testing_run_id=reference.id,
            threshold_start_timestamp=BASE,
            threshold_end_timestamp=BASE + timedelta(minutes=9),
            config=_event_config(threshold_mode="quantile", manual_threshold=None),
        ))
        assert db.get(models.AnomalyDetectionRun, detection.id) is not None
        monkeypatch.setattr(testing_service.scheduler, "wake", lambda: None)
        restarted = testing_service.restart_testing_run(db, reference.id)
        assert restarted is not None
        assert db.get(models.AnomalyDetectionRun, detection.id) is None
