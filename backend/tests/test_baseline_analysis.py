from datetime import datetime, timedelta

import pytest

from app import models
from app.analysis.baseline import calculate, compute_analytics
from app.schemas import BaselineNormalizationRequest
from tests.test_testing_service import make_db


def _seed(values: list[float], *, duplicate_at: int | None = None) -> tuple[object, int, datetime]:
    db = make_db()
    run = models.TestingRun(
        name="baseline run",
        training_run_id=1,
        training_dataset_id=1,
        status="finished",
        training_run_name="model",
        training_pipeline_name="model",
        training_dataset_name="data",
        preprocessing_pipeline_name="prep",
        method_type="autoencoder",
        method_family="autoencoder",
        training_mode="gradient",
        artifact_kind="weights",
        artifact_path="/tmp/model.pt",
    )
    db.add(run)
    db.flush()
    start = datetime(2026, 1, 1)
    for index, value in enumerate(values):
        timestamp_index = index - 1 if duplicate_at == index else index
        db.add(models.TestingRunResult(
            testing_run_id=run.id,
            position=index,
            image_path=f"/{index}.tif",
            timestamp=start + timedelta(seconds=timestamp_index),
            score=value,
            full_mse=value,
            width=8,
            height=6,
        ))
    db.commit()
    return db, run.id, start


def _payload(run_id: int, start: datetime, **overrides) -> BaselineNormalizationRequest:
    body = {
        "traces": [{"testing_run_id": run_id, "label": "model", "color": "#123456", "start": start, "end": start + timedelta(seconds=9)}],
        "score_series": "score",
        "analytics_pipeline": [],
        "stage_index": -1,
        "sampling": 1,
        "baseline_regions": [
            {"id": "b1", "name": "Baseline 1", "start": start, "end": start + timedelta(seconds=2)},
            {"id": "b2", "name": "Baseline 2", "start": start + timedelta(seconds=1), "end": start + timedelta(seconds=3)},
        ],
        "analysis_regions": [{"id": "r1", "name": "Disturbance", "start": start + timedelta(seconds=4), "end": start + timedelta(seconds=9)}],
        "normalization": "classic",
        "thresholds": [3, 5],
        "max_points": 100,
    }
    body.update(overrides)
    return BaselineNormalizationRequest.model_validate(body)


def test_overlapping_baselines_count_each_sample_once_and_compute_classic_stats() -> None:
    db, run_id, start = _seed([0, 1, 2, 3, 10, 10, 10, 0, 0, 0])
    try:
        result = calculate(db, _payload(run_id, start))
        trace = result.traces[0]
        assert trace.baseline.sample_count == 4
        assert trace.baseline.mean == pytest.approx(1.5)
        assert trace.baseline.std == pytest.approx((1.25) ** 0.5)
        region = trace.regions[0]
        assert region.sample_count == 6
        assert region.raw_max == 10
        assert region.thresholds[0].sample_count == 3
        assert region.thresholds[0].longest_seconds == 2
    finally:
        db.close()


def test_robust_constant_baseline_uses_stable_minimum_scale() -> None:
    db, run_id, start = _seed([2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
    try:
        result = calculate(db, _payload(run_id, start, normalization="robust"))
        assert result.traces[0].baseline.scale > 0
        assert all(point.z is None or point.z == pytest.approx(point.z) for point in result.traces[0].series)
    finally:
        db.close()


def test_sampling_precedes_statistics_and_plot_decimation_does_not_change_counts() -> None:
    db, run_id, start = _seed([float(index) for index in range(10)])
    try:
        payload = _payload(run_id, start, sampling=2, max_points=100)
        result = calculate(db, payload)
        assert result.traces[0].baseline.sample_count == 2
        assert result.traces[0].regions[0].sample_count == 3
        assert result.traces[0].total_points == 3
        assert result.traces[0].regions[0].total_points == 3
    finally:
        db.close()


def test_backend_ewma_matches_causal_frontend_semantics() -> None:
    start = datetime(2026, 1, 1)
    values = [0.0, 10.0, 10.0]
    output = compute_analytics("ewma", {"alpha": 0.5}, values, [start + timedelta(seconds=index) for index in range(3)])
    assert output == pytest.approx([0.0, 5.0, 7.5])


def test_analysis_robust_z_uses_configurable_scale_floor() -> None:
    start = datetime(2026, 1, 1)
    values = [1.0] * 5 + [1.0001]
    timestamps = [start + timedelta(seconds=index) for index in range(len(values))]

    output = compute_analytics(
        "robust_z",
        {"alpha": 1.0, "windowSamples": 6, "minimumScaleRelative": 1e-3},
        values,
        timestamps,
    )

    assert output[-1] == pytest.approx(0.1)


def test_analysis_state_machine_recovers_despite_high_prior_cusum() -> None:
    start = datetime(2026, 1, 1)
    values = [1.0] * 5 + [2.0, 2.0, 1.0]
    timestamps = [start + timedelta(seconds=index) for index in range(len(values))]

    output = compute_analytics(
        "state_machine",
        {
            "alpha": 1.0,
            "windowSamples": 5,
            "minimumScaleAbsolute": 0.1,
            "minimumScaleRelative": 0.0,
            "hHigh": 5,
            "offThreshold": 0.5,
        },
        values,
        timestamps,
    )

    assert 3.0 in output
    assert output[-1] == 0.0


def test_region_signal_keeps_full_trace_history_for_causal_processing() -> None:
    db, run_id, start = _seed([0, 0, 0, 8, 8, 0, 0, 0, 0, 0])
    try:
        payload = _payload(
            run_id,
            start,
            moving_average=4,
            baseline_regions=[{"id": "b1", "name": "Baseline", "start": start, "end": start + timedelta(seconds=2)}],
            analysis_regions=[{"id": "r1", "name": "Region", "start": start + timedelta(seconds=3), "end": start + timedelta(seconds=4)}],
        )
        region = calculate(db, payload).traces[0].regions[0]
        assert [point.signal for point in region.series] == pytest.approx([2.0, 4.0])
    finally:
        db.close()


def test_anomaly_event_requires_configured_consecutive_samples() -> None:
    db, run_id, start = _seed([0, 1, 2, 3, 10, 10, 0, 10, 0, 0])
    try:
        result = calculate(db, _payload(run_id, start, thresholds=[3], persistence_samples=2))
        events = result.traces[0].events
        assert events == []
        region = result.traces[0].regions[0]
        assert len(region.events) == 1
        assert region.events[0].threshold == 3
        assert region.events[0].start == start + timedelta(seconds=4)
        assert region.events[0].end == start + timedelta(seconds=5)
        assert region.events[0].sample_count == 2
    finally:
        db.close()


def test_data_gap_resets_consecutive_sample_sequence() -> None:
    db, run_id, start = _seed([0, 1, 2, 3, 10, 10, 10, 10, 0, 0])
    try:
        row = db.query(models.TestingRunResult).filter_by(testing_run_id=run_id, position=6).one()
        row.timestamp = start + timedelta(seconds=60)
        following = db.query(models.TestingRunResult).filter_by(testing_run_id=run_id, position=7).one()
        following.timestamp = start + timedelta(seconds=61)
        db.commit()
        payload = _payload(run_id, start, thresholds=[3], persistence_samples=3)
        payload.traces[0].end = start + timedelta(seconds=70)
        payload.analysis_regions[0].end = start + timedelta(seconds=70)
        result = calculate(db, payload)
        assert result.traces[0].regions[0].events == []
    finally:
        db.close()


def test_region_series_and_events_are_strictly_limited_to_each_analysis_region() -> None:
    db, run_id, start = _seed([0, 1, 2, 3, 10, 10, 0, 10, 10, 0])
    try:
        payload = _payload(
            run_id,
            start,
            thresholds=[3],
            persistence_samples=2,
            analysis_regions=[
                {"id": "r1", "name": "First", "start": start + timedelta(seconds=4), "end": start + timedelta(seconds=5)},
                {"id": "r2", "name": "Second", "start": start + timedelta(seconds=7), "end": start + timedelta(seconds=8)},
            ],
        )
        result = calculate(db, payload)
        trace = result.traces[0]
        assert trace.series == []
        assert trace.events == []
        assert [[point.timestamp for point in region.series] for region in trace.regions] == [
            [start + timedelta(seconds=4), start + timedelta(seconds=5)],
            [start + timedelta(seconds=7), start + timedelta(seconds=8)],
        ]
        assert [[event.start for event in region.events] for region in trace.regions] == [
            [start + timedelta(seconds=4)],
            [start + timedelta(seconds=7)],
        ]
    finally:
        db.close()


def test_persistence_resets_at_region_boundaries_and_overlaps_are_independent() -> None:
    db, run_id, start = _seed([0, 1, 2, 3, 10, 10, 10, 10, 0, 0])
    try:
        payload = _payload(
            run_id,
            start,
            thresholds=[3],
            persistence_samples=3,
            analysis_regions=[
                {"id": "short", "name": "Short", "start": start + timedelta(seconds=4), "end": start + timedelta(seconds=5)},
                {"id": "overlap", "name": "Overlap", "start": start + timedelta(seconds=5), "end": start + timedelta(seconds=7)},
            ],
        )
        regions = calculate(db, payload).traces[0].regions
        assert regions[0].events == []
        assert len(regions[1].events) == 1
        assert regions[1].events[0].start == start + timedelta(seconds=5)
        assert regions[1].events[0].sample_count == 3
    finally:
        db.close()


def test_region_plot_budget_is_shared_without_changing_full_resolution_results() -> None:
    db, run_id, start = _seed([float(index) for index in range(120)])
    try:
        payload = _payload(
            run_id,
            start,
            max_points=100,
            traces=[{"testing_run_id": run_id, "label": "model", "color": "#123456", "start": start, "end": start + timedelta(seconds=119)}],
            analysis_regions=[
                {"id": "r1", "name": "First", "start": start + timedelta(seconds=10), "end": start + timedelta(seconds=69)},
                {"id": "r2", "name": "Second", "start": start + timedelta(seconds=60), "end": start + timedelta(seconds=119)},
            ],
        )
        trace = calculate(db, payload).traces[0]
        assert sum(len(region.series) for region in trace.regions) <= 100
        assert [region.total_points for region in trace.regions] == [60, 60]
        assert [region.sample_count for region in trace.regions] == [60, 60]
        assert all(region.decimated for region in trace.regions)
    finally:
        db.close()
