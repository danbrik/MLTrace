from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.evaluation.metrics import (
    OPERATING_QUANTILES,
    EvaluationEvent,
    EvaluationMetricError,
    ScorePoint,
    TimeRange,
    calculate_detection,
    calculate_drift,
    calculate_separation,
    empirical_wasserstein_1,
)


BASE = datetime(2026, 1, 1)


def _at(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


def _range(start: float, end: float) -> TimeRange:
    return TimeRange(_at(start), _at(end))


def _event(event_id: str, start: float, end: float) -> EvaluationEvent:
    return EvaluationEvent(event_id, _at(start), _at(end), f"Event {event_id}", "target")


def _points(
    values: list[float],
    *,
    start: float = 0,
    step: float = 1,
    segments: list[int] | None = None,
) -> list[ScorePoint]:
    if segments is None:
        segments = [0] * len(values)
    return [
        ScorePoint(_at(start + index * step), index, value, segments[index])
        for index, value in enumerate(values)
    ]


def _error_code(error: pytest.ExceptionInfo[EvaluationMetricError]) -> str:
    return error.value.code


def test_separation_uses_local_median_mad_and_aggregates_events() -> None:
    values = [
        9, 9,
        0, 2, 0, 2,
        4, 6,
        9, 9, 9, 9,
        1, 3, 1, 3,
        5, 5,
        9, 9,
    ]
    result = calculate_separation(
        list(reversed(_points(values))),
        [_event("e2", 16, 17), _event("e1", 6, 7)],
        evaluation_range=_range(0, 20),
        normal_window_duration_seconds=4,
    )

    scale = 1.4826 + 1e-12
    expected = [(5 - 1) / scale, (5 - 2) / scale]
    assert [row.event_id for row in result.events] == ["e1", "e2"]
    assert [row.normal_mad for row in result.events] == [1.0, 1.0]
    assert [row.separation for row in result.events] == pytest.approx(expected)
    assert result.sep_min == pytest.approx(min(expected))
    assert result.sep_median == pytest.approx(np.median(expected))
    assert result.events[0].separation_p95 == pytest.approx((5.9 - 1) / scale)
    assert result.warnings == ()


def test_separation_override_and_zero_mad_are_explicit_diagnostics() -> None:
    points = _points([1, 1, 1, 1, 2, 2, 9, 9])
    result = calculate_separation(
        points,
        [_event("e1", 4, 5)],
        evaluation_range=_range(0, 8),
        normal_window_duration_seconds=2,
        normal_window_overrides={"e1": {"start_timestamp": _at(0), "end_timestamp": _at(4)}},
    )

    assert result.events[0].normal_start == _at(0)
    assert result.events[0].normal_mad == 0
    assert result.events[0].separation == pytest.approx(1e12)
    assert [warning.code for warning in result.warnings] == ["near_zero_normal_mad"]
    payload = result.to_dict()
    assert payload["events"][0]["event_start"] == _at(4).isoformat()
    assert payload["warnings"][0]["context"]["event_id"] == "e1"


def test_separation_fails_whole_stage_for_gap_or_missing_event_scores() -> None:
    with pytest.raises(EvaluationMetricError) as caught:
        calculate_separation(
            [
                ScorePoint(_at(1), 0, 0, 0),
                ScorePoint(_at(2), 1, 0, 1),
                ScorePoint(_at(3), 2, 0, 1),
                ScorePoint(_at(4), 3, 2, 1),
            ],
            [_event("e1", 4, 5), _event("no-points", 6, 7)],
            evaluation_range=_range(0, 8),
            normal_window_duration_seconds=3,
        )

    assert _error_code(caught) == "invalid_separation_events"
    details = {item["event_id"]: item["reasons"] for item in caught.value.details["events"]}
    assert "continuity_gap" in details["e1"]
    assert "no_event_scores" in details["no-points"]


def test_exact_empirical_wasserstein_supports_unequal_samples_and_duplicates() -> None:
    assert empirical_wasserstein_1([0, 2], [1, 3]) == pytest.approx(1)
    assert empirical_wasserstein_1([0], [0, 2]) == pytest.approx(1)
    assert empirical_wasserstein_1([0, 0, 2], [1, 1]) == pytest.approx(1)


def test_drift_uses_anchored_complete_windows_and_excludes_target_windows() -> None:
    result = calculate_drift(
        [
            ScorePoint(_at(1), 0, 0),
            ScorePoint(_at(2), 1, 2),
            ScorePoint(_at(11), 2, 1),
            ScorePoint(_at(12), 3, 3),
            ScorePoint(_at(21), 4, 100),
        ],
        [ScorePoint(_at(100), 0, 0), ScorePoint(_at(102), 1, 2)],
        evaluation_range=_range(0, 30),
        reference_range=_range(100, 102),
        window_duration_seconds=10,
        events=[_event("e1", 20, 25)],
    )

    assert result.reference_iqr == pytest.approx(1)
    assert result.reference_point_count == 2  # Outer reference end is inclusive.
    assert result.d_mean == pytest.approx(0.5)
    assert result.d_max == pytest.approx(1)
    assert result.valid_window_count == 2
    assert result.discarded_window_count == 1
    assert [window.status for window in result.windows] == ["valid", "valid", "discarded"]
    assert result.windows[-1].exclusion_reason == "target_event"


def test_drift_bin_starting_at_inclusive_event_end_is_discarded() -> None:
    result = calculate_drift(
        [
            ScorePoint(_at(1), 0, 0),
            ScorePoint(_at(11), 1, 0),
            ScorePoint(_at(21), 2, 0),
        ],
        [ScorePoint(_at(100), 0, 0), ScorePoint(_at(101), 1, 1)],
        evaluation_range=_range(0, 30),
        reference_range=_range(100, 102),
        window_duration_seconds=10,
        events=[_event("ends-on-boundary", 5, 10)],
    )

    assert result.windows[0].exclusion_reason == "target_event"
    # The label contains t=10, so the half-open [10,20) bin also intersects it.
    assert result.windows[1].exclusion_reason == "target_event"
    assert result.windows[2].status == "valid"


def test_drift_records_incomplete_remainder_and_continuity_gaps() -> None:
    result = calculate_drift(
        [
            ScorePoint(_at(1), 0, 0, 0),
            ScorePoint(_at(9), 1, 0, 1),
            ScorePoint(_at(11), 2, 0, 1),
            ScorePoint(_at(12), 3, 2, 1),
            ScorePoint(_at(21), 4, 1, 1),
        ],
        [ScorePoint(_at(100), 0, 0), ScorePoint(_at(101), 1, 2)],
        evaluation_range=_range(0, 25),
        reference_range=_range(100, 102),
        window_duration_seconds=10,
    )

    assert result.valid_window_count == 1
    assert result.discarded_window_count == 2
    assert [window.exclusion_reason for window in result.windows] == [
        "continuity_gap",
        None,
        "incomplete_window",
    ]
    assert result.windows[-1].end == _at(25)


def test_drift_near_zero_iqr_warns_but_remains_finite() -> None:
    result = calculate_drift(
        [ScorePoint(_at(1), 0, 1), ScorePoint(_at(2), 1, 2)],
        [ScorePoint(_at(100), 0, 1), ScorePoint(_at(101), 1, 1)],
        evaluation_range=_range(0, 10),
        reference_range=_range(100, 102),
        window_duration_seconds=10,
    )

    assert result.reference_iqr == 0
    assert math_is_finite(result.d_mean)
    assert result.d_mean == pytest.approx(5e11)
    assert [warning.code for warning in result.warnings] == ["near_zero_reference_iqr"]


def test_drift_fails_if_every_window_is_invalid() -> None:
    with pytest.raises(EvaluationMetricError) as caught:
        calculate_drift(
            _points([0, 1], start=1),
            _points([0, 1], start=100),
            evaluation_range=_range(0, 5),
            reference_range=_range(100, 102),
            window_duration_seconds=10,
        )
    assert _error_code(caught) == "no_valid_drift_windows"
    assert caught.value.details["windows"][0]["exclusion_reason"] == "incomplete_window"


def test_detection_calculates_recall_delay_fpr_and_far_for_every_quantile() -> None:
    result = calculate_detection(
        _points([1, 1, 1, 0, 0, 1, 0, 0, 1, 1]),
        _points([0] * 100, start=100),
        [_event("e1", 3, 4), _event("e2", 7, 8)],
        evaluation_range=_range(0, 10),
        calibration_range=_range(100, 200),
        standard_duration_seconds=10,
        anticipation_seconds=1,
    )

    assert [row.quantile for row in result.operating_points] == list(OPERATING_QUANTILES)
    for row in result.operating_points:
        assert row.threshold == 0
        assert row.event_recall == 1
        assert row.median_delay_seconds == 0
        assert [event.delay_seconds for event in row.events] == [-1, 1]
        assert row.normal_frame_count == 4
        assert row.false_positive_frame_count == 4
        assert row.frame_fpr == 1
        assert row.false_alarm_event_count == 3
        assert [(alarm.start, alarm.end, alarm.point_count) for alarm in row.false_alarms] == [
            (_at(0), _at(1), 2),
            (_at(5), _at(5), 1),
            (_at(9), _at(9), 1),
        ]
        assert row.normal_observation_seconds == 1
        assert row.far_t0 == 30


def test_detection_uses_strict_threshold_and_keeps_no_hit_delay_null() -> None:
    result = calculate_detection(
        _points([1, 1, 1, 1, 1, 1]),
        _points([1] * 20, start=100),
        [_event("e1", 2, 3)],
        evaluation_range=_range(0, 6),
        calibration_range=_range(100, 120),
        standard_duration_seconds=10,
    )

    row = result.operating_points[2]
    assert row.threshold == 1
    assert row.event_recall == 0
    assert row.detected_event_count == 0
    assert row.missed_event_count == 1
    assert row.median_delay_seconds is None
    assert row.events[0].first_detection is None
    assert row.frame_fpr == 0
    assert row.far_t0 == 0


def test_event_end_point_is_included_in_separation_and_detection() -> None:
    points = _points([0, 0, 0, 10, 0, 0])
    event = _event("e1", 2, 3)

    separation = calculate_separation(
        points,
        [event],
        evaluation_range=_range(0, 5),
        normal_window_duration_seconds=2,
    )
    assert separation.events[0].event_point_count == 2
    assert separation.events[0].separation > 0

    detection = calculate_detection(
        points,
        _points([0] * 20, start=100),
        [event],
        evaluation_range=_range(0, 5),
        calibration_range=_range(100, 120),
        standard_duration_seconds=1,
    )
    row = detection.operating_points[0]
    assert row.events[0].first_detection == _at(3)
    assert row.events[0].delay_seconds == 1
    # t=3 is the inclusive event endpoint and therefore not a normal frame.
    assert row.normal_frame_count == 4
    assert row.false_positive_frame_count == 0


def test_detection_counts_duplicate_timestamps_as_distinct_frames() -> None:
    points = [
        ScorePoint(_at(0), 0, 1),
        ScorePoint(_at(0), 1, 1),
        ScorePoint(_at(1), 2, 0),
        ScorePoint(_at(2), 4, 1),
        ScorePoint(_at(2), 3, 1),
        ScorePoint(_at(3), 5, 0),
        ScorePoint(_at(4), 6, 0),
    ]
    result = calculate_detection(
        list(reversed(points)),
        _points([0] * 20, start=100),
        [_event("e1", 2, 2.5)],
        evaluation_range=_range(0, 5),
        calibration_range=_range(100, 120),
        standard_duration_seconds=1,
    )

    row = result.operating_points[0]
    assert row.detected_event_count == 1
    assert row.events[0].first_detection == _at(2)
    assert row.normal_frame_count == 5
    assert row.false_positive_frame_count == 2
    assert row.frame_fpr == pytest.approx(0.4)
    assert row.false_alarm_event_count == 1


def test_detection_ends_false_alarm_at_continuity_gap() -> None:
    points = [
        ScorePoint(_at(0), 0, 1, 0),
        ScorePoint(_at(1), 1, 1, 0),
        ScorePoint(_at(10), 2, 1, 1),
        ScorePoint(_at(11), 3, 1, 1),
        ScorePoint(_at(20), 4, 0, 1),
        ScorePoint(_at(21), 5, 0, 1),
        ScorePoint(_at(22), 6, 0, 1),
    ]
    result = calculate_detection(
        points,
        _points([0] * 20, start=100),
        [_event("e1", 20, 20.5)],
        evaluation_range=_range(0, 23),
        calibration_range=_range(100, 120),
        standard_duration_seconds=1,
    )

    row = result.operating_points[0]
    assert row.false_alarm_event_count == 2
    assert row.normal_observation_seconds == 3
    assert row.far_t0 == pytest.approx(2 / 3)


def test_detection_rejects_zero_normal_duration_and_overlapping_anticipation() -> None:
    with pytest.raises(EvaluationMetricError) as no_duration:
        calculate_detection(
            _points([0, 1, 1, 0]),
            _points([0] * 20, start=100),
            [_event("e1", 1, 2)],
            evaluation_range=_range(0, 4),
            calibration_range=_range(100, 120),
            standard_duration_seconds=10,
        )
    assert _error_code(no_duration) == "no_normal_observation_duration"

    with pytest.raises(EvaluationMetricError) as overlap:
        calculate_detection(
            _points([0] * 12),
            _points([0] * 20, start=100),
            [_event("e1", 3, 5), _event("e2", 6, 8)],
            evaluation_range=_range(0, 12),
            calibration_range=_range(100, 120),
            standard_duration_seconds=10,
            anticipation_seconds=2,
        )
    assert _error_code(overlap) == "overlapping_anticipation_windows"


def test_overlapping_target_events_are_rejected() -> None:
    with pytest.raises(EvaluationMetricError) as caught:
        calculate_detection(
            _points([0] * 12),
            _points([0] * 20, start=100),
            [_event("e1", 3, 6), _event("e2", 5, 8)],
            evaluation_range=_range(0, 12),
            calibration_range=_range(100, 120),
            standard_duration_seconds=10,
        )
    assert _error_code(caught) == "overlapping_events"


@pytest.mark.parametrize(
    ("points", "expected_code"),
    [
        ([{"timestamp": BASE, "position": 0}], "missing_score_series"),
        ([{"timestamp": BASE, "position": 0, "score": float("nan")}], "non_finite_score"),
        ([{"timestamp": BASE, "position": 0, "score": float("inf")}], "non_finite_score"),
    ],
)
def test_missing_or_non_finite_score_values_are_never_silently_dropped(
    points: list[dict[str, object]],
    expected_code: str,
) -> None:
    with pytest.raises(EvaluationMetricError) as caught:
        calculate_separation(
            points,
            [_event("e1", 2, 3)],
            evaluation_range=_range(0, 4),
            normal_window_duration_seconds=1,
        )
    assert _error_code(caught) == expected_code


def test_small_calibration_tail_produces_diagnostic_per_operating_point() -> None:
    result = calculate_detection(
        _points([0, 0, 1, 0, 0]),
        _points(list(range(10)), start=100),
        [_event("e1", 2, 2.5)],
        evaluation_range=_range(0, 5),
        calibration_range=_range(100, 110),
        standard_duration_seconds=1,
    )

    assert len(result.warnings) == len(OPERATING_QUANTILES)
    assert {warning.code for warning in result.warnings} == {"underpopulated_calibration_tail"}


def math_is_finite(value: float) -> bool:
    return not np.isnan(value) and not np.isinf(value)
