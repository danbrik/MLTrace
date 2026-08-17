from datetime import datetime, timedelta

import numpy as np

from app.continuity import continuity_segments
from app.inspect.diagnostics import _plot_preview


def _times(*seconds: float) -> list[datetime]:
    start = datetime(2026, 1, 1)
    return [start + timedelta(seconds=value) for value in seconds]


def test_fast_cadence_ignores_single_missing_point_and_breaks_only_above_fifteen_seconds() -> None:
    assert continuity_segments(_times(0, 1, 3, 4, 5)) == [0, 0, 0, 0, 0]
    assert continuity_segments(_times(0, 1, 16, 17, 18)) == [0, 0, 0, 0, 0]
    assert continuity_segments(_times(0, 1, 17, 18, 19)) == [0, 0, 1, 1, 1]


def test_slow_cadence_breaks_only_above_five_times_the_typical_step() -> None:
    assert continuity_segments(_times(0, 10, 60, 70, 80)) == [0, 0, 0, 0, 0]
    assert continuity_segments(_times(0, 10, 61, 71, 81)) == [0, 0, 1, 1, 1]


def test_source_group_change_only_breaks_when_the_time_gap_is_long() -> None:
    assert continuity_segments(
        _times(0, 1, 2, 3),
        ["folder-a", "folder-a", "folder-b", "folder-b"],
    ) == [0, 0, 0, 0]
    assert continuity_segments(
        _times(0, 1, 20, 21),
        ["folder-a", "folder-a", "folder-b", "folder-b"],
    ) == [0, 0, 1, 1]
    assert continuity_segments(
        _times(0, 20),
        ["folder-a", "folder-b"],
    ) == [0, 1]


def test_continuity_rejects_mismatched_inputs() -> None:
    try:
        continuity_segments(_times(0, 1), ["folder-a"])
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("Expected mismatched continuity input to fail")


def test_static_inspect_preview_does_not_draw_across_time_gap() -> None:
    example = np.zeros((100, 100, 3), dtype=np.uint8)
    specs = [{"key": "total", "label": "Total", "color": (255, 0, 0)}]

    def render(seconds: list[int]) -> np.ndarray:
        rows = [{
            "timestamp": _times(second)[0].isoformat(),
            "frame_a": f"/images/{index}.tif",
            "frame_b": f"/images/{index + 1}.tif",
            "energy_total": float(index),
        } for index, second in enumerate(seconds)]
        return _plot_preview(example, rows, specs, "energy_total")

    connected = render([0, 1, 2, 3, 4])
    gapped = render([0, 1, 17, 18, 19])
    # This pixel lies halfway between samples 1 and 2 on their red line.
    assert tuple(connected[430, 311]) == (255, 0, 0)
    assert tuple(gapped[430, 311]) != (255, 0, 0)
