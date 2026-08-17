from datetime import datetime, timedelta

import numpy as np

from app.continuity import continuity_segments
from app.inspect.diagnostics import _plot_preview


def _times(*seconds: float) -> list[datetime]:
    start = datetime(2026, 1, 1)
    return [start + timedelta(seconds=value) for value in seconds]


def test_continuity_breaks_only_above_one_and_a_half_times_cadence() -> None:
    assert continuity_segments(_times(0, 1, 2, 3.5, 4.5, 5.5)) == [0, 0, 0, 0, 0, 0]
    assert continuity_segments(_times(0, 1, 2, 4, 5, 6)) == [0, 0, 0, 1, 1, 1]


def test_continuity_breaks_when_source_group_changes() -> None:
    assert continuity_segments(
        _times(0, 1, 2, 3),
        ["folder-a", "folder-a", "folder-b", "folder-b"],
    ) == [0, 0, 1, 1]


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
    gapped = render([0, 1, 3, 4, 5])
    # This pixel lies halfway between samples 1 and 2 on their red line.
    assert tuple(connected[430, 311]) == (255, 0, 0)
    assert tuple(gapped[430, 311]) != (255, 0, 0)
