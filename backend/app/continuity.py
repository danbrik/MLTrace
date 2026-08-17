from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path


def source_group(image_path: str | None) -> str | None:
    """Return the source directory used to keep unrelated image folders apart."""
    if not image_path:
        return None
    return str(Path(image_path).parent)


def continuity_segments(
    timestamps: Sequence[datetime],
    source_groups: Sequence[str | None] | None = None,
    *,
    gap_factor: float = 1.5,
) -> list[int]:
    """Assign stable contiguous segment ids without inserting synthetic samples.

    Cadence is estimated independently for every consecutive source group.  A
    group change is always a boundary; inside a group only a timestamp delta
    strictly larger than ``gap_factor`` times its median positive cadence opens
    a new segment.
    """
    if source_groups is None:
        source_groups = [None] * len(timestamps)
    if len(timestamps) != len(source_groups):
        raise ValueError("timestamps and source_groups must have the same length")
    if not timestamps:
        return []

    block_cadence: list[float | None] = [None] * len(timestamps)
    block_start = 0
    while block_start < len(timestamps):
        block_end = block_start + 1
        while block_end < len(timestamps) and source_groups[block_end] == source_groups[block_start]:
            block_end += 1
        positive_deltas = [
            (timestamps[index] - timestamps[index - 1]).total_seconds()
            for index in range(block_start + 1, block_end)
            if timestamps[index] > timestamps[index - 1]
        ]
        cadence = statistics.median(positive_deltas) if positive_deltas else None
        for index in range(block_start, block_end):
            block_cadence[index] = cadence
        block_start = block_end

    segments = [0]
    segment = 0
    for index in range(1, len(timestamps)):
        group_changed = source_groups[index] != source_groups[index - 1]
        delta = (timestamps[index] - timestamps[index - 1]).total_seconds()
        cadence = block_cadence[index]
        timestamp_gap = cadence is not None and delta > cadence * gap_factor
        if group_changed or timestamp_gap:
            segment += 1
        segments.append(segment)
    return segments
