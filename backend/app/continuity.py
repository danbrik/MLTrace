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
    gap_factor: float = 5.0,
    minimum_gap_seconds: float = 15.0,
) -> list[int]:
    """Assign stable contiguous segment ids without inserting synthetic samples.

    Cadence is estimated independently for every consecutive source group.  A
    group change alone is not a boundary.  A new segment starts only when the
    timestamp delta is strictly larger than both ``minimum_gap_seconds`` and
    ``gap_factor`` times the relevant median positive cadence.
    """
    if source_groups is None:
        source_groups = [None] * len(timestamps)
    if len(timestamps) != len(source_groups):
        raise ValueError("timestamps and source_groups must have the same length")
    if not timestamps:
        return []

    block_cadence: list[float | None] = [None] * len(timestamps)
    all_within_group_deltas: list[float] = []
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
        all_within_group_deltas.extend(positive_deltas)
        cadence = statistics.median(positive_deltas) if positive_deltas else None
        for index in range(block_start, block_end):
            block_cadence[index] = cadence
        block_start = block_end

    global_cadence = statistics.median(all_within_group_deltas) if all_within_group_deltas else None

    segments = [0]
    segment = 0
    for index in range(1, len(timestamps)):
        delta = (timestamps[index] - timestamps[index - 1]).total_seconds()
        if source_groups[index] == source_groups[index - 1]:
            cadence = block_cadence[index] or global_cadence
        else:
            boundary_cadences = [
                value for value in (block_cadence[index - 1], block_cadence[index])
                if value is not None
            ]
            cadence = statistics.median(boundary_cadences) if boundary_cadences else global_cadence
        gap_threshold = max(
            minimum_gap_seconds,
            cadence * gap_factor if cadence is not None else 0.0,
        )
        if delta > gap_threshold:
            segment += 1
        segments.append(segment)
    return segments
