from __future__ import annotations

import re

import numpy as np

FRAME_SCORE_AGGREGATION_OPTIONS = ["mean", "p95", "p99", "p99.9", "max"]

_PERCENTILE_PATTERN = re.compile(r"^p(\d+(?:\.\d+)?)$")


def normalize_aggregation(aggregation: str | None) -> str:
    """Return a supported aggregation name, falling back to ``mean``."""
    candidate = str(aggregation or "mean").strip().lower()
    if candidate == "max" or candidate == "mean":
        return candidate
    match = _PERCENTILE_PATTERN.match(candidate)
    if match and 0.0 < float(match.group(1)) < 100.0:
        return candidate
    return "mean"


def aggregate_score(values, aggregation: str | None = "mean") -> float:
    """Reduce a residual/error map to a single frame score."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    kind = normalize_aggregation(aggregation)
    if kind == "max":
        return float(flat.max())
    match = _PERCENTILE_PATTERN.match(kind)
    if match:
        return float(np.percentile(flat, float(match.group(1))))
    return float(flat.mean())
