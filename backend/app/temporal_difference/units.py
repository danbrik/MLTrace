"""Display conversion only; persisted measurements remain in pipeline units."""
import numpy as np

DIVISOR = 65535
PERCENT_FACTOR = 100.0 / DIVISOR
NORMALIZATION = {"unit": "percent", "normalization_divisor": DIVISOR}


def to_percent(value):
    if value is None:
        return None
    # Float conversion also makes this safe for unsigned integer image arrays.
    if isinstance(value, np.ndarray):
        return value.astype(np.float64) * PERCENT_FACTOR
    return float(value) * PERCENT_FACTOR


def convert_row(row, unit, fields):
    if unit == "raw":
        return row
    return {**row, **{key: to_percent(row[key]) for key in fields}, **NORMALIZATION}
