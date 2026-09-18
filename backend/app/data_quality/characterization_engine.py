"""Statistics over canonical grid observations; never bridge missing observations."""
from datetime import datetime, timedelta
import math
import numpy as np

VERSION = 1
STAT_KEYS = ('mean', 'median', 'std', 'iqr', 'min', 'q01', 'q05', 'q25', 'q75', 'q95', 'q99', 'max')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def summary(name, data_type, values):
    row = dict(variable=name, data_type=data_type, valid_n=len(values), **dict.fromkeys(STAT_KEYS))
    if data_type == 'numeric' and len(values):
        with np.errstate(over='ignore', invalid='ignore'):
            q = np.quantile(values, [.01, .05, .25, .5, .75, .95, .99], method='linear')
            row.update(mean=np.mean(values), std=np.std(values, ddof=1) if len(values) > 1 else None,
                       iqr=q[4] - q[2], min=np.min(values), max=np.max(values),
                       **dict(zip(('q01', 'q05', 'q25', 'median', 'q75', 'q95', 'q99'), q)))
    return clean(row)


def histogram(values):
    if not len(values):
        return dict(edges=[], counts=[])
    low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        # Relative padding also works for very large finite constants.
        pad = max(abs(low) * .01, .5)
        return dict(edges=[low - pad, high + pad], counts=[len(values)])
    q25, q75 = np.quantile(values, [.25, .75])
    width = 2 * (q75 - q25) / np.cbrt(len(values))
    count = min(100, max(1, math.ceil((high - low) / width))) if width > 0 else min(100, math.ceil(math.sqrt(len(values))))
    counts, edges = np.histogram(values, bins=count)
    return dict(edges=edges.tolist(), counts=counts.tolist())


def details(indices, values):
    values = np.asarray(values, dtype=float)
    indices = np.asarray(indices, dtype=np.int64)
    delta = np.diff(values)[np.diff(indices) == 1]
    finite_delta = delta[np.isfinite(delta)]
    absolute = np.abs(finite_delta)
    dynamics = dict(pair_count=len(delta), non_finite_pair_count=len(delta) - len(finite_delta),
                    median_abs=None, q95_abs=None, q99_abs=None, max_abs=None, unchanged_percent=None)
    if len(finite_delta):
        median, q95, q99 = np.quantile(absolute, [.5, .95, .99])
        dynamics.update(median_abs=median, q95_abs=q95, q99_abs=q99, max_abs=np.max(absolute),
                        unchanged_percent=float(np.count_nonzero(delta == 0)) / len(delta) * 100)
    box = None
    if len(values):
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        inside = values[(values >= q25 - 1.5 * (q75 - q25)) & (values <= q75 + 1.5 * (q75 - q25))]
        box = dict(q25=q25, median=median, q75=q75, lower=float(np.min(inside)), upper=float(np.max(inside)), outlier_count=len(values) - len(inside))
    return clean(dict(histogram=histogram(values), box=box, dynamics=dynamics,
                      delta_histogram=histogram(finite_delta), absolute_delta_histogram=histogram(absolute)))


def temporal(metadata, indices, values, start=None, end=None):
    origin = datetime.fromisoformat(metadata['grid_start'])
    seconds = metadata['interval_seconds']
    count = metadata['grid_count']
    lo = max(0, math.ceil((start - origin).total_seconds() / seconds)) if start else 0
    hi = min(count, math.floor((end - origin).total_seconds() / seconds) + 1) if end else count
    indices, values = np.asarray(indices), np.asarray(values)
    width = 1 if hi - lo <= 2000 else math.ceil((hi - lo) / 1000)
    points = []
    previous_last = None
    previous_continuous = False
    for a in range(lo, max(lo, hi), width):
        b = min(hi, a + width)
        left, right = np.searchsorted(indices, [a, b])
        ix, vals = indices[left:right], values[left:right]
        continuous = bool(len(ix) and np.all(np.diff(ix) == 1))
        connect = bool(continuous and previous_continuous and previous_last is not None and ix[0] == previous_last + 1)
        q = np.quantile(vals, [.25, .5, .75]) if len(vals) else [None] * 3
        timestamp = lambda index: (origin + timedelta(seconds=int(index) * seconds)).isoformat()
        points.append(dict(time=timestamp(a), end=timestamp(b - 1), valid_n=len(vals), q25=q[0], value=q[1], q75=q[2],
                           connect_previous=connect, has_internal_gap=bool(len(ix) and not continuous)))
        previous_last = int(ix[-1]) if len(ix) else None
        previous_continuous = continuous
    return clean(dict(aggregated=width > 1, interval_seconds=width * seconds, points=points))
