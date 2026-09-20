"""One train-only scaling and window contract shared by every architecture."""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd

from app.time_series.service import boundary, iso, parse_times, read_csv

SUBSETS = ("train", "validation", "test")
GAP_FACTOR = 1.5


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass
class Prepared:
    values: np.ndarray
    scaled: np.ndarray
    timestamps: np.ndarray
    groups: np.ndarray
    interval_indices: np.ndarray
    segment_ids: np.ndarray
    endpoints: np.ndarray
    columns: list[str]
    intervals: list[dict]
    window_length: int
    summary: dict

    def windows(self, endpoints):
        return self.scaled[np.asarray(endpoints)[:, None] - np.arange(self.window_length - 1, -1, -1)]


def prepare(content: bytes, dataset: dict, intervals: list[dict], window_length: int | None) -> Prepared:
    frame = read_csv(content)
    columns = [c for c in dataset["selected_columns"] if c != dataset["timestamp_column"]]
    if not columns:
        raise ValueError("Mindestens eine Sensorspalte muss ausgewählt sein.")
    times = parse_times(frame, dataset["timestamp_column"], dataset["timestamp_format"])
    order = np.argsort(times.asi8, kind="stable")
    times = times.take(order)
    ns = times.asi8
    if np.any(np.diff(ns) == 0):
        raise ValueError("Doppelte Zeitpunkte: pro Zeitpunkt muss genau eine Sensorzeile vorhanden sein.")
    frame = frame.iloc[order]
    groups = np.full(len(frame), -1, dtype=np.int8)
    interval_indices = np.full(len(frame), -1, dtype=np.int32)
    for i, entry in enumerate(intervals):
        selected = (ns >= boundary(entry["start"]).value) & (ns <= boundary(entry["end"]).value)
        if (groups[selected] != -1).any():
            raise ValueError("Der Split enthält überlappende Zuordnungen.")
        groups[selected] = SUBSETS.index(entry["subset"])
        interval_indices[selected] = i
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    bad = np.argwhere(~np.isfinite(numeric))
    if len(bad):
        row, column = bad[0]
        raise ValueError(f"Sensor {columns[column]} bei {iso(times[row])}: kein endlicher numerischer Wert.")
    # Unassigned cells never enter training or inference. Keep their positions to
    # prevent joining assigned windows across excluded source rows.
    train = groups == 0
    if not train.any():
        raise ValueError("Der Split enthält keine Trainingszeilen.")
    delta = np.diff(ns)
    train_pairs = train[1:] & train[:-1]
    if not train_pairs.any():
        raise ValueError("Für das Samplingintervall sind mindestens zwei aufeinanderfolgende Trainingszeilen erforderlich.")
    cadence_ns = float(np.median(delta[train_pairs]))
    cadence_s = cadence_ns / 1e9
    suggested = max(1, math.floor(10800 / cadence_s + 0.5))
    length = suggested if window_length is None else window_length
    if type(length) is not int or not 1 <= length <= 100000:
        raise ValueError("Fensterlänge muss eine ganze Zahl zwischen 1 und 100000 sein.")
    minimum = numeric[train].min(axis=0)
    maximum = numeric[train].max(axis=0)
    constant = maximum == minimum
    with np.errstate(over="ignore"):
        denominator = np.where(constant, 1.0, maximum - minimum)
    if not np.isfinite(denominator).all():
        raise ValueError("Train-Min/Max-Spanne überschreitet den numerischen Wertebereich. Bitte Sensoreinheiten prüfen.")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = ((numeric - minimum) / denominator).astype(np.float32)
    if not np.isfinite(scaled[groups >= 0]).all():
        raise ValueError("Skalierte Werte überschreiten den numerischen Wertebereich. Bitte Sensoreinheiten prüfen.")
    scaled[groups < 0] = 0
    segment_ids = np.full(len(frame), -1, dtype=np.int64)
    endpoints = []
    segment = -1
    segment_start = 0
    gaps = {name: 0 for name in SUBSETS}
    segments = {name: 0 for name in SUBSETS}
    for i, group in enumerate(groups):
        if group < 0:
            continue
        same_group = i > 0 and group == groups[i - 1]
        gap = same_group and delta[i - 1] > GAP_FACTOR * cadence_ns
        if not same_group or gap:
            segment += 1
            segment_start = i
            segments[SUBSETS[group]] += 1
            gaps[SUBSETS[group]] += int(gap)
        segment_ids[i] = segment
        if i - segment_start + 1 >= length:
            endpoints.append(i)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    counts = {}
    errors = []
    for group, subset in enumerate(SUBSETS):
        rows = int((groups == group).sum())
        windows = int((groups[endpoints] == group).sum())
        counts[subset] = dict(rows=rows, windows=windows, warmup_rows=rows - windows, segments=segments[subset], gaps=gaps[subset])
        if (subset in ("train", "test") or any(entry["subset"] == subset for entry in intervals)) and not windows:
            errors.append(f"{subset}: kein vollständiges Fenster mit L={length} ({rows} zugeordnete Zeilen).")
    scaler = dict(method="minmax", fit_subset="train", clip=False, minimum=minimum.tolist(), maximum=maximum.tolist(),
                  denominator=denominator.tolist(), constant_columns=[c for c, flag in zip(columns, constant) if flag], columns=columns)
    spans = (ns[endpoints] - ns[endpoints - length + 1]) / 1e9
    summary = dict(timestamp_span_min_seconds=float(spans.min()) if len(spans) else None,
                   timestamp_span_max_seconds=float(spans.max()) if len(spans) else None,
                   columns=columns, sensor_count=len(columns), sampling_interval_seconds=cadence_s,
                   suggested_window_length=suggested, window_length=length, step=1, gap_factor=GAP_FACTOR,
                   nominal_history_seconds=length * cadence_s, timestamp_span_seconds=float(np.median(spans)) if len(spans) else (length - 1) * cadence_s,
                   detected_gaps=sum(gaps.values()), counts=counts, scaler=scaler, errors=errors)
    summary["input_fingerprint"] = fingerprint(dict(source=hashlib.sha256(content).hexdigest(),
        columns=columns, intervals=intervals, window_length=length, step=1, scaler=scaler, gap_factor=GAP_FACTOR))
    return Prepared(numeric, scaled, ns, groups, interval_indices, segment_ids, endpoints, columns, intervals, length, summary)


def timestamp(ns):
    return iso(pd.Timestamp(int(ns), unit="ns", tz="UTC"))
