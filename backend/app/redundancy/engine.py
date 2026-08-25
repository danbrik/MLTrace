from __future__ import annotations

import csv
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_MISSING_TOKENS = ["", "na", "n/a", "null", "none", "nan"]
DEFAULT_CONFIG = {
    "high_missing_fraction": 0.30,
    "nearly_constant_fraction": 0.95,
    "min_valid_values": 10,
    "min_pair_values": 10,
    "numeric_candidate_fraction": 0.80,
    "missing_tokens": DEFAULT_MISSING_TOKENS,
    "linkage_method": "average",
}
INCOMPLETE_MATRIX_MESSAGE = "Excluded from clustering because the correlation matrix is incomplete."

_TIME_RE = re.compile(
    r"^(\d{4})([-.])(\d{2})\2(\d{2})(?:([ T])(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?)?$"
)


class AnalysisCancelled(RuntimeError):
    pass


def parse_timestamp(value: str) -> datetime | None:
    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        return None
    year, _sep, month, day, _join, hour, minute, second, fraction = match.groups()
    try:
        return datetime(
            int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0),
            int((fraction or "").ljust(6, "0") or 0),
        )
    except ValueError:
        return None


def parse_number(value: str | None, missing_tokens: set[str]) -> tuple[float, str]:
    if value is None or value.strip().lower() in missing_tokens:
        return math.nan, "missing"
    text = value.strip()
    if "," in text and "." not in text and text.count(",") == 1:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return math.nan, "invalid"
    return (number, "valid") if math.isfinite(number) else (math.nan, "invalid")


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def profile_csv(path: Path, preview_limit: int = 20) -> dict:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError("The CSV is empty.") from exc
        if not headers or any(not item.strip() for item in headers):
            raise ValueError("CSV headers must not be empty.")
        if len(set(headers)) != len(headers):
            raise ValueError("CSV headers must be unique.")
        missing_tokens = set(DEFAULT_MISSING_TOKENS)
        counters = [
            {"non_missing": 0, "numeric": 0, "timestamp": 0, "timestamp_start": None, "timestamp_end": None}
            for _ in headers
        ]
        preview: list[list[str | None]] = []
        rows = 0
        for row in reader:
            rows += 1
            if len(row) != len(headers):
                raise ValueError(f"CSV row {rows + 1} has {len(row)} fields; expected {len(headers)}.")
            if len(preview) < preview_limit:
                preview.append([value if value != "" else None for value in row])
            for index, value in enumerate(row):
                if value.strip().lower() in missing_tokens:
                    continue
                counters[index]["non_missing"] += 1
                number, state = parse_number(value, missing_tokens)
                if state == "valid" and math.isfinite(number):
                    counters[index]["numeric"] += 1
                timestamp = parse_timestamp(value)
                if timestamp is not None:
                    counters[index]["timestamp"] += 1
                    if counters[index]["timestamp_start"] is None or timestamp < counters[index]["timestamp_start"]:
                        counters[index]["timestamp_start"] = timestamp
                    if counters[index]["timestamp_end"] is None or timestamp > counters[index]["timestamp_end"]:
                        counters[index]["timestamp_end"] = timestamp
    profiles = []
    for name, counts in zip(headers, counters, strict=True):
        non_missing = counts["non_missing"]
        profiles.append({
            "name": name,
            **{
                **counts,
                "timestamp_start": counts["timestamp_start"].isoformat() if counts["timestamp_start"] else None,
                "timestamp_end": counts["timestamp_end"].isoformat() if counts["timestamp_end"] else None,
            },
            "numeric_fraction": counts["numeric"] / non_missing if non_missing else 0.0,
            "timestamp_fraction": counts["timestamp"] / non_missing if non_missing else 0.0,
        })
    return {"delimiter": delimiter, "headers": headers, "row_count": rows, "column_profiles": profiles, "preview_rows": preview}


def _rank_correlation(left: np.ndarray, right: np.ndarray, minimum: int) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(left) & np.isfinite(right)
    count = int(mask.sum())
    if count < minimum:
        return None, None, count
    x = left[mask]
    y = right[mask]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None, count
    from scipy.stats import rankdata

    rho = float(np.corrcoef(rankdata(x, method="average"), rankdata(y, method="average"))[0, 1])
    pearson = float(np.corrcoef(x, y)[0, 1])
    return (rho if math.isfinite(rho) else None, pearson if math.isfinite(pearson) else None, count)


def _quality(name: str, values: np.ndarray, invalid: int, total: int, config: dict) -> dict:
    finite = values[np.isfinite(values)]
    valid = int(finite.size)
    unique, counts = np.unique(finite, return_counts=True) if valid else (np.array([]), np.array([]))
    missing_fraction = 1.0 - valid / total if total else 1.0
    statuses: list[str] = []
    if valid < int(config["min_valid_values"]):
        statuses.append("Insufficient data")
    if unique.size <= 1 and valid:
        statuses.append("Constant")
    elif counts.size and float(counts.max() / valid) >= float(config["nearly_constant_fraction"]):
        statuses.append("Nearly constant")
    if missing_fraction >= float(config["high_missing_fraction"]):
        statuses.append("High missingness")
    if invalid:
        statuses.append("Invalid numeric values treated as missing")
    return {
        "variable": name,
        "valid_n": valid,
        "missing_n": total - valid,
        "missing_fraction": missing_fraction,
        "invalid_numeric_n": invalid,
        "mean": float(np.mean(finite)) if valid else None,
        "median": float(np.median(finite)) if valid else None,
        "std": float(np.std(finite, ddof=1)) if valid > 1 else None,
        "min": float(np.min(finite)) if valid else None,
        "max": float(np.max(finite)) if valid else None,
        "unique_n": int(unique.size),
        "statuses": statuses or ["OK"],
    }


def cluster_cut(result: dict, cutoff: float) -> dict:
    names: list[str] = result.get("clusterable_variables", [])
    linkage_rows = result.get("linkage", [])
    if not names:
        return {"cutoff": cutoff, "distance_cutoff": 1.0 - cutoff, "assignments": [], "clusters": []}
    if len(names) == 1:
        labels = np.array([1], dtype=int)
    else:
        from scipy.cluster.hierarchy import fcluster

        labels = fcluster(np.asarray(linkage_rows, dtype=float), t=1.0 - cutoff, criterion="distance")
    matrix_names = result["variables"]
    signed = np.asarray([[math.nan if value is None else value for value in row] for row in result["spearman"]])
    by_cluster: dict[int, list[str]] = {}
    for name, label in zip(names, labels.tolist(), strict=True):
        by_cluster.setdefault(int(label), []).append(name)
    ordered = sorted(by_cluster.values(), key=lambda group: min(names.index(item) for item in group))
    assignments = []
    clusters = []
    for cluster_id, members in enumerate(ordered, start=1):
        assignments.extend({"variable": member, "cluster_id": cluster_id} for member in members)
        correlations = []
        for left_index, left in enumerate(members):
            for right in members[left_index + 1:]:
                value = signed[matrix_names.index(left), matrix_names.index(right)]
                if math.isfinite(value):
                    correlations.append(abs(float(value)))
        clusters.append({
            "cluster_id": cluster_id,
            "variable_count": len(members),
            "variables": members,
            "mean_abs_rho": float(np.mean(correlations)) if correlations else None,
            "min_abs_rho": min(correlations) if correlations else None,
            "max_abs_rho": max(correlations) if correlations else None,
        })
    return {"cutoff": cutoff, "distance_cutoff": 1.0 - cutoff, "assignments": assignments, "clusters": clusters}


def analyze_sensor_redundancy(
    path: Path,
    start_time: datetime,
    end_time: datetime,
    time_column: str,
    columns: list[str],
    config: dict | None = None,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    missing_tokens = {str(item).strip().lower() for item in cfg["missing_tokens"]}
    delimiter = detect_delimiter(path)
    timestamps: list[datetime] = []
    positions: list[int] = []
    values_by_column = {name: [] for name in columns}
    invalid_by_column = Counter({name: 0 for name in columns})
    invalid_time_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if time_column not in (reader.fieldnames or []):
            raise ValueError(f"Time column '{time_column}' does not exist.")
        missing = [name for name in columns if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Selected columns do not exist: {', '.join(missing)}")
        for position, row in enumerate(reader):
            if position % 4096 == 0 and cancelled and cancelled():
                raise AnalysisCancelled("Calculation cancelled.")
            timestamp = parse_timestamp(row.get(time_column) or "")
            if timestamp is None:
                invalid_time_rows += 1
                continue
            if timestamp < start_time or timestamp > end_time:
                continue
            timestamps.append(timestamp)
            positions.append(position)
            for name in columns:
                number, state = parse_number(row.get(name), missing_tokens)
                values_by_column[name].append(number)
                if state == "invalid":
                    invalid_by_column[name] += 1
    if not timestamps:
        raise ValueError("The selected range contains no valid timestamped rows.")
    order = sorted(range(len(timestamps)), key=lambda index: (timestamps[index], positions[index]))
    timestamps = [timestamps[index] for index in order]
    arrays = {name: np.asarray([values_by_column[name][index] for index in order], dtype=float) for name in columns}
    quality = [_quality(name, arrays[name], invalid_by_column[name], len(timestamps), cfg) for name in columns]
    if progress:
        progress(0.2)
    count = len(columns)
    spearman: list[list[float | None]] = [[None] * count for _ in range(count)]
    common_n: list[list[int]] = [[0] * count for _ in range(count)]
    pairs = []
    total_pairs = max(1, count * (count - 1) // 2)
    completed = 0
    for left_index, left in enumerate(columns):
        finite_count = int(np.isfinite(arrays[left]).sum())
        if finite_count >= int(cfg["min_pair_values"]) and np.unique(arrays[left][np.isfinite(arrays[left])]).size > 1:
            spearman[left_index][left_index] = 1.0
            common_n[left_index][left_index] = finite_count
        for right_index in range(left_index + 1, count):
            if cancelled and cancelled():
                raise AnalysisCancelled("Calculation cancelled.")
            right = columns[right_index]
            rho, pearson, used = _rank_correlation(arrays[left], arrays[right], int(cfg["min_pair_values"]))
            spearman[left_index][right_index] = spearman[right_index][left_index] = rho
            common_n[left_index][right_index] = common_n[right_index][left_index] = used
            if rho is not None:
                pairs.append({
                    "variable_a": left, "variable_b": right, "spearman_rho": rho,
                    "absolute_rho": abs(rho), "pearson_r": pearson, "common_n": used,
                })
            completed += 1
            if progress and completed % max(1, total_pairs // 20) == 0:
                progress(0.2 + 0.55 * completed / total_pairs)
    pairs.sort(key=lambda item: (-item["absolute_rho"], item["variable_a"], item["variable_b"]))
    eligible = [
        item["variable"] for item in quality
        if item["valid_n"] >= int(cfg["min_valid_values"]) and item["unique_n"] > 1
    ]
    incomplete = []
    for name in eligible:
        index = columns.index(name)
        missing_with = [other for other in eligible if other != name and spearman[index][columns.index(other)] is None]
        if missing_with:
            incomplete.append({"variable": name, "missing_with": missing_with, "message": INCOMPLETE_MATRIX_MESSAGE})
    excluded = {item["variable"] for item in incomplete}
    clusterable = [name for name in eligible if name not in excluded]
    linkage_rows: list[list[float]] = []
    leaf_order = list(clusterable)
    dendrogram_payload = {"icoord": [], "dcoord": [], "labels": leaf_order}
    if len(clusterable) > 1:
        from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
        from scipy.spatial.distance import squareform

        indices = [columns.index(name) for name in clusterable]
        distance = np.asarray([[1.0 - abs(float(spearman[i][j])) for j in indices] for i in indices], dtype=float)
        np.fill_diagonal(distance, 0.0)
        linkage_matrix = linkage(squareform(distance, checks=True), method="average", optimal_ordering=True)
        linkage_rows = linkage_matrix.tolist()
        leaf_order = [clusterable[index] for index in leaves_list(linkage_matrix).tolist()]
        rendered = dendrogram(linkage_matrix, labels=clusterable, no_plot=True)
        dendrogram_payload = {"icoord": rendered["icoord"], "dcoord": rendered["dcoord"], "labels": rendered["ivl"]}
    for item in quality:
        if item["variable"] in excluded:
            item["statuses"] = [status for status in item["statuses"] if status != "OK"] + [INCOMPLETE_MATRIX_MESSAGE]
    result = {
        "parameters": {
            "start_timestamp": start_time.isoformat(), "end_timestamp": end_time.isoformat(),
            "time_column": time_column, "selected_columns": columns, "correlation_method": "spearman",
            "clustering_distance": "1 - abs(spearman_rho)", "linkage_method": "average", "config": cfg,
        },
        "summary": {
            "timepoint_count": len(timestamps), "numeric_variable_count": len(columns),
            "clusterable_variable_count": len(clusterable), "excluded_from_clustering_count": len(incomplete),
            "invalid_time_row_count": invalid_time_rows,
        },
        "variables": columns,
        "quality": quality,
        "spearman": spearman,
        "common_n": common_n,
        "pairs": pairs,
        "clusterable_variables": clusterable,
        "clustering_exclusions": incomplete,
        "linkage": linkage_rows,
        "leaf_order": leaf_order,
        "dendrogram": dendrogram_payload,
    }
    result["cluster_cut"] = cluster_cut(result, 0.9)
    if progress:
        progress(1.0)
    return result
