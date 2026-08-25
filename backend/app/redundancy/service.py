from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import models
from app.continuity import continuity_segments
from app.database import data_dir
from app.redundancy.engine import (
    AnalysisCancelled,
    DEFAULT_CONFIG,
    analyze_sensor_redundancy,
    cluster_cut,
    parse_number,
    parse_timestamp,
    profile_csv,
)
from app.schemas import RedundancyAnalysisCreate, RedundancyAnalysisUpdate

MAX_UPLOAD_BYTES = 1_000_000_000


def _source_dir() -> Path:
    path = data_dir() / "redundancy_sources"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _locked_analysis(db: Session, analysis_id: int) -> models.RedundancyAnalysis | None:
    return db.scalar(
        select(models.RedundancyAnalysis)
        .where(models.RedundancyAnalysis.id == analysis_id)
        .with_for_update()
    )


def create_source(db: Session, stream: BinaryIO, filename: str, name: str | None = None) -> models.RedundancyCsvSource:
    safe_filename = Path(filename or "source.csv").name
    temporary = _source_dir() / f"upload-{datetime.now().timestamp():.6f}.tmp"
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as output:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("CSV upload exceeds the 1 GB limit.")
                digest.update(chunk)
                output.write(chunk)
        sha256 = digest.hexdigest()
        existing = db.scalar(select(models.RedundancyCsvSource).where(models.RedundancyCsvSource.sha256 == sha256))
        if existing is not None:
            temporary.unlink(missing_ok=True)
            return existing
        try:
            profile = profile_csv(temporary)
        except UnicodeDecodeError as exc:
            raise ValueError("The CSV must be UTF-8 encoded.") from exc
        target = _source_dir() / f"{sha256}.csv"
        temporary.replace(target)
        row = models.RedundancyCsvSource(
            name=(name or Path(safe_filename).stem or "CSV source")[:255],
            original_filename=safe_filename[:255],
            sha256=sha256,
            artifact_path=str(target),
            byte_size=size,
            delimiter=profile["delimiter"],
            encoding="utf-8",
            row_count=profile["row_count"],
            headers=profile["headers"],
            column_profiles=profile["column_profiles"],
            preview_rows=profile["preview_rows"],
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    finally:
        temporary.unlink(missing_ok=True)


def list_sources(db: Session) -> list[models.RedundancyCsvSource]:
    return list(db.scalars(select(models.RedundancyCsvSource).order_by(models.RedundancyCsvSource.created_at.desc())))


def delete_source(db: Session, source_id: int) -> bool:
    row = db.get(models.RedundancyCsvSource, source_id)
    if row is None:
        return False
    count = db.scalar(select(func.count(models.RedundancyAnalysis.id)).where(models.RedundancyAnalysis.source_id == source_id)) or 0
    if count:
        raise ValueError("CSV source is used by redundancy analyses.")
    artifact_path = Path(row.artifact_path)
    db.delete(row)
    db.commit()
    artifact_path.unlink(missing_ok=True)
    return True


def _assert_source_columns(source: models.RedundancyCsvSource, payload: RedundancyAnalysisCreate) -> None:
    headers = set(source.headers or [])
    if payload.time_column not in headers:
        raise ValueError("The selected time column does not exist in the CSV source.")
    missing = [name for name in payload.selected_columns if name not in headers]
    if missing:
        raise ValueError(f"Selected variables do not exist: {', '.join(missing)}")


def create_analysis(db: Session, payload: RedundancyAnalysisCreate) -> models.RedundancyAnalysis:
    source = db.get(models.RedundancyCsvSource, payload.source_id)
    if source is None:
        raise ValueError("CSV source not found.")
    _assert_source_columns(source, payload)
    row = models.RedundancyAnalysis(
        source_id=source.id,
        name=payload.name.strip(),
        description=payload.description,
        status="draft",
        job_status="not_calculated",
        progress=0.0,
        source_sha256=source.sha256,
        time_column=payload.time_column,
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        selected_columns=payload.selected_columns,
        config=payload.config.model_dump(),
        active_cutoff=payload.active_cutoff,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_analyses(db: Session) -> list[models.RedundancyAnalysis]:
    return list(db.scalars(select(models.RedundancyAnalysis).order_by(models.RedundancyAnalysis.created_at.desc())))


def update_analysis(db: Session, analysis_id: int, payload: RedundancyAnalysisUpdate) -> models.RedundancyAnalysis | None:
    row = db.get(models.RedundancyAnalysis, analysis_id)
    if row is None:
        return None
    if row.status == "finalized":
        raise ValueError("Finalized redundancy snapshots are immutable.")
    if row.job_status in {"queued", "running"}:
        raise ValueError("Running redundancy analyses cannot be edited.")
    values = payload.model_dump(exclude_unset=True)
    base_fields = {"time_column", "start_timestamp", "end_timestamp", "selected_columns", "config"}
    changed_base = any(field in values and getattr(row, field) != (values[field].model_dump() if field == "config" else values[field]) for field in base_fields)
    for field, value in values.items():
        setattr(row, field, value.model_dump() if field == "config" and value is not None else value)
    source = db.get(models.RedundancyCsvSource, row.source_id)
    if source is None:
        raise ValueError("CSV source not found.")
    candidate = RedundancyAnalysisCreate(
        source_id=row.source_id,
        name=row.name,
        description=row.description,
        time_column=row.time_column,
        start_timestamp=row.start_timestamp,
        end_timestamp=row.end_timestamp,
        selected_columns=list(row.selected_columns),
        config=dict(row.config),
        active_cutoff=row.active_cutoff,
    )
    _assert_source_columns(source, candidate)
    if row.end_timestamp < row.start_timestamp:
        raise ValueError("end_timestamp must be at or after start_timestamp")
    if changed_base:
        row.job_status = "stale" if row.result else "not_calculated"
    db.commit()
    db.refresh(row)
    return row


def calculate_analysis(db: Session, analysis_id: int) -> models.RedundancyAnalysis:
    row = _locked_analysis(db, analysis_id)
    if row is None:
        raise ValueError("Redundancy analysis not found.")
    if row.status == "finalized":
        raise ValueError("Finalized redundancy snapshots are immutable.")
    source = db.get(models.RedundancyCsvSource, row.source_id)
    if source is None or not Path(source.artifact_path).exists():
        raise ValueError("The stored CSV source is missing.")
    if row.cancel_requested:
        row.job_status = "cancelled"
        row.error_message = "Calculation cancelled."
        db.commit()
        db.refresh(row)
        return row
    row.job_status = "running"
    row.progress = 0.01
    row.error_message = None
    db.commit()

    def update_progress(value: float) -> None:
        row.progress = max(0.0, min(1.0, float(value)))
        db.commit()

    def cancelled() -> bool:
        db.refresh(row, attribute_names=["cancel_requested"])
        return bool(row.cancel_requested)

    try:
        result = analyze_sensor_redundancy(
            Path(source.artifact_path), row.start_timestamp, row.end_timestamp, row.time_column,
            list(row.selected_columns), dict(row.config or DEFAULT_CONFIG), update_progress, cancelled,
        )
        result["cluster_cut"] = cluster_cut(result, row.active_cutoff)
        row.result = result
        row.job_status = "ready"
        row.progress = 1.0
    except AnalysisCancelled:
        row.job_status = "cancelled"
        row.error_message = "Calculation cancelled."
    except Exception as exc:
        row.job_status = "failed"
        row.error_message = str(exc)
    db.commit()
    db.refresh(row)
    return row


def request_cancel(db: Session, analysis_id: int) -> models.RedundancyAnalysis | None:
    row = db.get(models.RedundancyAnalysis, analysis_id)
    if row is None:
        return None
    if row.job_status not in {"queued", "running"}:
        raise ValueError("Only a queued or running analysis can be cancelled.")
    row.cancel_requested = True
    db.commit()
    return row


def preview_cluster_cut(db: Session, analysis_id: int, cutoff: float) -> dict:
    row = db.get(models.RedundancyAnalysis, analysis_id)
    if row is None or row.result is None:
        raise ValueError("A completed redundancy result is required.")
    return cluster_cut(row.result, cutoff)


def finalize_analysis(db: Session, analysis_id: int, cutoff: float) -> models.RedundancyAnalysis:
    row = _locked_analysis(db, analysis_id)
    if row is None:
        raise ValueError("Redundancy analysis not found.")
    if row.status == "finalized":
        return row
    if row.job_status != "ready" or row.result is None:
        raise ValueError("Only a current completed analysis can be finalized.")
    row.active_cutoff = cutoff
    row.result = {**row.result, "cluster_cut": cluster_cut(row.result, cutoff)}
    row.status = "finalized"
    row.finalized_at = models.utc_now()
    db.commit()
    db.refresh(row)
    return row


def duplicate_analysis(db: Session, analysis_id: int) -> models.RedundancyAnalysis:
    source = db.get(models.RedundancyAnalysis, analysis_id)
    if source is None:
        raise ValueError("Redundancy analysis not found.")
    row = models.RedundancyAnalysis(
        source_id=source.source_id,
        name=f"{source.name} copy"[:255], description=source.description,
        status="draft", job_status="not_calculated", progress=0,
        source_sha256=source.source_sha256, time_column=source.time_column,
        start_timestamp=source.start_timestamp, end_timestamp=source.end_timestamp,
        selected_columns=list(source.selected_columns), config=dict(source.config), active_cutoff=source.active_cutoff,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_analysis(db: Session, analysis_id: int) -> bool:
    row = db.get(models.RedundancyAnalysis, analysis_id)
    if row is None:
        return False
    if row.job_status in {"queued", "running"}:
        raise ValueError("Running redundancy analyses cannot be deleted.")
    db.delete(row)
    db.commit()
    return True


def _series_rows(row: models.RedundancyAnalysis, columns: list[str]) -> list[dict]:
    source = row.source
    missing_tokens = {str(item).strip().lower() for item in (row.config or DEFAULT_CONFIG).get("missing_tokens", [])}
    output = []
    with Path(source.artifact_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=source.delimiter)
        for position, source_row in enumerate(reader):
            timestamp = parse_timestamp(source_row.get(row.time_column) or "")
            if timestamp is None or timestamp < row.start_timestamp or timestamp > row.end_timestamp:
                continue
            values = {}
            for column in columns:
                value, state = parse_number(source_row.get(column), missing_tokens)
                values[column] = value if state == "valid" else None
            output.append({"timestamp": timestamp, "position": position, "values": values})
    output.sort(key=lambda item: (item["timestamp"], item["position"]))
    return output


def series(
    db: Session,
    analysis_id: int,
    columns: list[str],
    max_points: int = 8000,
    offset: int = 0,
    page_size: int | None = None,
) -> dict:
    row = db.get(models.RedundancyAnalysis, analysis_id)
    if row is None:
        raise ValueError("Redundancy analysis not found.")
    invalid = [name for name in columns if name not in row.selected_columns]
    if invalid:
        raise ValueError(f"Columns are not part of this analysis: {', '.join(invalid)}")
    rows = _series_rows(row, columns)
    total = len(rows)
    segments = continuity_segments([item["timestamp"] for item in rows])
    for item, segment in zip(rows, segments, strict=True):
        item["continuity_segment"] = segment
    if page_size is not None:
        page = rows[offset:offset + page_size]
        next_offset = offset + len(page) if offset + len(page) < total else None
        return {"total": total, "decimated": False, "next_offset": next_offset, "points": page}
    if max_points > 0 and total > max_points:
        # Preserve segment edges and local extrema from every visible value
        # rather than choosing only every k-th source row.
        target_buckets = max(1, max_points // max(2, 2 * len(columns)))
        bucket_width = max(1, math.ceil(total / target_buckets))
        selected = {0, total - 1}
        for start in range(0, total, bucket_width):
            end = min(total, start + bucket_width)
            selected.update((start, end - 1))
            for column in columns:
                finite = [
                    (index, rows[index]["values"].get(column))
                    for index in range(start, end)
                    if rows[index]["values"].get(column) is not None
                ]
                if finite:
                    selected.add(min(finite, key=lambda item: float(item[1]))[0])
                    selected.add(max(finite, key=lambda item: float(item[1]))[0])
        for index in range(1, total):
            if rows[index]["continuity_segment"] != rows[index - 1]["continuity_segment"]:
                selected.update((index - 1, index))
        selected = sorted(selected)
        if len(selected) > max_points:
            stride = len(selected) / max_points
            selected = sorted({selected[min(len(selected) - 1, int(index * stride))] for index in range(max_points)})
        rows = [rows[index] for index in selected]
    return {"total": total, "decimated": len(rows) < total, "next_offset": None, "points": rows}


def export_csv(db: Session, analysis_id: int, kind: str) -> str:
    row = db.get(models.RedundancyAnalysis, analysis_id)
    if row is None or row.result is None:
        raise ValueError("A completed redundancy result is required.")
    result = row.result
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    if kind == "quality":
        writer.writerow(["Variable", "Valid N", "Missing N", "Missing %", "Mean", "Median", "Std", "Min", "Max", "Unique", "Status"])
        for item in result["quality"]:
            writer.writerow([item["variable"], item["valid_n"], item["missing_n"], item["missing_fraction"] * 100, item["mean"], item["median"], item["std"], item["min"], item["max"], item["unique_n"], "; ".join(item["statuses"])])
    elif kind == "pairs":
        writer.writerow(["Variable A", "Variable B", "Spearman rho", "absolute rho", "Pearson r", "N"])
        for item in result["pairs"]:
            writer.writerow([item["variable_a"], item["variable_b"], item["spearman_rho"], item["absolute_rho"], item["pearson_r"], item["common_n"]])
    elif kind == "clusters":
        writer.writerow(["Variable", "Cluster ID"])
        for item in result["cluster_cut"]["assignments"]:
            writer.writerow([item["variable"], item["cluster_id"]])
    elif kind == "correlation":
        writer.writerow(["Variable", *result["variables"]])
        for name, values in zip(result["variables"], result["spearman"], strict=True):
            writer.writerow([name, *values])
    else:
        raise ValueError("Unknown redundancy export.")
    return "\ufeff" + buffer.getvalue()


def export_parameters(db: Session, analysis_id: int) -> str:
    row = db.get(models.RedundancyAnalysis, analysis_id)
    if row is None or row.result is None:
        raise ValueError("A completed redundancy result is required.")
    return json.dumps({
        **row.result["parameters"], "source_id": row.source_id, "source_sha256": row.source_sha256,
        "active_cutoff": row.active_cutoff, "snapshot_status": row.status,
    }, indent=2)
