"""Service helpers for heatmap-range (video) jobs: enqueue + dedup, listing,
abort/delete, log and frame access. Jobs execute through the shared scheduler
(kind ``heatmap``); see [engine.py](engine.py)."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.database import data_dir
from app.schemas import HeatmapRangeRunCreate, HeatmapRangeRunRead
from app.testing.service import _load_testing_run_for_heatmap, _utcnow
from app.training.scheduler import scheduler


def _range_signature(
    testing_run_id: int, start: datetime, end: datetime, stride: int, scale_mode: str
) -> str:
    raw = f"{testing_run_id}|{start.isoformat()}|{end.isoformat()}|{stride}|{scale_mode}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _serialize(run: models.HeatmapRangeRun) -> HeatmapRangeRunRead:
    return HeatmapRangeRunRead.model_validate(run)


def enqueue_heatmap_range(
    db: Session, payload: HeatmapRangeRunCreate, *, wake_scheduler: bool = True
) -> HeatmapRangeRunRead:
    testing_run = _load_testing_run_for_heatmap(db, payload.testing_run_id)
    if testing_run is None:
        raise ValueError(f"Testing run does not exist: {payload.testing_run_id}")
    if testing_run.status != "finished":
        raise ValueError("Heatmap videos can only be rendered for finished testing runs.")
    if payload.end_timestamp < payload.start_timestamp:
        raise ValueError("end_timestamp must not be before start_timestamp.")

    signature = _range_signature(
        payload.testing_run_id,
        payload.start_timestamp,
        payload.end_timestamp,
        payload.stride,
        payload.scale_mode,
    )
    # Dedup: reuse an existing job with the same configuration unless it failed/aborted.
    existing = db.scalar(
        select(models.HeatmapRangeRun).where(
            models.HeatmapRangeRun.config_signature == signature,
            models.HeatmapRangeRun.status.in_(("queued", "running", "finished")),
        )
    )
    if existing is not None:
        return _serialize(existing)

    run = models.HeatmapRangeRun(
        testing_run_id=testing_run.id,
        status="queued",
        enqueued_at=_utcnow(),
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        stride=payload.stride,
        scale_mode=payload.scale_mode,
        done_count=0,
        config_signature=signature,
        testing_run_name=testing_run.name,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    if wake_scheduler:
        scheduler.wake()
    return _serialize(run)


def list_heatmap_ranges(db: Session) -> list[HeatmapRangeRunRead]:
    rows = db.scalars(
        select(models.HeatmapRangeRun).order_by(models.HeatmapRangeRun.created_at.desc())
    ).all()
    return [_serialize(row) for row in rows]


def get_heatmap_range(db: Session, run_id: int) -> HeatmapRangeRunRead | None:
    run = db.get(models.HeatmapRangeRun, run_id)
    return _serialize(run) if run is not None else None


def abort_heatmap_range(db: Session, run_id: int) -> HeatmapRangeRunRead | None:
    run = db.get(models.HeatmapRangeRun, run_id)
    if run is None:
        return None
    if run.status == "queued":
        run.status = "aborted"
        run.ended_at = _utcnow()
        run.error_message = "Aborted before it started."
        db.commit()
        db.refresh(run)
    elif run.status == "running":
        scheduler.request_abort("heatmap", run.id, run.pid)
    else:
        raise ValueError("Only queued or running jobs can be aborted.")
    return _serialize(run)


def delete_heatmap_range(db: Session, run_id: int) -> bool:
    run = db.get(models.HeatmapRangeRun, run_id)
    if run is None:
        return False
    if run.status == "running":
        raise ValueError("Abort the heatmap video before removing it.")
    shutil.rmtree(data_dir() / "heatmap_ranges" / str(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return True


def read_heatmap_range_log(db: Session, run_id: int, max_lines: int = 400) -> str | None:
    run = db.get(models.HeatmapRangeRun, run_id)
    if run is None:
        return None
    if not run.log_path:
        return ""
    try:
        with open(run.log_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return ""
    return "".join(lines[-max_lines:])


def frame_path(db: Session, run_id: int, index: int) -> Path | None:
    run = db.get(models.HeatmapRangeRun, run_id)
    if run is None or not run.frames_dir:
        return None
    path = Path(run.frames_dir) / f"frame_{index:05d}.png"
    return path if path.exists() else None
