from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
import shutil
import threading
import time
import zipfile

import cv2
import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import data_dir
from app.preprocessing.steps.load_image import LoadImageStep
from app.preprocessing.steps.warp_perspective import WarpPerspectiveStep
from app.preprocessing.pipeline import encode_absolute_image_data_url
from app.schemas import (
    SpatialSensitivityConfigurationCreate, SpatialSensitivityConfigurationRead,
    SpatialSensitivityPreviewRequest, SpatialSensitivityPreviewRead,
    SpatialSensitivityRunCreate, SpatialSensitivityRunRead,
    SpatialSensitivityWarpPreviewRequest, SpatialSensitivityWarpPreviewRead,
)
from app.training.data import ResolvedDatasetImage, enumerate_training_dataset_image_records
from app.training.folder_time_index import iter_training_dataset_range_records, nearest_training_dataset_records
from app.training.scheduler import next_queue_rank, scheduler

HEIGHT, WIDTH = 960, 1280
MAD_SCALE = 1.4826
COMPUTE_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024
LOAD_WORKERS = 4
LOAD_PREFETCH = 8
# Optional test/debug override; production uses the memory-budget calculation.
TILE_ROWS: int | None = None


class AbortedError(Exception):
    pass


def _artifact_dir(run_id: int) -> Path:
    return data_dir() / "spatial_sensitivity_runs" / str(run_id)


def _dataset_query(dataset_id: int):
    return select(models.TrainingDataset).where(models.TrainingDataset.id == dataset_id).options(
        selectinload(models.TrainingDataset.rules).selectinload(models.TrainingDatasetRule.folder).selectinload(models.DatasetFolder.dataset)
    )


def configuration_signature(config: dict) -> str:
    normalized = SpatialSensitivityRunCreate.model_validate({**config, "configuration_id": None}).analysis_config()
    if normalized.get("warp_preview_config") is None:
        normalized.pop("warp_preview_config", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matching_finished_run(db: Session, signature: str) -> models.SpatialSensitivityRun | None:
    return db.scalar(select(models.SpatialSensitivityRun).where(
        models.SpatialSensitivityRun.config_signature == signature,
        models.SpatialSensitivityRun.status == "finished",
        models.SpatialSensitivityRun.abort_requested_at.is_(None),
    ).order_by(models.SpatialSensitivityRun.ended_at.desc(), models.SpatialSensitivityRun.id.desc()))


def _configuration_read(db: Session, row: models.SpatialSensitivityConfiguration) -> SpatialSensitivityConfigurationRead:
    run = _matching_finished_run(db, row.config_signature)
    return SpatialSensitivityConfigurationRead(
        id=row.id, name=row.name, description=row.description, config=row.config,
        config_signature=row.config_signature,
        latest_finished_run_id=run.id if run else None,
        latest_finished_at=run.ended_at if run else None,
        created_at=row.created_at, updated_at=row.updated_at,
    )


def list_configurations(db: Session) -> list[SpatialSensitivityConfigurationRead]:
    rows = db.scalars(select(models.SpatialSensitivityConfiguration).order_by(models.SpatialSensitivityConfiguration.updated_at.desc())).all()
    return [_configuration_read(db, row) for row in rows]


def get_configuration(db: Session, configuration_id: int) -> SpatialSensitivityConfigurationRead | None:
    row = db.get(models.SpatialSensitivityConfiguration, configuration_id)
    return _configuration_read(db, row) if row else None


def _configuration_with_name(db: Session, name: str, exclude_id: int | None = None):
    query = select(models.SpatialSensitivityConfiguration).where(func.lower(models.SpatialSensitivityConfiguration.name) == name.lower())
    if exclude_id is not None: query = query.where(models.SpatialSensitivityConfiguration.id != exclude_id)
    return db.scalar(query)


def create_configuration(db: Session, payload: SpatialSensitivityConfigurationCreate) -> SpatialSensitivityConfigurationRead:
    name = payload.name.strip()
    if not name: raise ValueError("Configuration name is required.")
    if _configuration_with_name(db, name): raise ValueError(f"Configuration name already exists: {name}")
    row = models.SpatialSensitivityConfiguration(name=name, description=payload.description, config=payload.config,
        config_signature=configuration_signature(payload.config))
    db.add(row); db.commit(); db.refresh(row); return _configuration_read(db, row)


def update_configuration(db: Session, configuration_id: int, payload: SpatialSensitivityConfigurationCreate) -> SpatialSensitivityConfigurationRead | None:
    row = db.get(models.SpatialSensitivityConfiguration, configuration_id)
    if row is None: return None
    name = payload.name.strip()
    if not name: raise ValueError("Configuration name is required.")
    if _configuration_with_name(db, name, configuration_id): raise ValueError(f"Configuration name already exists: {name}")
    row.name = name; row.description = payload.description; row.config = payload.config
    row.config_signature = configuration_signature(payload.config)
    db.commit(); db.refresh(row); return _configuration_read(db, row)


def delete_configuration(db: Session, configuration_id: int) -> bool:
    row = db.get(models.SpatialSensitivityConfiguration, configuration_id)
    if row is None: return False
    db.delete(row); db.commit(); return True


def load_valid_uint16(path: str) -> np.ndarray:
    array = LoadImageStep().apply(None, {"mode": "unchanged", "dtype": "uint16"}, {"source_image_path": path})
    if array.ndim != 2:
        raise ValueError(f"expected a two-dimensional grayscale TIFF, got shape {array.shape}")
    if array.shape != (HEIGHT, WIDTH):
        raise ValueError(f"expected {WIDTH}x{HEIGHT}, got {array.shape[1]}x{array.shape[0]}")
    return np.asarray(array, dtype=np.uint16)


def _preview_png(array: np.ndarray) -> str:
    lo, hi = np.quantile(array, [0.005, 0.995])
    if hi <= lo:
        rendered = np.zeros(array.shape, dtype=np.uint8)
    else:
        rendered = np.clip((array.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise ValueError("Could not encode preview image.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def nearest_valid(records: list[ResolvedDatasetImage], target: datetime) -> tuple[ResolvedDatasetImage, np.ndarray]:
    ordered = sorted(records, key=lambda item: (abs((item.timestamp_parsed - target).total_seconds()), item.timestamp_parsed, item.file_path))
    errors: list[str] = []
    for record in ordered:
        try:
            return record, load_valid_uint16(record.file_path)
        except Exception as exc:  # invalid inputs are deliberately skipped
            errors.append(f"{record.file_path}: {exc}")
    raise ValueError("No valid 1280x960 uint16 grayscale TIFF is available." + (f" First error: {errors[0]}" if errors else ""))


def _nearest_dataset_valid(dataset, target: datetime, range_start: datetime | None = None, range_end: datetime | None = None):
    if not dataset.rules:
        records = enumerate_training_dataset_image_records(dataset)
        if range_start is not None: records = [item for item in records if item.timestamp_parsed >= range_start]
        if range_end is not None: records = [item for item in records if item.timestamp_parsed <= range_end]
        return nearest_valid(records, target)
    seen: set[str] = set(); errors: list[str] = []
    limit = 16
    while True:
        records = nearest_training_dataset_records(dataset, target, start=range_start, end=range_end, limit_per_rule=limit)
        new_records = [record for record in records if record.file_path not in seen]
        for record in new_records:
            seen.add(record.file_path)
            try: return record, load_valid_uint16(record.file_path)
            except Exception as exc: errors.append(f"{record.file_path}: {exc}")
        if len(new_records) == 0 or len(records) < limit:
            break
        limit *= 2
    raise ValueError("No valid 1280x960 uint16 grayscale TIFF is available." + (f" First error: {errors[0]}" if errors else ""))


def preview(db: Session, payload: SpatialSensitivityPreviewRequest) -> SpatialSensitivityPreviewRead:
    dataset = db.scalar(_dataset_query(payload.training_dataset_id))
    if dataset is None:
        raise ValueError("Train/Test dataset not found.")
    record, array = _nearest_dataset_valid(dataset, payload.target_timestamp, payload.range_start, payload.range_end)
    return SpatialSensitivityPreviewRead(
        training_dataset_id=dataset.id, source_image_path=record.file_path,
        source_timestamp=record.timestamp_parsed, width=WIDTH, height=HEIGHT,
        dtype=str(array.dtype), image_data_url=_preview_png(array),
    )


def warp_preview(db: Session, payload: SpatialSensitivityWarpPreviewRequest) -> SpatialSensitivityWarpPreviewRead:
    dataset = db.scalar(_dataset_query(payload.training_dataset_id))
    if dataset is None: raise ValueError("Train/Test dataset not found.")
    record, array = _nearest_dataset_valid(dataset, payload.target_timestamp, payload.range_start, payload.range_end)
    config = payload.warp.model_dump(mode="json")
    transformed = WarpPerspectiveStep().apply(array, config, {})
    return SpatialSensitivityWarpPreviewRead(
        training_dataset_id=dataset.id, source_timestamp=record.timestamp_parsed,
        input_width=array.shape[1], input_height=array.shape[0],
        output_width=transformed.shape[1], output_height=transformed.shape[0],
        output_shape_mode=config["output_shape_mode"], interpolation=config["interpolation"],
        image_data_url=encode_absolute_image_data_url(transformed),
    )


def build_roi_mask(points: list[dict], shape: tuple[int, int] = (HEIGHT, WIDTH)) -> np.ndarray:
    polygon = np.rint([[float(point["x"]), float(point["y"])] for point in points]).astype(np.int32)
    if np.any(polygon[:, 0] < 0) or np.any(polygon[:, 0] >= shape[1]) or np.any(polygon[:, 1] < 0) or np.any(polygon[:, 1] >= shape[0]):
        raise ValueError("ROI points must lie inside the original image.")
    if abs(float(cv2.contourArea(polygon))) < 1:
        raise ValueError("ROI polygon is empty.")
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    if int(mask.sum()) in {0, mask.size}:
        raise ValueError("ROI must contain pixels and leave pixels outside it.")
    return mask.astype(bool)


def compute_maps(normal: np.ndarray, event: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    normal = np.asarray(normal)
    event = np.asarray(event)
    if normal.ndim != 3 or event.ndim != 3 or normal.shape[1:] != event.shape[1:]:
        raise ValueError("Normal and event stacks must be non-empty and have the same image shape.")
    median_normal = np.median(normal, axis=0).astype(np.float32)
    median_event = np.median(event, axis=0).astype(np.float32)
    mad = np.median(np.abs(normal.astype(np.float32) - median_normal), axis=0).astype(np.float32)
    difference = np.abs(median_event - median_normal).astype(np.float32)
    robust_sigma = (MAD_SCALE * mad).astype(np.float32)
    z_map = (difference / (robust_sigma + float(epsilon))).astype(np.float32)
    return {"median_normal": median_normal, "median_event": median_event, "mad_normal": mad,
            "robust_sigma": robust_sigma, "difference": difference, "z_map": z_map}


def map_metrics(difference: np.ndarray, z_map: np.ndarray, mask: np.ndarray, epsilon: float) -> dict[str, float | int]:
    inside, outside = mask, ~mask
    d_in, d_out = float(difference[inside].mean()), float(difference[outside].mean())
    z_in, z_out = float(z_map[inside].mean()), float(z_map[outside].mean())
    c_in, c_out = float(difference[inside].sum(dtype=np.float64)), float(difference[outside].sum(dtype=np.float64))
    total = c_in + c_out
    return {"area_in": int(inside.sum()), "area_out": int(outside.sum()), "mean_d_in": d_in, "mean_d_out": d_out,
            "q_d": d_in / (d_out + epsilon), "mean_z_in": z_in, "mean_z_out": z_out,
            "q_z": z_in / (z_out + epsilon), "c_in": c_in, "c_out": c_out,
            "p_in": (100.0 * c_in / total) if total else 0.0}


def enqueue(db: Session, payload: SpatialSensitivityRunCreate, *, wake_scheduler: bool = True) -> SpatialSensitivityRunRead:
    datasets = list(db.scalars(select(models.TrainingDataset).where(models.TrainingDataset.id.in_(payload.training_dataset_ids))).all())
    if len(datasets) != len(payload.training_dataset_ids):
        raise ValueError("One or more Train/Test datasets were not found.")
    snapshot = [{"id": item.id, "name": item.name, "usage_label": item.usage_label, "updated_at": item.updated_at.isoformat() if item.updated_at else None} for item in sorted(datasets, key=lambda x: x.id)]
    analysis_config = payload.analysis_config()
    signature = configuration_signature(analysis_config)
    if payload.configuration_id is not None:
        saved = db.get(models.SpatialSensitivityConfiguration, payload.configuration_id)
        if saved is None: raise ValueError("Spatial-sensitivity configuration not found.")
        if saved.config_signature != signature: raise ValueError("Run configuration differs from the selected saved configuration.")
    run = models.SpatialSensitivityRun(status="queued", current_step="queued", enqueued_at=models.utc_now(),
        queue_rank=next_queue_rank(db), configuration_id=payload.configuration_id, config_signature=signature,
        config=analysis_config, dataset_snapshot=snapshot)
    db.add(run); db.flush()
    for dataset_id in payload.training_dataset_ids:
        db.add(models.SpatialSensitivityRunDataset(run_id=run.id, training_dataset_id=dataset_id))
    db.commit(); db.refresh(run)
    if wake_scheduler:
        scheduler.wake()
    return SpatialSensitivityRunRead.model_validate(run)


def _run_read(row: models.SpatialSensitivityRun) -> SpatialSensitivityRunRead:
    result = SpatialSensitivityRunRead.model_validate(row)
    if row.abort_requested_at and row.pid is not None:
        # Older detached workers may publish a late terminal status. Until the
        # scheduler confirms their exit, clients must keep polling cancellation.
        result.status = "running"
        result.current_step = "abort_requested"
        result.result = None
    return result


def list_runs(db: Session) -> list[SpatialSensitivityRunRead]:
    return [_run_read(row) for row in db.scalars(select(models.SpatialSensitivityRun).order_by(models.SpatialSensitivityRun.created_at.desc())).all()]


def get_run(db: Session, run_id: int) -> SpatialSensitivityRunRead | None:
    row = db.get(models.SpatialSensitivityRun, run_id)
    return _run_read(row) if row else None


def abort_run(db: Session, run_id: int) -> SpatialSensitivityRunRead | None:
    run = db.get(models.SpatialSensitivityRun, run_id)
    if run is None: return None
    now = models.utc_now()
    db.execute(update(models.SpatialSensitivityRun).where(
        models.SpatialSensitivityRun.id == run_id, models.SpatialSensitivityRun.status == "queued",
    ).values(status="aborted", current_step="aborted", abort_requested_at=now, ended_at=now,
             error_message="Aborted before it started."))
    db.execute(update(models.SpatialSensitivityRun).where(
        models.SpatialSensitivityRun.id == run_id, models.SpatialSensitivityRun.status == "running",
        models.SpatialSensitivityRun.abort_requested_at.is_(None),
    ).values(abort_requested_at=now))
    db.commit(); db.refresh(run)
    # The durable request is handled by the scheduler in the correct project.
    scheduler.wake()
    return _run_read(run)


def delete_run(db: Session, run_id: int) -> bool:
    run = db.get(models.SpatialSensitivityRun, run_id)
    if run is None: return False
    if run.status == "running" or (run.abort_requested_at and run.pid is not None):
        raise ValueError("Wait for the spatial-sensitivity worker to stop before removing it.")
    shutil.rmtree(_artifact_dir(run.id), ignore_errors=True); db.delete(run); db.commit(); return True


def artifact_path(db: Session, run_id: int, name: str) -> Path | None:
    run = db.get(models.SpatialSensitivityRun, run_id)
    if run is None or Path(name).name != name: return None
    path = _artifact_dir(run_id) / name
    return path if path.is_file() else None


def read_log(db: Session, run_id: int) -> str | None:
    run = db.get(models.SpatialSensitivityRun, run_id)
    if run is None: return None
    try: return Path(run.log_path).read_text(encoding="utf-8", errors="replace")[-100_000:] if run.log_path else ""
    except OSError: return ""


def _abort_if_requested(abort_event) -> None:
    if abort_event.is_set():
        raise AbortedError()


def _iter_window_records(dataset, start: datetime, end: datetime, *, end_inclusive: bool, abort_event, pulse=None):
    """Use the scalable resolver in production while retaining lightweight test datasets."""
    if dataset.rules:
        try:
            def checked_pulse():
                _abort_if_requested(abort_event)
                if pulse: pulse()
            yield from iter_training_dataset_range_records(
                dataset, start, end, end_inclusive=end_inclusive,
                abort_check=checked_pulse,
            )
        except sqlite3.OperationalError:
            _abort_if_requested(abort_event)
            raise
        return
    for record in enumerate_training_dataset_image_records(dataset):
        if record.timestamp_parsed >= start and (record.timestamp_parsed <= end if end_inclusive else record.timestamp_parsed < end):
            yield record


def _candidate_key(seed: int, dataset_id: int, event_id: str, window_kind: str, start: datetime, end: datetime, path: str) -> str:
    payload = f"{seed}\0{dataset_id}\0{event_id}\0{window_kind}\0{start.isoformat()}\0{end.isoformat()}\0{path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_candidate_table(
    dataset, start: datetime, end: datetime, *, end_inclusive: bool,
    event_id: str, window_kind: str, seed: int, path: Path, abort_event, heartbeat=None,
) -> int:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE candidates (random_key TEXT NOT NULL, path TEXT NOT NULL UNIQUE, timestamp TEXT NOT NULL, folder_id INTEGER NOT NULL, folder_relative_path TEXT NOT NULL, file_name TEXT NOT NULL)")
        batch = []
        last_heartbeat = time.monotonic()
        def pulse():
            nonlocal last_heartbeat
            if heartbeat and time.monotonic() - last_heartbeat >= 2:
                heartbeat(); last_heartbeat = time.monotonic()
        def sqlite_progress() -> int:
            pulse()
            return 1 if abort_event.is_set() else 0
        connection.set_progress_handler(sqlite_progress, 10_000)
        for number, record in enumerate(_iter_window_records(dataset, start, end, end_inclusive=end_inclusive, abort_event=abort_event, pulse=pulse), 1):
            if number % 1000 == 0: _abort_if_requested(abort_event)
            batch.append((_candidate_key(seed, dataset.id, event_id, window_kind, start, end, record.file_path), record.file_path,
                          record.timestamp_parsed.isoformat(), record.folder_id, record.folder_relative_path, record.file_name))
            if len(batch) >= 2000:
                connection.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?)", batch)
                connection.commit(); batch.clear()
        if batch: connection.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?)", batch)
        connection.execute("CREATE INDEX ix_candidates_random ON candidates(random_key, path)")
        count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        connection.commit()
        return count
    except sqlite3.OperationalError:
        _abort_if_requested(abort_event)
        raise
    finally:
        connection.close()


def _candidate_records(path: Path, dataset):
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT path, timestamp, folder_id, folder_relative_path, file_name FROM candidates ORDER BY random_key, path")
        for row in rows:
            yield ResolvedDatasetImage(row[0], datetime.fromisoformat(row[1]), dataset.name, "", row[2], row[3], row[4])
    finally:
        connection.close()


def _nearest_candidate(path: Path, dataset, target: datetime) -> tuple[ResolvedDatasetImage, np.ndarray]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT path, timestamp, folder_id, folder_relative_path, file_name FROM candidates "
            "ORDER BY abs((julianday(timestamp) - julianday(?)) * 86400.0), timestamp, path",
            (target.isoformat(),),
        )
        errors = []
        for row in rows:
            record = ResolvedDatasetImage(row[0], datetime.fromisoformat(row[1]), dataset.name, "", row[2], row[3], row[4])
            try: return record, load_valid_uint16(record.file_path)
            except Exception as exc: errors.append(f"{record.file_path}: {exc}")
    raise ValueError("No valid image is available in the selected window." + (f" First error: {errors[0]}" if errors else ""))


def _load_result(record: ResolvedDatasetImage):
    try: return record, load_valid_uint16(record.file_path), None
    except Exception as exc: return record, None, str(exc)


def _stage_sample(
    candidate_path: Path, dataset, target_count: int, stack_path: Path, *,
    event_id: str, window_kind: str, abort_event, progress,
) -> tuple[np.memmap, dict, list[dict]]:
    candidate_count = 0
    with sqlite3.connect(candidate_path) as connection:
        candidate_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    capacity = min(target_count, candidate_count)
    if capacity < 1: raise ValueError(f"{event_id}: {window_kind} window contains no candidate images.")
    stack = np.lib.format.open_memmap(stack_path, mode="w+", dtype=np.uint16, shape=(capacity, HEIGHT, WIDTH))
    valid_count = attempts = 0
    selected_valid_images: list[dict[str, str]] = []
    rejected: list[dict] = []
    records = iter(_candidate_records(candidate_path, dataset))
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS, thread_name_prefix="spatial-image") as executor:
        while valid_count < capacity:
            _abort_if_requested(abort_event)
            batch = []
            for _ in range(min(LOAD_PREFETCH, capacity - valid_count)):
                try: batch.append(next(records))
                except StopIteration: break
            if not batch: break
            for record, array, reason in executor.map(_load_result, batch):
                _abort_if_requested(abort_event); attempts += 1
                if array is None:
                    rejected.append({"path": record.file_path, "timestamp": record.timestamp_parsed.isoformat(), "reason": reason})
                else:
                    stack[valid_count] = array; valid_count += 1
                    selected_valid_images.append({"path": record.file_path, "timestamp": record.timestamp_parsed.isoformat()})
                if attempts % 25 == 0 or valid_count == capacity:
                    progress(valid_count, capacity)
    if valid_count < 1:
        raise ValueError(f"{event_id}: {window_kind} window contains no valid 1280x960 uint16 grayscale TIFF.")
    stack.flush()
    summary = {"candidate_count": candidate_count, "requested_sample_size": target_count,
               "attempted_count": attempts, "valid_sample_size": valid_count,
               "rejected_attempt_count": len(rejected), "sample_shortfall": max(0, target_count - valid_count),
               "selected_valid_images": selected_valid_images}
    return stack[:valid_count], summary, rejected


def _tile_rows(sample_count: int, *, bytes_per_value: int) -> int:
    if TILE_ROWS is not None:
        return max(1, min(HEIGHT, TILE_ROWS))
    pixels = max(WIDTH, COMPUTE_MEMORY_BUDGET_BYTES // max(1, sample_count * bytes_per_value))
    return max(1, min(HEIGHT, pixels // WIDTH))


def _compute_normal_staged(normal: np.memmap, abort_event) -> dict[str, np.ndarray]:
    output = {name: np.empty((HEIGHT, WIDTH), dtype=np.float32) for name in ("median_normal", "mad_normal", "robust_sigma")}
    rows = _tile_rows(int(normal.shape[0]), bytes_per_value=8)
    for y in range(0, HEIGHT, rows):
        _abort_if_requested(abort_event); end = min(HEIGHT, y + rows)
        median = np.median(normal[:, y:end], axis=0).astype(np.float32)
        mad = np.median(np.abs(normal[:, y:end].astype(np.float32) - median), axis=0).astype(np.float32)
        output["median_normal"][y:end] = median; output["mad_normal"][y:end] = mad
        output["robust_sigma"][y:end] = MAD_SCALE * mad
    return output


def _compute_event_staged(event: np.memmap, abort_event) -> np.ndarray:
    output = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    rows = _tile_rows(int(event.shape[0]), bytes_per_value=4)
    for y in range(0, HEIGHT, rows):
        _abort_if_requested(abort_event); end = min(HEIGHT, y + rows)
        output[y:end] = np.median(event[:, y:end], axis=0).astype(np.float32)
    return output


def _save_figure(fig, base: Path) -> list[str]:
    names = []
    for suffix in ("png", "pdf"):
        path = base.with_suffix(f".{suffix}"); fig.savefig(path, dpi=300, bbox_inches="tight"); names.append(path.name)
    import matplotlib.pyplot as plt
    plt.close(fig); return names


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_roi(ax, points):
    xy = np.asarray([[p["x"], p["y"]] for p in points] + [[points[0]["x"], points[0]["y"]]])
    ax.plot(xy[:, 0], xy[:, 1], color="orange", linewidth=1.5)


def _map_figure(array, title, vmax, points, base, color_label):
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(8, 6)); image = ax.imshow(np.clip(array, 0, vmax), cmap="viridis", vmin=0, vmax=vmax)
    _draw_roi(ax, points); ax.set_title(title); ax.axis("off"); fig.colorbar(image, ax=ax, label=color_label)
    return _save_figure(fig, base)


def _cleanup_run_temporaries(run_id: int) -> None:
    root = _artifact_dir(run_id)
    if not root.is_dir(): return
    for path in root.glob(".*"):
        try:
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink(missing_ok=True)
        except OSError:
            pass


def calculate(run: models.SpatialSensitivityRun, datasets: dict[int, models.TrainingDataset], abort_event, report):
    plt = _plotting()
    config, root = run.config, _artifact_dir(run.id); root.mkdir(parents=True, exist_ok=True)
    points = config["roi_points"]; mask = build_roi_mask(points, (HEIGHT, WIDTH)); np.save(root / "roi_mask.npy", mask)
    events = config["events"]; hours = float(config["normal_window_hours"]); epsilon = float(config["epsilon"])
    normal_sample_size = int(config.get("normal_sample_size", 1000))
    event_sample_size = int(config.get("event_sample_size", 1000))
    sampling_seed = int(config.get("sampling_seed", 42))
    windows = []
    processed = successful = failed = 0
    manifest_windows, rows, event_outputs = [], [], []
    d_store = np.lib.format.open_memmap(root / ".all_difference.npy", mode="w+", dtype=np.float32, shape=(len(events), HEIGHT, WIDTH))
    z_store = np.lib.format.open_memmap(root / ".all_z.npy", mode="w+", dtype=np.float32, shape=(len(events), HEIGHT, WIDTH))
    temporary_paths: list[Path] = [root / ".all_difference.npy", root / ".all_z.npy"]
    for index, event in enumerate(events):
        _abort_if_requested(abort_event)
        start, end = datetime.fromisoformat(event["start"]), datetime.fromisoformat(event["end"])
        normal_start = datetime.fromisoformat(event["normal_start"]) if event.get("normal_start") else start - timedelta(hours=hours)
        dataset = datasets[int(event["training_dataset_id"])]
        normal_candidates = root / f".{index}_normal_candidates.sqlite3"
        event_candidates = root / f".{index}_event_candidates.sqlite3"
        temporary_paths.extend([normal_candidates, event_candidates])
        report("indexing_candidates", index * 2, len(events) * 2)
        normal_candidate_count = _build_candidate_table(dataset, normal_start, start, end_inclusive=False,
            event_id=event["id"], window_kind="normal", seed=sampling_seed, path=normal_candidates, abort_event=abort_event,
            heartbeat=lambda: report("indexing_candidates", index * 2, len(events) * 2))
        report("indexing_candidates", index * 2 + 1, len(events) * 2)
        event_candidate_count = _build_candidate_table(dataset, start, end, end_inclusive=True,
            event_id=event["id"], window_kind="event", seed=sampling_seed, path=event_candidates, abort_event=abort_event,
            heartbeat=lambda: report("indexing_candidates", index * 2 + 1, len(events) * 2))
        normal_stack_path = root / f".{index}_normal_stack.npy"; temporary_paths.append(normal_stack_path)
        normal_stack, normal_summary, rejected_normal = _stage_sample(
            normal_candidates, dataset, normal_sample_size, normal_stack_path, event_id=event["id"], window_kind="normal",
            abort_event=abort_event, progress=lambda done, total: report("loading_normal_sample", done, total))
        processed += normal_summary["attempted_count"]; successful += normal_summary["valid_sample_size"]; failed += normal_summary["rejected_attempt_count"]
        report("calculating_normal_statistics", 0, int(normal_stack.shape[0]))
        maps = _compute_normal_staged(normal_stack, abort_event)
        del normal_stack; normal_stack_path.unlink(missing_ok=True)
        event_stack_path = root / f".{index}_event_stack.npy"; temporary_paths.append(event_stack_path)
        event_stack, event_summary, rejected_event = _stage_sample(
            event_candidates, dataset, event_sample_size, event_stack_path, event_id=event["id"], window_kind="event",
            abort_event=abort_event, progress=lambda done, total: report("loading_event_sample", done, total))
        processed += event_summary["attempted_count"]; successful += event_summary["valid_sample_size"]; failed += event_summary["rejected_attempt_count"]
        report("calculating_event_statistics", 0, int(event_stack.shape[0]))
        maps["median_event"] = _compute_event_staged(event_stack, abort_event)
        del event_stack; event_stack_path.unlink(missing_ok=True)
        maps["difference"] = np.abs(maps["median_event"] - maps["median_normal"]).astype(np.float32)
        maps["z_map"] = (maps["difference"] / (maps["robust_sigma"] + epsilon)).astype(np.float32)
        safe_id = f"event_{index + 1:03d}"
        np.savez_compressed(root / f"{safe_id}_arrays.npz", **maps)
        metrics = map_metrics(maps["difference"], maps["z_map"], mask, epsilon)
        row = {"event_id": event["id"], "training_dataset_id": dataset.id, "training_dataset": dataset.name,
               "normal_start": normal_start.isoformat(), "normal_end": start.isoformat(),
               "event_start": start.isoformat(), "event_end": end.isoformat(),
               "normal_candidate_count": normal_candidate_count, "normal_requested_sample_size": normal_sample_size,
               "normal_attempted_count": normal_summary["attempted_count"], "normal_image_count": normal_summary["valid_sample_size"],
               "invalid_normal_count": normal_summary["rejected_attempt_count"], "normal_sample_shortfall": normal_summary["sample_shortfall"],
               "event_candidate_count": event_candidate_count, "event_requested_sample_size": event_sample_size,
               "event_attempted_count": event_summary["attempted_count"], "event_image_count": event_summary["valid_sample_size"],
               "invalid_event_count": event_summary["rejected_attempt_count"], "event_sample_shortfall": event_summary["sample_shortfall"],
               "sampling_seed": sampling_seed, "sampling_mode": "deterministic_uniform", "epsilon": epsilon, **metrics}
        manifest_windows.extend([
            {"event_id": event["id"], "window": "normal", **normal_summary, "rejected": rejected_normal},
            {"event_id": event["id"], "window": "event", **event_summary, "rejected": rejected_event},
        ])
        d_store[index] = maps["difference"]; z_store[index] = maps["z_map"]
        rows.append(row); event_outputs.append((safe_id, event, row, normal_candidates, event_candidates))
        windows.append((event, start, end, normal_start, normal_candidates, event_candidates))
        del maps
    report("aggregating", len(events), len(events))
    d_store.flush(); z_store.flush()
    d_agg = np.mean(d_store, axis=0, dtype=np.float64).astype(np.float32)
    z_agg = np.mean(z_store, axis=0, dtype=np.float64).astype(np.float32)
    np.savez_compressed(root / "aggregate_arrays.npz", difference=d_agg, z_map=z_agg)
    vmax_d = float(np.quantile(d_store, .995))
    vmax_z = float(np.quantile(z_store, .995))
    vmax_d = vmax_d if vmax_d > 0 else 1.0; vmax_z = vmax_z if vmax_z > 0 else 1.0
    artifact_names = ["roi_mask.npy", "aggregate_arrays.npz"]
    for safe_id, event, _, _, _ in event_outputs:
        _abort_if_requested(abort_event)
        with np.load(root / f"{safe_id}_arrays.npz") as saved:
            maps = {key: saved[key] for key in saved.files}
        artifact_names.append(f"{safe_id}_arrays.npz")
        artifact_names += _map_figure(maps["difference"], f"{event['id']} – absolute change D", vmax_d, points, root / f"{safe_id}_D", "Absolute intensity difference")
        artifact_names += _map_figure(maps["z_map"], f"{event['id']} – normalized change Z", vmax_z, points, root / f"{safe_id}_Z", "Robust normalized change")
    artifact_names += _map_figure(d_agg, "Aggregated absolute change", vmax_d, points, root / "aggregate_D", "Absolute intensity difference")
    artifact_names += _map_figure(z_agg, "Aggregated normalized change", vmax_z, points, root / "aggregate_Z", "Robust normalized change")
    # Shared-scale event grids.
    for metric, vmax, label in (("difference", vmax_d, "D"), ("z_map", vmax_z, "Z")):
        _abort_if_requested(abort_event)
        cols = min(3, len(events)); grid_rows = int(np.ceil(len(events) / cols)); fig, axes = plt.subplots(grid_rows, cols, figsize=(5 * cols, 4 * grid_rows), squeeze=False)
        shown = None
        for ax, (event_index, (_, event, _, _, _)) in zip(axes.flat, enumerate(event_outputs)):
            values = d_store[event_index] if metric == "difference" else z_store[event_index]
            shown = ax.imshow(np.clip(values, 0, vmax), cmap="viridis", vmin=0, vmax=vmax); _draw_roi(ax, points); ax.set_title(event["id"]); ax.axis("off")
        for ax in axes.flat[len(events):]: ax.axis("off")
        fig.colorbar(shown, ax=axes.ravel().tolist(), label=label, shrink=.8); artifact_names += _save_figure(fig, root / f"events_grid_{label}")
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * .8), 4)); x = np.arange(len(rows)); width = .38
    ax.bar(x-width/2, [r["mean_z_in"] for r in rows], width, label="Inside ROI"); ax.bar(x+width/2, [r["mean_z_out"] for r in rows], width, label="Outside ROI")
    ax.set_xticks(x, [r["event_id"] for r in rows], rotation=30, ha="right"); ax.set_ylabel("Mean Z"); ax.legend(); artifact_names += _save_figure(fig, root / "z_inside_outside")
    # Reproducible illustrative raw images plus the corresponding median products.
    example_index = next((i for i, item in enumerate(events) if item["id"] == config.get("example_event_id")), 0)
    example_event, example_start, example_end, example_normal_start, example_normal_candidates, example_event_candidates = windows[example_index]
    normal_target = datetime.fromisoformat(config["example_normal_timestamp"]) if config.get("example_normal_timestamp") else example_start - timedelta(hours=hours / 2)
    event_target = datetime.fromisoformat(config["example_event_timestamp"]) if config.get("example_event_timestamp") else example_start + (example_end - example_start) / 2
    example_dataset = datasets[int(example_event["training_dataset_id"])]
    normal_record, normal_raw = _nearest_candidate(example_normal_candidates, example_dataset, normal_target)
    event_record, event_raw = _nearest_candidate(example_event_candidates, example_dataset, event_target)
    source_target = datetime.fromisoformat(config["roi_source_timestamp"])
    source_dataset = datasets[int(config["roi_source_dataset_id"])]
    source_record, roi_raw = _nearest_dataset_valid(source_dataset, source_target)
    with np.load(root / f"{event_outputs[example_index][0]}_arrays.npz") as saved:
        example_maps = {key: saved[key] for key in saved.files}
    gray_values = np.concatenate([array[::8, ::8].ravel().astype(np.float32) for array in (roi_raw, normal_raw, event_raw, example_maps["median_normal"], example_maps["median_event"])])
    gray_min, gray_max = [float(value) for value in np.quantile(gray_values, [.005, .995])]
    if gray_max <= gray_min: gray_max = gray_min + 1
    for name, title, array, show_roi in (
        ("roi_source", "Original image with fixed ROI", roi_raw, True),
        ("example_normal_raw", "Selected normal raw image", normal_raw, False),
        ("example_event_raw", "Selected event raw image", event_raw, False),
        ("example_normal_median", "Normal-window temporal median", example_maps["median_normal"], False),
        ("example_event_median", "Event temporal median", example_maps["median_event"], False),
    ):
        fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(np.clip(array, gray_min, gray_max), cmap="gray", vmin=gray_min, vmax=gray_max)
        if show_roi: _draw_roi(ax, points)
        ax.set_title(title); ax.axis("off"); artifact_names += _save_figure(fig, root / name)
    panels = [
        (roi_raw, "(a) ROI source image", True, "gray", gray_min, gray_max),
        (normal_raw, "(b) Selected normal raw image", False, "gray", gray_min, gray_max),
        (event_raw, "(c) Selected event raw image", False, "gray", gray_min, gray_max),
        (example_maps["median_normal"], "(d) Normal median", False, "gray", gray_min, gray_max),
        (example_maps["median_event"], "(e) Event median", False, "gray", gray_min, gray_max),
        (example_maps["difference"], "(f) Absolute change D", True, "viridis", 0, vmax_d),
        (example_maps["z_map"], "(g) Normalized change Z", True, "viridis", 0, vmax_z),
        (d_agg, "(h) Aggregated D", True, "viridis", 0, vmax_d),
        (z_agg, "(i) Aggregated Z", True, "viridis", 0, vmax_z),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, (array, title, roi, cmap, vmin, vmax) in zip(axes.flat, panels):
        shown = ax.imshow(np.clip(array, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax)
        if roi: _draw_roi(ax, points)
        ax.set_title(title); ax.axis("off")
        if cmap == "viridis": fig.colorbar(shown, ax=ax, fraction=.046, pad=.02)
    fig.suptitle("Spatial analysis of event-related image changes", fontsize=18)
    artifact_names += _save_figure(fig, root / "publication_overview")
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key not in {"training_dataset_id"}]
    median_row = {key: "" for key in rows[0]}; median_row["event_id"] = "Median"; median_row["training_dataset"] = "All events"
    for key in numeric: median_row[key] = float(np.median([float(row[key]) for row in rows]))
    import pandas as pd
    csv_path = root / "results.csv"
    pd.DataFrame(rows + [median_row], columns=list(rows[0])).to_csv(csv_path, index=False)
    (root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_payload = {"sampling": {"mode": "deterministic_uniform", "seed": sampling_seed,
        "normal_sample_size": normal_sample_size, "event_sample_size": event_sample_size},
        "windows": manifest_windows, "illustrations": {"roi_source": source_record.file_path,
        "normal": normal_record.file_path, "event": event_record.file_path}, "display_range": [gray_min, gray_max]}
    (root / "input_manifest.json").write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_names += ["results.csv", "config.json", "input_manifest.json"]
    del d_store, z_store
    for temporary in temporary_paths: temporary.unlink(missing_ok=True)
    archive = root / "spatial_sensitivity_artifacts.zip"
    _abort_if_requested(abort_event)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.iterdir()):
            if path.is_file() and path != archive: bundle.write(path, path.name)
    artifact_names.append(archive.name)
    result = {"events": rows, "median": median_row, "vmax_d": vmax_d, "vmax_z": vmax_z,
              "epsilon": epsilon, "normal_window_hours": hours, "artifact_names": artifact_names,
              "normal_sample_size": normal_sample_size, "event_sample_size": event_sample_size,
              "sampling_seed": sampling_seed, "sampling_mode": "deterministic_uniform",
              "aggregate_metrics": map_metrics(d_agg, z_agg, mask, epsilon)}
    return result, csv_path, archive, successful, failed


def run_scheduled(run_id: int, abort_event: threading.Event | None = None) -> None:
    from app.database import SessionLocal
    abort_event = abort_event or threading.Event(); started = time.perf_counter(); db = SessionLocal()
    owns_run = False
    try:
        # Wait for the scheduler's atomic queued -> running claim. In particular,
        # never resurrect a queued job which was cancelled while Popen started.
        for _ in range(100):
            db.expire_all()
            run = db.get(models.SpatialSensitivityRun, run_id)
            if run is None or run.status not in {"queued", "running"}: return
            if run.status == "running" and run.pid == os.getpid():
                owns_run = True
                break
            db.rollback()
            if abort_event.wait(0.1): return
        else:
            return

        def write_active(**values):
            changed = db.execute(update(models.SpatialSensitivityRun).where(
                models.SpatialSensitivityRun.id == run_id,
                models.SpatialSensitivityRun.status == "running",
                models.SpatialSensitivityRun.abort_requested_at.is_(None),
            ).values(**values))
            db.commit()
            if not changed.rowcount:
                raise AbortedError()

        def report(step, done, total):
            if abort_event.is_set(): raise AbortedError()
            write_active(current_step=step, processed_images=done, total_images=total, heartbeat_at=models.utc_now())
        try:
            write_active(current_step="loading_configuration")
            datasets = {}
            for item in run.dataset_snapshot:
                dataset = db.scalar(_dataset_query(int(item["id"])))
                if dataset is None: raise ValueError(f"Train/Test dataset #{item['id']} not found.")
                datasets[dataset.id] = dataset
            result, csv_path, archive, successful, failed = calculate(run, datasets, abort_event, report)
            _abort_if_requested(abort_event)
            write_active(status="finished", current_step="finished", ended_at=models.utc_now(),
                duration_seconds=round(time.perf_counter()-started, 3), successful_images=successful,
                failed_images=failed, csv_path=str(csv_path), archive_path=str(archive), result=result)
        except AbortedError:
            db.rollback()
            # Keep the scheduler slot until this process has actually exited.
            db.execute(update(models.SpatialSensitivityRun).where(
                models.SpatialSensitivityRun.id == run_id, models.SpatialSensitivityRun.status == "running",
                models.SpatialSensitivityRun.abort_requested_at.is_(None),
            ).values(abort_requested_at=models.utc_now()))
            db.commit()
        except Exception as exc:
            db.rollback()
            db.execute(update(models.SpatialSensitivityRun).where(
                models.SpatialSensitivityRun.id == run_id, models.SpatialSensitivityRun.status == "running",
                models.SpatialSensitivityRun.abort_requested_at.is_(None),
            ).values(status="failed", current_step="failed", ended_at=models.utc_now(), error_message=str(exc)))
            db.commit()
            raise
    finally:
        if owns_run:
            _cleanup_run_temporaries(run_id)
        db.close()
