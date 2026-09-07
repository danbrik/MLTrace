from __future__ import annotations

import base64
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import threading
import time
import zipfile

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import data_dir
from app.preprocessing.steps.load_image import LoadImageStep
from app.schemas import SpatialSensitivityPreviewRequest, SpatialSensitivityPreviewRead, SpatialSensitivityRunCreate, SpatialSensitivityRunRead
from app.training.data import ResolvedDatasetImage, enumerate_training_dataset_image_records
from app.training.scheduler import next_queue_rank, scheduler

HEIGHT, WIDTH = 960, 1280
MAD_SCALE = 1.4826
TILE_ROWS = 64


class AbortedError(Exception):
    pass


def _artifact_dir(run_id: int) -> Path:
    return data_dir() / "spatial_sensitivity_runs" / str(run_id)


def _dataset_query(dataset_id: int):
    return select(models.TrainingDataset).where(models.TrainingDataset.id == dataset_id).options(
        selectinload(models.TrainingDataset.rules).selectinload(models.TrainingDatasetRule.folder).selectinload(models.DatasetFolder.dataset)
    )


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


def preview(db: Session, payload: SpatialSensitivityPreviewRequest) -> SpatialSensitivityPreviewRead:
    dataset = db.scalar(_dataset_query(payload.training_dataset_id))
    if dataset is None:
        raise ValueError("Train/Test dataset not found.")
    records = enumerate_training_dataset_image_records(dataset)
    if payload.range_start is not None:
        records = [item for item in records if item.timestamp_parsed >= payload.range_start]
    if payload.range_end is not None:
        records = [item for item in records if item.timestamp_parsed <= payload.range_end]
    record, array = nearest_valid(records, payload.target_timestamp)
    return SpatialSensitivityPreviewRead(
        training_dataset_id=dataset.id, source_image_path=record.file_path,
        source_timestamp=record.timestamp_parsed, width=WIDTH, height=HEIGHT,
        dtype=str(array.dtype), image_data_url=_preview_png(array),
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
    run = models.SpatialSensitivityRun(status="queued", current_step="queued", enqueued_at=models.utc_now(),
        queue_rank=next_queue_rank(db), config=payload.model_dump(mode="json"), dataset_snapshot=snapshot)
    db.add(run); db.flush()
    for dataset_id in payload.training_dataset_ids:
        db.add(models.SpatialSensitivityRunDataset(run_id=run.id, training_dataset_id=dataset_id))
    db.commit(); db.refresh(run)
    if wake_scheduler:
        scheduler.wake()
    return SpatialSensitivityRunRead.model_validate(run)


def list_runs(db: Session) -> list[SpatialSensitivityRunRead]:
    return [SpatialSensitivityRunRead.model_validate(row) for row in db.scalars(select(models.SpatialSensitivityRun).order_by(models.SpatialSensitivityRun.created_at.desc())).all()]


def get_run(db: Session, run_id: int) -> SpatialSensitivityRunRead | None:
    row = db.get(models.SpatialSensitivityRun, run_id)
    return SpatialSensitivityRunRead.model_validate(row) if row else None


def abort_run(db: Session, run_id: int) -> SpatialSensitivityRunRead | None:
    run = db.get(models.SpatialSensitivityRun, run_id)
    if run is None: return None
    if run.status == "queued":
        run.status = run.current_step = "aborted"; run.ended_at = models.utc_now(); run.error_message = "Aborted before it started."
        db.commit(); db.refresh(run)
    elif run.status == "running": scheduler.request_abort("spatial_sensitivity", run.id, run.pid)
    else: raise ValueError("Only queued or running jobs can be aborted.")
    return SpatialSensitivityRunRead.model_validate(run)


def delete_run(db: Session, run_id: int) -> bool:
    run = db.get(models.SpatialSensitivityRun, run_id)
    if run is None: return False
    if run.status == "running": raise ValueError("Abort the spatial-sensitivity analysis before removing it.")
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


def _stage(records: list[ResolvedDatasetImage], path: Path, rejected: list[dict], manifest: list[dict], report_one) -> np.memmap:
    valid: list[tuple[ResolvedDatasetImage, np.ndarray]] = []
    for record in records:
        try:
            array = load_valid_uint16(record.file_path)
            valid.append((record, array)); manifest.append({"path": record.file_path, "timestamp": record.timestamp_parsed.isoformat(), "valid": True})
        except Exception as exc:
            item = {"path": record.file_path, "timestamp": record.timestamp_parsed.isoformat(), "valid": False, "reason": str(exc)}
            rejected.append(item); manifest.append(item)
        report_one(bool(valid and valid[-1][0] is record))
    if not valid: raise ValueError("Window contains no valid 1280x960 uint16 grayscale TIFF.")
    stack = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(len(valid), HEIGHT, WIDTH))
    for index, (_, array) in enumerate(valid): stack[index] = array
    stack.flush(); return stack


def _compute_staged(normal: np.memmap, event: np.memmap, epsilon: float) -> dict[str, np.ndarray]:
    output = {name: np.empty((HEIGHT, WIDTH), dtype=np.float32) for name in ("median_normal", "median_event", "mad_normal", "robust_sigma", "difference", "z_map")}
    for y in range(0, HEIGHT, TILE_ROWS):
        end = min(HEIGHT, y + TILE_ROWS)
        tile = compute_maps(normal[:, y:end], event[:, y:end], epsilon)
        for name, value in tile.items(): output[name][y:end] = value
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


def calculate(run: models.SpatialSensitivityRun, datasets: dict[int, models.TrainingDataset], abort_event, report):
    plt = _plotting()
    config, root = run.config, _artifact_dir(run.id); root.mkdir(parents=True, exist_ok=True)
    points = config["roi_points"]; mask = build_roi_mask(points, (HEIGHT, WIDTH)); np.save(root / "roi_mask.npy", mask)
    records_by_dataset = {key: sorted(enumerate_training_dataset_image_records(value), key=lambda r: (r.timestamp_parsed, r.file_path)) for key, value in datasets.items()}
    events = config["events"]; hours = float(config["normal_window_hours"]); epsilon = float(config["epsilon"])
    candidates = 0
    windows = []
    for event in events:
        start, end = datetime.fromisoformat(event["start"]), datetime.fromisoformat(event["end"])
        records = records_by_dataset[int(event["training_dataset_id"])]
        normal = [r for r in records if start - timedelta(hours=hours) <= r.timestamp_parsed < start]
        active = [r for r in records if start <= r.timestamp_parsed <= end]
        candidates += len(normal) + len(active); windows.append((event, start, end, normal, active))
    report("loading_images", 0, candidates)
    processed = successful = failed = 0
    manifest, rows, event_outputs = [], [], []
    d_store = np.lib.format.open_memmap(root / ".all_difference.npy", mode="w+", dtype=np.float32, shape=(len(events), HEIGHT, WIDTH))
    z_store = np.lib.format.open_memmap(root / ".all_z.npy", mode="w+", dtype=np.float32, shape=(len(events), HEIGHT, WIDTH))
    def counted(ok):
        nonlocal processed, successful, failed
        processed += 1; successful += int(ok); failed += int(not ok); report("loading_images", processed, candidates)
    for index, (event, start, end, normal_records, event_records) in enumerate(windows):
        if abort_event.is_set(): raise AbortedError()
        rejected_normal, rejected_event = [], []
        normal_stack = _stage(normal_records, root / f".{index}_normal_stack.npy", rejected_normal, manifest, counted)
        event_stack = _stage(event_records, root / f".{index}_event_stack.npy", rejected_event, manifest, counted)
        report("calculating_maps", index, len(events))
        maps = _compute_staged(normal_stack, event_stack, epsilon)
        for temporary in (root / f".{index}_normal_stack.npy", root / f".{index}_event_stack.npy"):
            temporary.unlink(missing_ok=True)
        safe_id = f"event_{index + 1:03d}"
        np.savez_compressed(root / f"{safe_id}_arrays.npz", **maps)
        metrics = map_metrics(maps["difference"], maps["z_map"], mask, epsilon)
        dataset = datasets[int(event["training_dataset_id"])]
        row = {"event_id": event["id"], "training_dataset_id": dataset.id, "training_dataset": dataset.name,
               "normal_start": (start - timedelta(hours=hours)).isoformat(), "normal_end": start.isoformat(),
               "event_start": start.isoformat(), "event_end": end.isoformat(), "normal_image_count": int(normal_stack.shape[0]),
               "event_image_count": int(event_stack.shape[0]), "invalid_normal_count": len(rejected_normal),
               "invalid_event_count": len(rejected_event), "epsilon": epsilon, **metrics}
        d_store[index] = maps["difference"]; z_store[index] = maps["z_map"]
        rows.append(row); event_outputs.append((safe_id, event, row))
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
    for safe_id, event, _ in event_outputs:
        with np.load(root / f"{safe_id}_arrays.npz") as saved:
            maps = {key: saved[key] for key in saved.files}
        artifact_names.append(f"{safe_id}_arrays.npz")
        artifact_names += _map_figure(maps["difference"], f"{event['id']} – absolute change D", vmax_d, points, root / f"{safe_id}_D", "Absolute intensity difference")
        artifact_names += _map_figure(maps["z_map"], f"{event['id']} – normalized change Z", vmax_z, points, root / f"{safe_id}_Z", "Robust normalized change")
    artifact_names += _map_figure(d_agg, "Aggregated absolute change", vmax_d, points, root / "aggregate_D", "Absolute intensity difference")
    artifact_names += _map_figure(z_agg, "Aggregated normalized change", vmax_z, points, root / "aggregate_Z", "Robust normalized change")
    # Shared-scale event grids.
    for metric, vmax, label in (("difference", vmax_d, "D"), ("z_map", vmax_z, "Z")):
        cols = min(3, len(events)); grid_rows = int(np.ceil(len(events) / cols)); fig, axes = plt.subplots(grid_rows, cols, figsize=(5 * cols, 4 * grid_rows), squeeze=False)
        shown = None
        for ax, (event_index, (_, event, _)) in zip(axes.flat, enumerate(event_outputs)):
            values = d_store[event_index] if metric == "difference" else z_store[event_index]
            shown = ax.imshow(np.clip(values, 0, vmax), cmap="viridis", vmin=0, vmax=vmax); _draw_roi(ax, points); ax.set_title(event["id"]); ax.axis("off")
        for ax in axes.flat[len(events):]: ax.axis("off")
        fig.colorbar(shown, ax=axes.ravel().tolist(), label=label, shrink=.8); artifact_names += _save_figure(fig, root / f"events_grid_{label}")
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * .8), 4)); x = np.arange(len(rows)); width = .38
    ax.bar(x-width/2, [r["mean_z_in"] for r in rows], width, label="Inside ROI"); ax.bar(x+width/2, [r["mean_z_out"] for r in rows], width, label="Outside ROI")
    ax.set_xticks(x, [r["event_id"] for r in rows], rotation=30, ha="right"); ax.set_ylabel("Mean Z"); ax.legend(); artifact_names += _save_figure(fig, root / "z_inside_outside")
    # Reproducible illustrative raw images plus the corresponding median products.
    example_index = next((i for i, item in enumerate(events) if item["id"] == config.get("example_event_id")), 0)
    example_event, example_start, example_end, example_normal_records, example_event_records = windows[example_index]
    normal_target = datetime.fromisoformat(config["example_normal_timestamp"]) if config.get("example_normal_timestamp") else example_start - timedelta(hours=hours / 2)
    event_target = datetime.fromisoformat(config["example_event_timestamp"]) if config.get("example_event_timestamp") else example_start + (example_end - example_start) / 2
    normal_record, normal_raw = nearest_valid(example_normal_records, normal_target)
    event_record, event_raw = nearest_valid(example_event_records, event_target)
    source_record, roi_raw = nearest_valid(records_by_dataset[int(config["roi_source_dataset_id"])], datetime.fromisoformat(config["roi_source_timestamp"]))
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
    manifest_payload = {"images": manifest, "illustrations": {"roi_source": source_record.file_path,
        "normal": normal_record.file_path, "event": event_record.file_path}, "display_range": [gray_min, gray_max]}
    (root / "input_manifest.json").write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_names += ["results.csv", "config.json", "input_manifest.json"]
    del d_store, z_store
    (root / ".all_difference.npy").unlink(missing_ok=True); (root / ".all_z.npy").unlink(missing_ok=True)
    archive = root / "spatial_sensitivity_artifacts.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.iterdir()):
            if path.is_file() and path != archive: bundle.write(path, path.name)
    artifact_names.append(archive.name)
    result = {"events": rows, "median": median_row, "vmax_d": vmax_d, "vmax_z": vmax_z,
              "epsilon": epsilon, "normal_window_hours": hours, "artifact_names": artifact_names,
              "aggregate_metrics": map_metrics(d_agg, z_agg, mask, epsilon)}
    return result, csv_path, archive, successful, failed


def run_scheduled(run_id: int, abort_event: threading.Event | None = None) -> None:
    from app.database import SessionLocal
    abort_event = abort_event or threading.Event(); started = time.perf_counter(); db = SessionLocal()
    try:
        run = db.get(models.SpatialSensitivityRun, run_id)
        if run is None: return
        run.status = "running"; run.current_step = "loading_configuration"; run.started_at = run.started_at or models.utc_now(); run.device = "CPU"; run.error_message = None; db.commit()
        def report(step, done, total):
            current = db.get(models.SpatialSensitivityRun, run_id)
            if current is None or abort_event.is_set(): raise AbortedError()
            current.current_step = step; current.processed_images = done; current.total_images = total; current.heartbeat_at = models.utc_now(); db.commit()
        datasets = {}
        for item in run.dataset_snapshot:
            dataset = db.scalar(_dataset_query(int(item["id"])))
            if dataset is None: raise ValueError(f"Train/Test dataset #{item['id']} not found.")
            datasets[dataset.id] = dataset
        try:
            result, csv_path, archive, successful, failed = calculate(run, datasets, abort_event, report)
            run = db.get(models.SpatialSensitivityRun, run_id); assert run is not None
            run.status = run.current_step = "finished"; run.ended_at = models.utc_now(); run.duration_seconds = round(time.perf_counter()-started, 3)
            run.successful_images = successful; run.failed_images = failed; run.csv_path = str(csv_path); run.archive_path = str(archive); run.result = result; db.commit()
        except AbortedError:
            db.rollback(); run = db.get(models.SpatialSensitivityRun, run_id)
            if run: run.status = run.current_step = "aborted"; run.ended_at = models.utc_now(); run.error_message = "Spatial sensitivity analysis aborted by user."; db.commit()
        except Exception as exc:
            db.rollback(); run = db.get(models.SpatialSensitivityRun, run_id)
            if run: run.status = run.current_step = "failed"; run.ended_at = models.utc_now(); run.error_message = str(exc); db.commit()
            raise
    finally: db.close()
