from __future__ import annotations

import bisect
import csv
from datetime import datetime
from pathlib import Path
import shutil
import threading
import time

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import data_dir
from app.metrics.ssim import ssim_map_np
from app.preprocessing.pipeline import compile_pipeline
from app.schemas import (
    PreprocessingGraph,
    ResolutionSensitivityRunCreate,
    ResolutionSensitivityRunRead,
)
from app.training.data import ResolvedDatasetImage, enumerate_training_dataset_image_records
from app.training.scheduler import next_queue_rank, scheduler


RESOLUTIONS = (840, 512, 256, 128)
FEATURES = ("mean_intensity", "q95_intensity", "spatial_std_intensity")
EPSILON = 1e-12


class AbortedError(Exception):
    pass


def _artifact_dir(run_id: int) -> Path:
    return data_dir() / "resolution_sensitivity_runs" / str(run_id)


def _serialize(run: models.ResolutionSensitivityRun) -> ResolutionSensitivityRunRead:
    return ResolutionSensitivityRunRead.model_validate(run)


def _pipeline_mapping(pipelines: list[models.PreprocessingPipeline]) -> dict[int, models.PreprocessingPipeline]:
    mapping: dict[int, models.PreprocessingPipeline] = {}
    for pipeline in pipelines:
        width, height = pipeline.output_width, pipeline.output_height
        if width != height or width not in RESOLUTIONS:
            raise ValueError(
                f"Preprocessing pipeline '{pipeline.name}' must have one of the square outputs "
                "840x840, 512x512, 256x256, or 128x128."
            )
        if width in mapping:
            raise ValueError(f"More than one preprocessing pipeline produces {width}x{height}.")
        mapping[int(width)] = pipeline
    missing = [value for value in RESOLUTIONS if value not in mapping]
    if missing:
        raise ValueError(f"Missing preprocessing pipeline for: {', '.join(f'{value}x{value}' for value in missing)}.")
    return mapping


def enqueue(db: Session, payload: ResolutionSensitivityRunCreate, *, wake_scheduler: bool = True) -> ResolutionSensitivityRunRead:
    dataset = db.get(models.TrainingDataset, payload.training_dataset_id)
    if dataset is None:
        raise ValueError("Train/Test dataset not found.")
    pipelines = list(db.scalars(select(models.PreprocessingPipeline).where(
        models.PreprocessingPipeline.id.in_(payload.pipeline_ids)
    )).all())
    if len(pipelines) != 4:
        raise ValueError("One or more preprocessing pipelines were not found.")
    mapping = _pipeline_mapping(pipelines)
    label_set = None
    if payload.label_set_id is not None:
        label_set = db.get(models.EvaluationLabelSet, payload.label_set_id)
        if label_set is None or label_set.training_dataset_id != dataset.id:
            raise ValueError("The selected label set does not belong to the selected dataset.")
    snapshot = [{
        "resolution": resolution,
        "pipeline_id": pipeline.id,
        "name": pipeline.name,
        "graph": pipeline.graph,
    } for resolution, pipeline in sorted(mapping.items(), reverse=True)]
    run = models.ResolutionSensitivityRun(
        training_dataset_id=dataset.id,
        label_set_id=label_set.id if label_set else None,
        training_dataset_name=dataset.name,
        label_set_name=label_set.name if label_set else None,
        status="queued",
        current_step="queued",
        enqueued_at=models.utc_now(),
        queue_rank=next_queue_rank(db),
        config=payload.model_dump(mode="json"),
        pipeline_snapshot=snapshot,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    if wake_scheduler:
        scheduler.wake()
    return _serialize(run)


def list_runs(db: Session) -> list[ResolutionSensitivityRunRead]:
    rows = db.scalars(select(models.ResolutionSensitivityRun).order_by(
        models.ResolutionSensitivityRun.created_at.desc()
    )).all()
    return [_serialize(row) for row in rows]


def get_run(db: Session, run_id: int) -> ResolutionSensitivityRunRead | None:
    row = db.get(models.ResolutionSensitivityRun, run_id)
    return _serialize(row) if row else None


def abort_run(db: Session, run_id: int) -> ResolutionSensitivityRunRead | None:
    run = db.get(models.ResolutionSensitivityRun, run_id)
    if run is None:
        return None
    if run.status == "queued":
        run.status = "aborted"
        run.current_step = "aborted"
        run.ended_at = models.utc_now()
        run.error_message = "Aborted before it started."
        db.commit()
        db.refresh(run)
    elif run.status == "running":
        scheduler.request_abort("resolution_sensitivity", run.id, run.pid)
    else:
        raise ValueError("Only queued or running jobs can be aborted.")
    return _serialize(run)


def delete_run(db: Session, run_id: int) -> bool:
    run = db.get(models.ResolutionSensitivityRun, run_id)
    if run is None:
        return False
    if run.status == "running":
        raise ValueError("Abort the resolution-sensitivity analysis before removing it.")
    shutil.rmtree(_artifact_dir(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return True


def read_log(db: Session, run_id: int, max_lines: int = 400) -> str | None:
    run = db.get(models.ResolutionSensitivityRun, run_id)
    if run is None:
        return None
    if not run.log_path:
        return ""
    try:
        with open(run.log_path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-max_lines:])
    except FileNotFoundError:
        return ""


def export_path(db: Session, run_id: int, kind: str) -> Path | None:
    run = db.get(models.ResolutionSensitivityRun, run_id)
    if run is None:
        return None
    raw = run.detail_csv_path if kind == "details" else run.summary_csv_path if kind == "summary" else None
    return Path(raw) if raw and Path(raw).is_file() else None


def sample_intervals(
    images: list[ResolvedDatasetImage], intervals: list[dict], samples_per_interval: int
) -> list[dict]:
    ordered = sorted(images, key=lambda image: (image.timestamp_parsed, image.file_path))
    timestamps = [image.timestamp_parsed for image in ordered]
    selected: list[dict] = []
    usage: dict[str, int] = {}
    for interval in intervals:
        start = datetime.fromisoformat(str(interval["start"]))
        end = datetime.fromisoformat(str(interval["end"]))
        left = bisect.bisect_left(timestamps, start)
        right = bisect.bisect_left(timestamps, end)
        candidates = ordered[left:right]
        candidate_times = timestamps[left:right]
        if not candidates:
            raise ValueError(f"Interval '{interval['name']}' contains no available image.")
        duration = end - start
        for sample_index in range(samples_per_interval):
            target = start + duration * ((sample_index + 0.5) / samples_per_interval)
            position = bisect.bisect_left(candidate_times, target)
            choices = [index for index in (position - 1, position) if 0 <= index < len(candidates)]
            # Stable tie-break: earlier timestamp, then path.
            nearest = min(choices, key=lambda index: (
                abs((candidate_times[index] - target).total_seconds()),
                candidate_times[index],
                candidates[index].file_path,
            ))
            image = candidates[nearest]
            previous_count = usage.get(image.file_path, 0)
            usage[image.file_path] = previous_count + 1
            selected.append({
                "interval_id": interval["id"], "interval_name": interval["name"],
                "interval_type": interval["type"], "sample_index": sample_index,
                "target_timestamp": target, "image": image, "duplicate": previous_count > 0,
            })
    return selected


def _metric_summary(values: list[float]) -> dict:
    if not values:
        return {"median": None, "q25": None, "q75": None, "iqr": None}
    q25, median, q75 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25)}


def _features(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "mean_intensity": float(np.mean(values)),
        "q95_intensity": float(np.quantile(values, 0.95)),
        "spatial_std_intensity": float(np.std(values, ddof=0)),
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def calculate(run: models.ResolutionSensitivityRun, dataset: models.TrainingDataset,
              abort_event: threading.Event, report) -> tuple[dict, Path, Path, float, int, int]:
    config = run.config
    report("resolving_images", 0, None)
    images = enumerate_training_dataset_image_records(dataset)
    samples = sample_intervals(images, config["intervals"], int(config["samples_per_interval"]))
    unique = {sample["image"].file_path: sample["image"] for sample in samples}
    snapshots = {int(item["resolution"]): item for item in run.pipeline_snapshot}
    compiled = {resolution: compile_pipeline(PreprocessingGraph.model_validate(item["graph"]))
                for resolution, item in snapshots.items()}

    data_range_override = config.get("ssim_data_range")
    reference_errors: dict[str, str] = {}
    if data_range_override is None:
        minimum, maximum = float("inf"), float("-inf")
        for index, image in enumerate(unique.values()):
            if abort_event.is_set():
                raise AbortedError()
            try:
                array = np.asarray(compiled[840].run(image.file_path))
                if array.shape[:2] != (840, 840) or not np.isfinite(array).all():
                    raise ValueError(f"840 pipeline produced invalid shape or non-finite pixels: {array.shape}")
                minimum = min(minimum, float(np.min(array)))
                maximum = max(maximum, float(np.max(array)))
            except Exception as exc:  # paired failure is finalized in the main pass
                reference_errors[image.file_path] = f"{type(exc).__name__}: {exc}"
            report("determining_data_range", index + 1, len(unique))
        data_range = maximum - minimum
        if not np.isfinite(data_range) or data_range <= 0:
            raise ValueError("SSIM data range could not be determined from the 840x840 reference images. Provide an override.")
    else:
        data_range = float(data_range_override)

    computed: dict[str, dict] = {}
    successful = 0
    failed = 0
    for index, image in enumerate(unique.values()):
        if abort_event.is_set():
            raise AbortedError()
        try:
            if image.file_path in reference_errors:
                raise ValueError(reference_errors[image.file_path])
            outputs: dict[int, np.ndarray] = {}
            channels: int | None = None
            for resolution in RESOLUTIONS:
                output = np.asarray(compiled[resolution].run(image.file_path))
                actual_channels = 1 if output.ndim == 2 else output.shape[2] if output.ndim == 3 else -1
                if output.shape[:2] != (resolution, resolution):
                    raise ValueError(f"{resolution} pipeline produced {output.shape}, expected {resolution}x{resolution}.")
                if actual_channels < 1 or (channels is not None and actual_channels != channels):
                    raise ValueError("Preprocessing pipelines produced incompatible channel counts.")
                if not np.isfinite(output).all():
                    raise ValueError(f"{resolution} pipeline produced non-finite pixels.")
                channels = actual_channels
                outputs[resolution] = output
            reference = outputs[840].astype(np.float64)
            metrics: dict[int, dict] = {}
            for resolution, output in outputs.items():
                if resolution == 840:
                    ssim, mae = 1.0, 0.0
                else:
                    upscaled = cv2.resize(output.astype(np.float64), (840, 840), interpolation=cv2.INTER_LINEAR)
                    if reference.ndim == 3 and reference.shape[2] == 1 and upscaled.ndim == 2:
                        upscaled = upscaled[..., None]
                    similarity, _ = ssim_map_np(reference, upscaled, {"ssim_data_range": data_range})
                    ssim = float(np.mean(similarity))
                    mae = float(np.mean(np.abs(reference - upscaled)))
                metrics[resolution] = {**_features(output), "ssim": ssim, "mae": mae}
            computed[image.file_path] = {"metrics": metrics, "error": ""}
            successful += 1
        except Exception as exc:
            computed[image.file_path] = {"metrics": {}, "error": f"{type(exc).__name__}: {exc}"}
            failed += 1
        report("processing_images", index + 1, len(unique))

    detail_rows: list[dict] = []
    for sample in samples:
        image = sample["image"]
        item = computed[image.file_path]
        for resolution in RESOLUTIONS:
            metric = item["metrics"].get(resolution, {})
            detail_rows.append({
                "interval_id": sample["interval_id"], "interval_name": sample["interval_name"],
                "interval_type": sample["interval_type"], "sample_index": sample["sample_index"],
                "target_timestamp": sample["target_timestamp"].isoformat(),
                "actual_timestamp": image.timestamp_parsed.isoformat(),
                "relative_path": str(Path(image.folder_relative_path) / image.file_name),
                "duplicate": str(bool(sample["duplicate"])).lower(), "resolution": resolution,
                "mean_intensity": metric.get("mean_intensity", ""),
                "q95_intensity": metric.get("q95_intensity", ""),
                "spatial_std_intensity": metric.get("spatial_std_intensity", ""),
                "ssim": metric.get("ssim", ""), "mae": metric.get("mae", ""), "error": item["error"],
            })

    valid_rows = [row for row in detail_rows if not row["error"]]
    normal = [row for row in valid_rows if row["interval_type"] == "normal"]
    if not normal:
        raise ValueError("No paired preprocessing result remains in the normal intervals.")
    normal_stats: dict[int, dict] = {}
    for resolution in RESOLUTIONS:
        normal_stats[resolution] = {}
        for feature in FEATURES:
            values = np.asarray([float(row[feature]) for row in normal if row["resolution"] == resolution])
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            normal_stats[resolution][feature] = {"median": median, "mad": mad, "robust_scale": 1.4826 * mad + EPSILON}

    event_intervals = [interval for interval in config["intervals"] if interval["type"] == "event"]
    separations: list[dict] = []
    for interval in event_intervals:
        for resolution in RESOLUTIONS:
            for feature in FEATURES:
                stat = normal_stats[resolution][feature]
                values = [float(row[feature]) for row in valid_rows
                          if row["interval_id"] == interval["id"] and row["resolution"] == resolution]
                separation = float(np.median([(value - stat["median"]) / stat["robust_scale"] for value in values])) if values else None
                separations.append({"event_id": interval["id"], "event_name": interval["name"],
                                    "resolution": resolution, "feature": feature, "separation": separation, "retention": None})
    reference_sep = {(row["event_id"], row["feature"]): row["separation"] for row in separations if row["resolution"] == 840}
    for row in separations:
        value, reference = row["separation"], reference_sep[(row["event_id"], row["feature"])]
        row["retention"] = abs(value) / (abs(reference) + EPSILON) if value is not None and reference is not None else None

    overview: list[dict] = []
    for resolution in RESOLUTIONS:
        rows = [row for row in valid_rows if row["resolution"] == resolution]
        separation_rows = [row for row in separations if row["resolution"] == resolution and row["separation"] is not None]
        feature_summary = {}
        for feature in FEATURES:
            values = [row["separation"] for row in separation_rows if row["feature"] == feature]
            feature_summary[feature] = {
                "median_separation": float(np.median(values)) if values else None,
                "minimum_absolute_separation": min((abs(value) for value in values), default=None),
            }
        weakest = min(separation_rows, key=lambda row: abs(row["separation"]), default=None)
        overview.append({
            "resolution": resolution, "pixel_count": resolution * resolution,
            "pixel_share": (resolution * resolution) / (840 * 840),
            "ssim": _metric_summary([float(row["ssim"]) for row in rows]),
            "mae": _metric_summary([float(row["mae"]) for row in rows]),
            "features": feature_summary,
            "weakest_event": weakest["event_name"] if weakest else None,
            "weakest_feature": weakest["feature"] if weakest else None,
        })

    result = {
        "sample_count": len(samples), "unique_image_count": len(unique),
        "duplicate_sample_count": sum(bool(sample["duplicate"]) for sample in samples),
        "successful_unique_images": successful, "failed_unique_images": failed,
        "data_range": data_range, "normal_statistics": normal_stats,
        "overview": overview, "separations": separations,
    }
    directory = _artifact_dir(run.id)
    detail_path, summary_path = directory / "details.csv", directory / "summary.csv"
    detail_fields = list(detail_rows[0].keys())
    _write_csv(detail_path, detail_rows, detail_fields)
    summary_rows = []
    for row in overview:
        flat = {"resolution": row["resolution"], "pixel_count": row["pixel_count"], "pixel_share": row["pixel_share"],
                "ssim_median": row["ssim"]["median"], "ssim_q25": row["ssim"]["q25"], "ssim_q75": row["ssim"]["q75"],
                "mae_median": row["mae"]["median"], "mae_q25": row["mae"]["q25"], "mae_q75": row["mae"]["q75"],
                "weakest_event": row["weakest_event"], "weakest_feature": row["weakest_feature"]}
        for feature in FEATURES:
            flat[f"{feature}_median_separation"] = row["features"][feature]["median_separation"]
            flat[f"{feature}_minimum_absolute_separation"] = row["features"][feature]["minimum_absolute_separation"]
        summary_rows.append(flat)
    _write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    return result, detail_path, summary_path, data_range, successful, failed


def run_scheduled(run_id: int, abort_event: threading.Event | None = None) -> None:
    from app.database import SessionLocal
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    db = SessionLocal()
    try:
        run = db.get(models.ResolutionSensitivityRun, run_id)
        if run is None:
            return
        run.status = "running"
        run.current_step = "loading_configuration"
        run.started_at = run.started_at or models.utc_now()
        run.device = "CPU"
        run.error_message = None
        run.heartbeat_at = models.utc_now()
        db.commit()

        def report(step: str, done: int, total: int | None) -> None:
            current = db.get(models.ResolutionSensitivityRun, run_id)
            if current is None or abort_event.is_set():
                raise AbortedError()
            current.current_step = step
            current.processed_images = done
            current.total_images = total
            current.heartbeat_at = models.utc_now()
            db.commit()

        try:
            dataset = db.scalar(select(models.TrainingDataset).where(
                models.TrainingDataset.id == run.training_dataset_id
            ).options(selectinload(models.TrainingDataset.rules).selectinload(
                models.TrainingDatasetRule.folder
            ).selectinload(models.DatasetFolder.dataset)))
            if dataset is None:
                raise ValueError("Train/Test dataset not found.")
            result, detail_path, summary_path, data_range, successful, failed = calculate(
                run, dataset, abort_event, report
            )
            run = db.get(models.ResolutionSensitivityRun, run_id)
            assert run is not None
            run.status = "finished"
            run.current_step = "finished"
            run.ended_at = models.utc_now()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            run.successful_images = successful
            run.failed_images = failed
            run.data_range = data_range
            run.detail_csv_path = str(detail_path)
            run.summary_csv_path = str(summary_path)
            run.result = result
            run.heartbeat_at = models.utc_now()
            db.commit()
        except AbortedError:
            db.rollback()
            run = db.get(models.ResolutionSensitivityRun, run_id)
            if run:
                run.status = "aborted"; run.current_step = "aborted"; run.ended_at = models.utc_now()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = "Resolution-sensitivity analysis aborted by user."
                db.commit()
        except Exception as exc:
            db.rollback()
            run = db.get(models.ResolutionSensitivityRun, run_id)
            if run:
                run.status = "failed"; run.current_step = "failed"; run.ended_at = models.utc_now()
                run.duration_seconds = round(time.perf_counter() - started, 3); run.error_message = str(exc)
                db.commit()
            raise
    finally:
        db.close()
