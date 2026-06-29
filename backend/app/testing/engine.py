"""Execution of one queued testing (inference) run.

``run_testing(run_id)`` is invoked by the testing worker subprocess (launched by
the shared scheduler). It loads the trained artifact via the existing
``ArtifactEvaluator``, runs every test image through the model's preprocessing
pipeline, stores per-image reconstruction errors, writes the results CSV, and
updates aggregates — mirroring the training engine's lifecycle (running →
finished/failed, SIGTERM → aborted).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app import models
from app.database import SessionLocal
from app.logging_setup import log_device_diagnostics
from app.preprocessing.pipeline import run_pipeline_array
from app.schemas import PreprocessingGraph
from app.testing.service import (
    ArtifactEvaluator,
    _load_training_dataset,
    _load_training_run,
    _testing_run_dir,
    _utcnow,
)
from app.training.data import enumerate_training_dataset_image_records
from app.training.data import enumerate_training_dataset_clip_samples
from app.training.engine import _to_nchw
from app.metrics.ssim import ssim_distance_map_np

logger = logging.getLogger("mltrace.testing")

# Images per (GPU-)batched reconstruction. Bounded RAM: only this many images
# are decoded/preprocessed and held at once.
_INFER_BATCH = 16

_CSV_HEADER = [
    "position", "timestamp", "image_path", "score", "full_mse",
    "roi_mse", "tile_scores_json", "width", "height", "result_metadata_json",
]


class AbortedError(Exception):
    """Raised internally when an abort signal is observed mid-inference."""


def _resolve_device(gpu_index: int | None) -> str:
    try:
        import torch

        if torch.cuda.is_available() and gpu_index is not None:
            return f"GPU:{gpu_index}"
    except Exception:  # noqa: BLE001 - torch optional / cpu-only is fine
        pass
    return "CPU"


def _clip_tensor_from_paths(graph: PreprocessingGraph, paths: list[str]) -> "np.ndarray":
    import numpy as np

    frames = [_to_nchw(run_pipeline_array(graph, path)) for path in paths]
    return np.ascontiguousarray(np.stack(frames, axis=1))


def _aggregate(values, aggregation: str) -> float:
    import numpy as np

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    if aggregation == "p95":
        return float(np.percentile(flat, 95))
    return float(np.mean(flat))


def _residual(left, right, mode: str):
    import numpy as np

    delta = left.astype(np.float64) - right.astype(np.float64)
    if mode == "squared":
        return delta * delta
    return np.abs(delta)


def _score_clip_pair(source, reconstruction, *, error_metric: str, residual_mode: str, aggregation: str, config: dict) -> tuple[float, dict]:
    if error_metric == "ssim_distance":
        values, metadata = ssim_distance_map_np(source, reconstruction, config)
        return _aggregate(values, aggregation), metadata
    if error_metric == "mae":
        return _aggregate(_residual(source, reconstruction, "absolute"), aggregation), {}
    return _aggregate(_residual(source, reconstruction, residual_mode), aggregation), {}


def _clip_frame_metadata(frames) -> list[dict]:
    return [
        {
            "path": frame.file_path,
            "timestamp": frame.timestamp_parsed.isoformat(),
            "file_name": frame.file_name,
            "dataset_name": frame.dataset_name,
            "folder_id": frame.folder_id,
        }
        for frame in frames
    ]


def _combine_scores(reconstruction_score: float, prediction_score: float | None, inference_config: dict) -> float:
    mode = str(inference_config.get("score_mode", "weighted_sum"))
    if mode == "reconstruction_only" or prediction_score is None:
        return reconstruction_score
    if mode == "prediction_only":
        return prediction_score
    rec_weight = float(inference_config.get("reconstruction_weight", 1.0))
    pred_weight = float(inference_config.get("prediction_weight", 1.0))
    denominator = max(1e-12, rec_weight + pred_weight)
    return (rec_weight * reconstruction_score + pred_weight * prediction_score) / denominator


def run_testing(run_id: int, abort_event: threading.Event | None = None) -> None:
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    db = SessionLocal()
    try:
        run = db.get(models.TestingRun, run_id)
        if run is None:
            logger.error("Testing run %s not found", run_id)
            return

        run.status = "running"
        run.started_at = _utcnow()
        run.device = _resolve_device(run.gpu_index)
        run.error_message = None
        db.commit()
        log_device_diagnostics(logger, run.gpu_index)
        logger.info("Testing run %s started on %s", run_id, run.device)

        try:
            training_run = _load_training_run(db, run.training_run_id)
            if training_run is None:
                raise ValueError("Training run no longer exists.")
            training_dataset = _load_training_dataset(db, run.training_dataset_id)
            if training_dataset is None:
                raise ValueError("Train/test dataset no longer exists.")
            roi = db.get(models.RoiDefinition, run.roi_id) if run.roi_id is not None else None

            pipeline = training_run.training_pipeline
            graph = PreprocessingGraph.model_validate(pipeline.preprocessing_pipeline.graph)
            evaluator = ArtifactEvaluator(training_run, run.inference_config)
            is_stae = pipeline.method_configuration.builder_kind == "spatiotemporal_autoencoder"
            if is_stae:
                method_config = pipeline.method_configuration.method_config or {}
                future_length = int(method_config.get("future_length") or 0) if method_config.get("prediction_branch") else 0
                clip_summary = enumerate_training_dataset_clip_samples(
                    training_dataset,
                    clip_length=int(method_config.get("clip_length") or 1),
                    future_length=future_length,
                    temporal_stride=int(method_config.get("temporal_stride") or 1),
                    future_stride=int(method_config.get("future_stride") or method_config.get("temporal_stride") or 1),
                    missing_frame_policy=str(method_config.get("missing_frame_policy") or "skip"),
                    score_timestamp_mode=str(method_config.get("score_timestamp_mode") or "last_input"),
                )
                clips = list(clip_summary.clips)
                if not clips:
                    raise ValueError("Train/test dataset produced no valid sequence clips.")
                run.expected_image_count = len(clips)
                db.commit()

                results_path = _testing_run_dir(run.id) / "reconstruction_errors.csv"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                total = len(clips)
                prep_workers = min(8, os.cpu_count() or 1)
                count = 0
                score_sum = full_sum = 0.0
                score_min: float | None = None
                score_max: float | None = None
                inference_config = {**(pipeline.method_configuration.inference_config or {}), **(run.inference_config or {})}
                error_metric = str(inference_config.get("error_metric") or ("mse" if str(inference_config.get("residual_mode", "absolute")) == "squared" else "mae"))
                residual_mode = str(inference_config.get("residual_mode", "absolute"))
                aggregation = str(inference_config.get("frame_score_aggregation", "mean"))

                def _prep_clip(clip):
                    input_paths = [frame.file_path for frame in clip.input_frames]
                    future_paths = [frame.file_path for frame in clip.future_frames]
                    input_tensor = _clip_tensor_from_paths(graph, input_paths)
                    future_tensor = _clip_tensor_from_paths(graph, future_paths) if future_paths else None
                    return clip, input_tensor, future_tensor

                with open(results_path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(_CSV_HEADER)
                    with ThreadPoolExecutor(max_workers=prep_workers) as pool:
                        for start in range(0, total, _INFER_BATCH):
                            if abort_event.is_set():
                                raise AbortedError()
                            prepared = list(pool.map(_prep_clip, clips[start : start + _INFER_BATCH]))
                            outputs = evaluator.reconstruct_clip_batch([item[1] for item in prepared])
                            mappings = []
                            for offset, ((clip, input_tensor, future_tensor), output) in enumerate(zip(prepared, outputs)):
                                position = start + offset
                                reconstruction = output["reconstruction"]
                                prediction = output["prediction"]
                                reconstruction_score, reconstruction_metric_metadata = _score_clip_pair(
                                    input_tensor,
                                    reconstruction,
                                    error_metric=error_metric,
                                    residual_mode=residual_mode,
                                    aggregation=aggregation,
                                    config=inference_config,
                                )
                                prediction_score = None
                                future_scores = []
                                prediction_metric_metadata = {}
                                if future_tensor is not None and prediction is not None:
                                    prediction_score, prediction_metric_metadata = _score_clip_pair(
                                        future_tensor,
                                        prediction,
                                        error_metric=error_metric,
                                        residual_mode=residual_mode,
                                        aggregation=aggregation,
                                        config=inference_config,
                                    )
                                    for horizon in range(future_tensor.shape[1]):
                                        horizon_score, horizon_metadata = _score_clip_pair(
                                            future_tensor[:, horizon : horizon + 1],
                                            prediction[:, horizon : horizon + 1],
                                            error_metric=error_metric,
                                            residual_mode=residual_mode,
                                            aggregation=aggregation,
                                            config=inference_config,
                                        )
                                        future_scores.append(
                                            {
                                                "horizon": horizon + 1,
                                                "score": horizon_score,
                                                "score_metric": error_metric,
                                                "ssim_parameters": horizon_metadata or None,
                                            }
                                        )
                                combined = _combine_scores(reconstruction_score, prediction_score, inference_config)
                                metadata = {
                                    "sample_kind": "clip",
                                    "clip_start": clip.clip_start.isoformat(),
                                    "clip_end": clip.clip_end.isoformat(),
                                    "score_timestamp_mode": method_config.get("score_timestamp_mode", "last_input"),
                                    "input_frames": _clip_frame_metadata(clip.input_frames),
                                    "future_frames": _clip_frame_metadata(clip.future_frames),
                                    "reconstruction_score": reconstruction_score,
                                    "prediction_score": prediction_score,
                                    "combined_score": combined,
                                    "future_scores": future_scores,
                                    "score_metric": error_metric,
                                    "ssim_parameters": reconstruction_metric_metadata or prediction_metric_metadata or None,
                                    "residual_mode": residual_mode,
                                    "frame_score_aggregation": aggregation,
                                }
                                first = clip.input_frames[0]
                                mappings.append({
                                    "testing_run_id": run.id,
                                    "position": position,
                                    "image_path": first.file_path,
                                    "timestamp": clip.score_timestamp,
                                    "score": combined,
                                    "full_mse": reconstruction_score,
                                    "roi_mse": prediction_score,
                                    "tile_scores": future_scores,
                                    "result_metadata": metadata,
                                    "width": int(input_tensor.shape[3]),
                                    "height": int(input_tensor.shape[2]),
                                })
                                writer.writerow([
                                    position,
                                    clip.score_timestamp.isoformat(),
                                    first.file_path,
                                    combined,
                                    reconstruction_score,
                                    "" if prediction_score is None else prediction_score,
                                    json.dumps(future_scores, sort_keys=True),
                                    int(input_tensor.shape[3]),
                                    int(input_tensor.shape[2]),
                                    json.dumps(metadata, sort_keys=True),
                                ])
                                count += 1
                                score_sum += combined
                                full_sum += reconstruction_score
                                score_min = combined if score_min is None else min(score_min, combined)
                                score_max = combined if score_max is None else max(score_max, combined)
                            db.bulk_insert_mappings(models.TestingRunResult, mappings)
                            run.image_count = start + len(prepared)
                            db.commit()
                            if (start // _INFER_BATCH) % 20 == 0:
                                rate = count / max(1e-6, time.perf_counter() - started)
                                logger.info("STAE testing run %s: %s/%s (%.0f clips/s)", run_id, count, total, rate)

                run.status = "finished"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.image_count = count
                run.score_mean = score_sum / count if count else None
                run.score_min = score_min
                run.score_max = score_max
                run.full_mse_mean = full_sum / count if count else None
                run.roi_mse_mean = None
                run.results_path = str(results_path)
                run.results_size_bytes = results_path.stat().st_size
                db.commit()
                logger.info("STAE testing run %s finished (%s clips)", run_id, count)
                return

            records = enumerate_training_dataset_image_records(training_dataset)
            if not records:
                raise ValueError("Train/test dataset produced no images.")
            run.expected_image_count = len(records)
            db.commit()

            def _prep(record):
                return run_pipeline_array(graph, record.file_path)

            # Streaming, batched inference: preprocess a batch (parallel), score it
            # with one reconstruction pass, write rows via bulk insert + the CSV
            # incrementally, and keep only running aggregates. Nothing accumulates
            # in RAM, so this scales to hundreds of thousands of images.
            results_path = _testing_run_dir(run.id) / "reconstruction_errors.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            total = len(records)
            prep_workers = min(8, os.cpu_count() or 1)

            count = 0
            score_sum = full_sum = roi_sum = 0.0
            roi_count = 0
            score_min: float | None = None
            score_max: float | None = None

            with open(results_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(_CSV_HEADER)
                with ThreadPoolExecutor(max_workers=prep_workers) as pool:
                    for start in range(0, total, _INFER_BATCH):
                        if abort_event.is_set():
                            raise AbortedError()
                        batch = records[start : start + _INFER_BATCH]
                        images = list(pool.map(_prep, batch))
                        scored = evaluator.score_batch(images, roi)
                        mappings = []
                        for offset, (record, (full_mse, roi_mse, width, height, tile_scores, score_metadata)) in enumerate(
                            zip(batch, scored)
                        ):
                            position = start + offset
                            fast_meta = score_metadata.get("fast_anogan") if isinstance(score_metadata, dict) else None
                            score = (
                                float(fast_meta["combined_score"])
                                if isinstance(fast_meta, dict) and "combined_score" in fast_meta
                                else (roi_mse if roi_mse is not None else full_mse)
                            )
                            mappings.append({
                                "testing_run_id": run.id,
                                "position": position,
                                "image_path": record.file_path,
                                "timestamp": record.timestamp_parsed,
                                "score": score,
                                "full_mse": full_mse,
                                "roi_mse": roi_mse,
                                "tile_scores": tile_scores,
                                "result_metadata": {"sample_kind": "image", **score_metadata},
                                "width": width,
                                "height": height,
                            })
                            writer.writerow([
                                position,
                                record.timestamp_parsed.isoformat(),
                                record.file_path,
                                score,
                                full_mse,
                                "" if roi_mse is None else roi_mse,
                                "" if tile_scores is None else json.dumps(tile_scores, sort_keys=True),
                                width,
                                height,
                                json.dumps({"sample_kind": "image", **score_metadata}, sort_keys=True),
                            ])
                            count += 1
                            score_sum += score
                            full_sum += full_mse
                            score_min = score if score_min is None else min(score_min, score)
                            score_max = score if score_max is None else max(score_max, score)
                            if roi_mse is not None:
                                roi_sum += roi_mse
                                roi_count += 1
                        db.bulk_insert_mappings(models.TestingRunResult, mappings)
                        run.image_count = start + len(batch)
                        db.commit()
                        if (start // _INFER_BATCH) % 20 == 0:
                            rate = count / max(1e-6, time.perf_counter() - started)
                            logger.info("Testing run %s: %s/%s (%.0f img/s)", run_id, count, total, rate)

            run.status = "finished"
            run.ended_at = _utcnow()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            run.image_count = count
            run.score_mean = score_sum / count if count else None
            run.score_min = score_min
            run.score_max = score_max
            run.full_mse_mean = full_sum / count if count else None
            run.roi_mse_mean = roi_sum / roi_count if roi_count else None
            run.results_path = str(results_path)
            run.results_size_bytes = results_path.stat().st_size
            db.commit()
            logger.info("Testing run %s finished (%s images)", run_id, count)
        except AbortedError:
            db.rollback()
            run = db.get(models.TestingRun, run_id)
            if run is not None:
                run.status = "aborted"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = "Testing aborted by user."
                db.commit()
            logger.info("Testing run %s aborted", run_id)
        except Exception as exc:  # noqa: BLE001 - record any failure on the run row
            db.rollback()
            run = db.get(models.TestingRun, run_id)
            if run is not None:
                run.status = "failed"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = str(exc)
                db.commit()
            logger.exception("Testing run %s failed", run_id)
    finally:
        db.close()
