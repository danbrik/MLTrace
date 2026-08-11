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

from sqlalchemy import delete, select

from app import models
from app.database import SessionLocal
from app.logging_setup import log_device_diagnostics
from app.preprocessing.pipeline import ImageLoadError, run_pipeline_array
from app.schemas import PreprocessingGraph
from app.testing.service import (
    ArtifactEvaluator,
    _testing_run_dir,
    _utcnow,
    resolve_testing_input_context,
    validate_testing_checkpoint,
)
from app.testing.checkpoint import CHECKPOINT_INTERVAL, MAX_SKIPPED_PATHS, make_checkpoint_state
from app.training.engine import _to_nchw
from app.metrics.aggregation import normalize_aggregation
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


def _csv_row(values) -> list:
    tile_scores = values["tile_scores"] if isinstance(values, dict) else values.tile_scores
    metadata = values["result_metadata"] if isinstance(values, dict) else values.result_metadata
    timestamp = values["timestamp"] if isinstance(values, dict) else values.timestamp
    return [
        values["position"] if isinstance(values, dict) else values.position,
        timestamp.isoformat(),
        values["image_path"] if isinstance(values, dict) else values.image_path,
        values["score"] if isinstance(values, dict) else values.score,
        values["full_mse"] if isinstance(values, dict) else values.full_mse,
        "" if (values["roi_mse"] if isinstance(values, dict) else values.roi_mse) is None else (
            values["roi_mse"] if isinstance(values, dict) else values.roi_mse
        ),
        "" if tile_scores is None else json.dumps(tile_scores, sort_keys=True),
        values["width"] if isinstance(values, dict) else values.width,
        values["height"] if isinstance(values, dict) else values.height,
        "" if metadata is None else json.dumps(metadata, sort_keys=True),
    ]


def _write_checkpoint_prefix(db, writer, run_id: int, result_count: int) -> None:
    if result_count <= 0:
        return
    rows = db.scalars(
        select(models.TestingRunResult)
        .where(
            models.TestingRunResult.testing_run_id == run_id,
            models.TestingRunResult.position < result_count,
        )
        .order_by(models.TestingRunResult.position)
    )
    for row in rows:
        writer.writerow(_csv_row(row))


def _restore_checkpoint(db, run: models.TestingRun, context) -> dict:
    _, state = validate_testing_checkpoint(db, run, context)
    result_count = int(state["result_count"])
    db.execute(
        delete(models.TestingRunResult).where(
            models.TestingRunResult.testing_run_id == run.id,
            models.TestingRunResult.position >= result_count,
        )
    )
    run.image_count = result_count
    db.commit()
    return state


def _checkpoint_due(processed_inputs: int, previous_checkpoint: int) -> bool:
    if CHECKPOINT_INTERVAL <= 0:
        return False
    return processed_inputs >= ((previous_checkpoint // CHECKPOINT_INTERVAL) + 1) * CHECKPOINT_INTERVAL


def _store_checkpoint(
    run: models.TestingRun,
    *,
    kind: str,
    signature: str,
    input_count: int,
    result_count: int,
    total_inputs: int,
    score_sum: float,
    full_sum: float,
    roi_sum: float,
    roi_count: int,
    score_min: float | None,
    score_max: float | None,
    skipped_count: int,
    skipped_paths: list[str],
) -> None:
    run.checkpoint_at = _utcnow()
    run.checkpoint_input_count = input_count
    run.checkpoint_result_count = result_count
    run.checkpoint_state = make_checkpoint_state(
        kind=kind,
        signature=signature,
        input_count=input_count,
        result_count=result_count,
        total_inputs=total_inputs,
        score_sum=score_sum,
        full_sum=full_sum,
        roi_sum=roi_sum,
        roi_count=roi_count,
        score_min=score_min,
        score_max=score_max,
        skipped_count=skipped_count,
        skipped_paths=skipped_paths[:MAX_SKIPPED_PATHS],
    )


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
    from app.metrics.aggregation import aggregate_score

    return aggregate_score(values, aggregation)


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
            resume_requested = run.restart_mode == "checkpoint"
            context = resolve_testing_input_context(db, run)
            training_run = context.training_run
            roi = context.roi
            pipeline = training_run.training_pipeline
            graph = context.graph
            evaluator = ArtifactEvaluator(training_run, run.inference_config)
            logger.info(
                "Testing run %s loaded pipeline and artifact (builder=%s, artifact=%s)",
                run_id,
                pipeline.method_configuration.builder_kind,
                evaluator.artifact_path,
            )
            total = len(context.inputs)
            state = _restore_checkpoint(db, run, context) if resume_requested else None
            if state is None:
                input_start = count = 0
                score_sum = full_sum = roi_sum = 0.0
                roi_count = 0
                score_min: float | None = None
                score_max: float | None = None
                skipped_count = 0
                skipped_paths: list[str] = []
            else:
                try:
                    input_start = int(state["input_count"])
                    count = int(state["result_count"])
                    score_sum = float(state["score_sum"])
                    full_sum = float(state["full_sum"])
                    roi_sum = float(state["roi_sum"])
                    roi_count = int(state["roi_count"])
                    score_min = None if state.get("score_min") is None else float(state["score_min"])
                    score_max = None if state.get("score_max") is None else float(state["score_max"])
                    skipped_count = int(state["skipped_count"])
                    skipped_paths = list(state.get("skipped_paths") or [])[:MAX_SKIPPED_PATHS]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("The inference checkpoint aggregate state is incomplete. Restart completely.") from exc
                logger.info(
                    "Testing run %s resuming from checkpoint at input %s with %s results",
                    run_id,
                    input_start,
                    count,
                )

            skipped_lock = threading.Lock()

            def _record_skip(path: str) -> None:
                nonlocal skipped_count
                with skipped_lock:
                    skipped_count += 1
                    if path not in skipped_paths and len(skipped_paths) < MAX_SKIPPED_PATHS:
                        skipped_paths.append(path)

            run.expected_image_count = total
            run.image_count = count
            db.commit()
            last_checkpoint_input = input_start
            is_stae = context.kind == "clip"
            if is_stae:
                method_config = pipeline.method_configuration.method_config or {}
                clip_summary = context.clip_summary
                clips = context.inputs
                logger.info(
                    "STAE testing run %s resolved %s clips (%s skipped, %s selected frames, %s possible clips, mode=%s)",
                    run_id,
                    len(clips),
                    clip_summary.skipped_missing,
                    clip_summary.selected_frame_count,
                    clip_summary.possible_clip_count,
                    clip_summary.sequence_contiguity_mode,
                )
                logger.info("STAE testing run %s starting preprocessing for %s clips", run_id, len(clips))

                results_path = _testing_run_dir(run.id) / "reconstruction_errors.csv"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                prep_workers = min(8, os.cpu_count() or 1)
                inference_config = {**(pipeline.method_configuration.inference_config or {}), **(run.inference_config or {})}
                error_metric = str(inference_config.get("error_metric") or ("mse" if str(inference_config.get("residual_mode", "absolute")) == "squared" else "mae"))
                residual_mode = str(inference_config.get("residual_mode", "absolute"))
                aggregation = normalize_aggregation(inference_config.get("frame_score_aggregation"))

                def _prep_clip(clip):
                    input_paths = [frame.file_path for frame in clip.input_frames]
                    future_paths = [frame.file_path for frame in clip.future_frames]
                    try:
                        input_tensor = _clip_tensor_from_paths(graph, input_paths)
                        future_tensor = _clip_tensor_from_paths(graph, future_paths) if future_paths else None
                    except ImageLoadError as exc:
                        logger.warning(
                            "STAE testing run %s skipping clip with unreadable frame: %s (%s)",
                            run_id, exc.path, exc.original,
                        )
                        _record_skip(exc.path)
                        return clip, None, None
                    return clip, input_tensor, future_tensor

                def _commit_stae_progress(processed_inputs: int) -> None:
                    nonlocal last_checkpoint_input
                    run.image_count = count
                    if _checkpoint_due(processed_inputs, last_checkpoint_input):
                        _store_checkpoint(
                            run,
                            kind=context.kind,
                            signature=context.signature,
                            input_count=processed_inputs,
                            result_count=count,
                            total_inputs=total,
                            score_sum=score_sum,
                            full_sum=full_sum,
                            roi_sum=0.0,
                            roi_count=0,
                            score_min=score_min,
                            score_max=score_max,
                            skipped_count=skipped_count,
                            skipped_paths=skipped_paths,
                        )
                        last_checkpoint_input = processed_inputs
                        logger.info(
                            "STAE testing run %s saved checkpoint at %s inputs (%s results)",
                            run_id,
                            processed_inputs,
                            count,
                        )
                    db.commit()

                with open(results_path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(_CSV_HEADER)
                    _write_checkpoint_prefix(db, writer, run.id, count)
                    with ThreadPoolExecutor(max_workers=prep_workers) as pool:
                        for start in range(input_start, total, _INFER_BATCH):
                            if abort_event.is_set():
                                raise AbortedError()
                            end = min(total, start + _INFER_BATCH)
                            prepared = list(pool.map(_prep_clip, clips[start:end]))
                            prepared = [item for item in prepared if item[1] is not None]
                            if not prepared:
                                _commit_stae_progress(end)
                                continue
                            first_inference = evaluator.model is None
                            if first_inference:
                                logger.info("STAE testing run %s initializing model for first batch", run_id)
                            outputs = evaluator.reconstruct_clip_batch([item[1] for item in prepared])
                            if first_inference:
                                logger.info("STAE testing run %s completed first inference batch", run_id)
                            mappings = []
                            for (clip, input_tensor, future_tensor), output in zip(prepared, outputs):
                                position = count
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
                                writer.writerow(_csv_row(mappings[-1]))
                                count += 1
                                score_sum += combined
                                full_sum += reconstruction_score
                                score_min = combined if score_min is None else min(score_min, combined)
                                score_max = combined if score_max is None else max(score_max, combined)
                            db.bulk_insert_mappings(models.TestingRunResult, mappings)
                            _commit_stae_progress(end)
                            if (start // _INFER_BATCH) % 20 == 0:
                                rate = count / max(1e-6, time.perf_counter() - started)
                                logger.info("STAE testing run %s: %s/%s (%.0f clips/s)", run_id, count, total, rate)

                if count == 0:
                    raise ValueError(
                        f"All {total} clips failed to load; see the skipped image list."
                    )
                run.status = "finished"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.image_count = count
                run.skipped_image_count = skipped_count
                run.skipped_images = sorted(set(skipped_paths))[:MAX_SKIPPED_PATHS]
                run.score_mean = score_sum / count if count else None
                run.score_min = score_min
                run.score_max = score_max
                run.full_mse_mean = full_sum / count if count else None
                run.roi_mse_mean = None
                run.results_path = str(results_path)
                run.results_size_bytes = results_path.stat().st_size
                run.checkpoint_at = None
                run.checkpoint_input_count = None
                run.checkpoint_result_count = None
                run.checkpoint_state = None
                run.restart_mode = None
                db.commit()
                logger.info("STAE testing run %s finished (%s clips)", run_id, count)
                return

            resolution_started = time.perf_counter()
            records = context.inputs
            logger.info(
                "Testing run %s resolved %s images in %.3fs",
                run_id,
                len(records),
                time.perf_counter() - resolution_started,
            )

            def _prep(record):
                try:
                    return record, run_pipeline_array(graph, record.file_path)
                except ImageLoadError as exc:
                    logger.warning(
                        "Testing run %s skipping unreadable image: %s (%s)",
                        run_id, record.file_path, exc.original,
                    )
                    _record_skip(record.file_path)
                    return record, None

            # Streaming, batched inference: preprocess a batch (parallel), score it
            # with one reconstruction pass, write rows via bulk insert + the CSV
            # incrementally, and keep only running aggregates. Nothing accumulates
            # in RAM, so this scales to hundreds of thousands of images.
            results_path = _testing_run_dir(run.id) / "reconstruction_errors.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            prep_workers = min(8, os.cpu_count() or 1)

            def _commit_image_progress(processed_inputs: int) -> None:
                nonlocal last_checkpoint_input
                run.image_count = count
                if _checkpoint_due(processed_inputs, last_checkpoint_input):
                    _store_checkpoint(
                        run,
                        kind=context.kind,
                        signature=context.signature,
                        input_count=processed_inputs,
                        result_count=count,
                        total_inputs=total,
                        score_sum=score_sum,
                        full_sum=full_sum,
                        roi_sum=roi_sum,
                        roi_count=roi_count,
                        score_min=score_min,
                        score_max=score_max,
                        skipped_count=skipped_count,
                        skipped_paths=skipped_paths,
                    )
                    last_checkpoint_input = processed_inputs
                    logger.info(
                        "Testing run %s saved checkpoint at %s inputs (%s results)",
                        run_id,
                        processed_inputs,
                        count,
                    )
                db.commit()

            with open(results_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(_CSV_HEADER)
                _write_checkpoint_prefix(db, writer, run.id, count)
                with ThreadPoolExecutor(max_workers=prep_workers) as pool:
                    for start in range(input_start, total, _INFER_BATCH):
                        if abort_event.is_set():
                            raise AbortedError()
                        end = min(total, start + _INFER_BATCH)
                        batch = records[start:end]
                        prepared = list(pool.map(_prep, batch))
                        prepared = [(record, image) for record, image in prepared if image is not None]
                        if not prepared:
                            _commit_image_progress(end)
                            continue
                        first_inference = evaluator.model is None
                        if first_inference:
                            logger.info("Testing run %s initializing model for first batch", run_id)
                        scored = evaluator.score_batch([image for _, image in prepared], roi)
                        if first_inference:
                            logger.info("Testing run %s completed first inference batch", run_id)
                        mappings = []
                        for (record, _), (full_mse, roi_mse, width, height, tile_scores, score_metadata) in zip(
                            prepared, scored
                        ):
                            position = count
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
                            writer.writerow(_csv_row(mappings[-1]))
                            count += 1
                            score_sum += score
                            full_sum += full_mse
                            score_min = score if score_min is None else min(score_min, score)
                            score_max = score if score_max is None else max(score_max, score)
                            if roi_mse is not None:
                                roi_sum += roi_mse
                                roi_count += 1
                        db.bulk_insert_mappings(models.TestingRunResult, mappings)
                        _commit_image_progress(end)
                        if (start // _INFER_BATCH) % 20 == 0:
                            rate = count / max(1e-6, time.perf_counter() - started)
                            logger.info("Testing run %s: %s/%s (%.0f img/s)", run_id, count, total, rate)

            if count == 0:
                raise ValueError(
                    f"All {total} images failed to load; see the skipped image list."
                )
            run.status = "finished"
            run.ended_at = _utcnow()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            run.image_count = count
            run.skipped_image_count = skipped_count
            run.skipped_images = sorted(set(skipped_paths))[:MAX_SKIPPED_PATHS]
            run.score_mean = score_sum / count if count else None
            run.score_min = score_min
            run.score_max = score_max
            run.full_mse_mean = full_sum / count if count else None
            run.roi_mse_mean = roi_sum / roi_count if roi_count else None
            run.results_path = str(results_path)
            run.results_size_bytes = results_path.stat().st_size
            run.checkpoint_at = None
            run.checkpoint_input_count = None
            run.checkpoint_result_count = None
            run.checkpoint_state = None
            run.restart_mode = None
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
                run.restart_mode = None
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
                run.restart_mode = None
                db.commit()
            logger.exception("Testing run %s failed", run_id)
    finally:
        db.close()
