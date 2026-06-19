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

logger = logging.getLogger("mltrace.testing")

# Images per (GPU-)batched reconstruction. Bounded RAM: only this many images
# are decoded/preprocessed and held at once.
_INFER_BATCH = 16

_CSV_HEADER = [
    "position", "timestamp", "image_path", "score", "full_mse",
    "roi_mse", "tile_scores_json", "width", "height",
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
            records = enumerate_training_dataset_image_records(training_dataset)
            if not records:
                raise ValueError("Train/test dataset produced no images.")
            run.expected_image_count = len(records)
            db.commit()

            evaluator = ArtifactEvaluator(training_run)

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
                        for offset, (record, (full_mse, roi_mse, width, height, tile_scores)) in enumerate(
                            zip(batch, scored)
                        ):
                            position = start + offset
                            score = roi_mse if roi_mse is not None else full_mse
                            mappings.append({
                                "testing_run_id": run.id,
                                "position": position,
                                "image_path": record.file_path,
                                "timestamp": record.timestamp_parsed,
                                "score": score,
                                "full_mse": full_mse,
                                "roi_mse": roi_mse,
                                "tile_scores": tile_scores,
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
