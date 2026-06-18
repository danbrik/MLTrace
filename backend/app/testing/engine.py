"""Execution of one queued testing (inference) run.

``run_testing(run_id)`` is invoked by the testing worker subprocess (launched by
the shared scheduler). It loads the trained artifact via the existing
``ArtifactEvaluator``, runs every test image through the model's preprocessing
pipeline, stores per-image reconstruction errors, writes the results CSV, and
updates aggregates — mirroring the training engine's lifecycle (running →
finished/failed, SIGTERM → aborted).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from app import models
from app.database import SessionLocal
from app.preprocessing.pipeline import run_pipeline_array
from app.schemas import PreprocessingGraph
from app.testing.service import (
    ArtifactEvaluator,
    _load_training_dataset,
    _load_training_run,
    _utcnow,
    _write_results_csv,
)
from app.training.data import enumerate_training_dataset_image_records

logger = logging.getLogger("mltrace.testing")

_COMMIT_EVERY = 25


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
            result_rows: list[models.TestingRunResult] = []
            scores: list[float] = []
            full_scores: list[float] = []
            roi_scores: list[float] = []

            for position, record in enumerate(records):
                if abort_event.is_set():
                    raise AbortedError()
                image = run_pipeline_array(graph, record.file_path)
                full_mse, roi_mse, width, height, tile_scores = evaluator.score(image, roi)
                score = roi_mse if roi_mse is not None else full_mse
                scores.append(score)
                full_scores.append(full_mse)
                if roi_mse is not None:
                    roi_scores.append(roi_mse)
                row = models.TestingRunResult(
                    testing_run_id=run.id,
                    position=position,
                    image_path=record.file_path,
                    timestamp=record.timestamp_parsed,
                    score=score,
                    full_mse=full_mse,
                    roi_mse=roi_mse,
                    tile_scores=tile_scores,
                    width=width,
                    height=height,
                )
                db.add(row)
                result_rows.append(row)
                if (position + 1) % _COMMIT_EVERY == 0:
                    run.image_count = position + 1
                    db.commit()

            db.flush()
            results_path = _write_results_csv(run, result_rows)
            run.status = "finished"
            run.ended_at = _utcnow()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            run.image_count = len(result_rows)
            run.score_mean = float(np.mean(scores))
            run.score_min = float(np.min(scores))
            run.score_max = float(np.max(scores))
            run.full_mse_mean = float(np.mean(full_scores))
            run.roi_mse_mean = float(np.mean(roi_scores)) if roi_scores else None
            run.results_path = str(results_path)
            run.results_size_bytes = results_path.stat().st_size
            db.commit()
            logger.info("Testing run %s finished (%s images)", run_id, len(result_rows))
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
