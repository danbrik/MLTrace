"""Execution of one queued heatmap-range (video) job.

``run_heatmap_range(run_id)`` is invoked by the heatmap worker subprocess
(launched by the shared scheduler). It selects the testing run's result images
within ``[start, end]`` (stride-sampled), reconstructs them in **batches**
(GPU when available), and renders one composited pixel-error overlay PNG per
frame to disk. These frames are played back as a fast heatmap video in the
Analysis page. Progress is reported through ``done_count``/``frame_count`` so the
UI can show a "Frame X / N" counter. Lifecycle mirrors the testing engine
(running → finished/failed, SIGTERM → aborted).
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import select

from app import models
from app.database import SessionLocal, data_dir
from app.preprocessing.pipeline import absolute_image_to_uint8, run_pipeline_array
from app.schemas import PreprocessingGraph
from app.testing.service import (
    ArtifactEvaluator,
    _as_image,
    _heatmap_overlay,
    _load_testing_run_for_heatmap,
    _pixel_error_map,
    _to_nchw,
    _utcnow,
)

logger = logging.getLogger("mltrace.heatmap")

_BATCH_SIZE = 8
_COMMIT_EVERY = 2  # commit progress every N batches


class AbortedError(Exception):
    """Raised internally when an abort signal is observed mid-render."""


def _resolve_device(gpu_index: int | None) -> str:
    try:
        import torch

        if torch.cuda.is_available() and gpu_index is not None:
            return f"GPU:{gpu_index}"
    except Exception:  # noqa: BLE001 - torch optional / cpu-only is fine
        pass
    return "CPU"


def _chunks(seq: list, size: int):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _source_to_rgb_uint8(source: np.ndarray) -> np.ndarray:
    """Convert a real source image to HxWx3 using the shared absolute scale."""
    array = absolute_image_to_uint8(source)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    return array


def _write_frame_png(source: np.ndarray, error_map: np.ndarray, vmax: float | None, path: Path) -> None:
    base = Image.fromarray(_source_to_rgb_uint8(source)).convert("RGBA")
    overlay = Image.fromarray(_heatmap_overlay(error_map, vmax=vmax), mode="RGBA")
    Image.alpha_composite(base, overlay).convert("RGB").save(path, format="PNG")


def run_heatmap_range(run_id: int, abort_event: threading.Event | None = None) -> None:
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    db = SessionLocal()
    try:
        run = db.get(models.HeatmapRangeRun, run_id)
        if run is None:
            logger.error("Heatmap range run %s not found", run_id)
            return

        run.status = "running"
        run.started_at = _utcnow()
        run.device = _resolve_device(run.gpu_index)
        run.error_message = None
        run.done_count = 0
        run.frame_max_errors = None
        db.commit()

        try:
            testing_run = _load_testing_run_for_heatmap(db, run.testing_run_id)
            if testing_run is None:
                raise ValueError("Testing run no longer exists.")
            if testing_run.status != "finished":
                raise ValueError("Heatmap videos can only be rendered for finished testing runs.")
            training_run = testing_run.training_run
            if training_run is None:
                raise ValueError("Training run no longer exists.")

            results = list(
                db.scalars(
                    select(models.TestingRunResult)
                    .where(
                        models.TestingRunResult.testing_run_id == run.testing_run_id,
                        models.TestingRunResult.timestamp >= run.start_timestamp,
                        models.TestingRunResult.timestamp <= run.end_timestamp,
                    )
                    .order_by(models.TestingRunResult.position)
                )
            )
            stride = max(1, run.stride)
            selected = results[::stride]
            if not selected:
                raise ValueError("No result images fall in the selected time range.")

            run.frame_count = len(selected)
            db.commit()

            frames_dir = data_dir() / "heatmap_ranges" / str(run.id)
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir.mkdir(parents=True, exist_ok=True)
            run.frames_dir = str(frames_dir)
            db.commit()

            evaluator = ArtifactEvaluator(training_run)
            graph = PreprocessingGraph.model_validate(
                training_run.training_pipeline.preprocessing_pipeline.graph
            )
            shared = run.scale_mode == "shared"
            tmp_dir = frames_dir / "_err"
            if shared:
                tmp_dir.mkdir(exist_ok=True)

            global_vmax = 0.0
            frame_max_errors: list[float] = []
            done = 0

            # Pass 1: reconstruct (batched) + error map. per_frame renders the PNG
            # immediately; shared caches the error map + source to re-render once
            # the global ceiling is known (single reconstruction pass either way).
            for batch_index, batch in enumerate(_chunks(selected, _BATCH_SIZE)):
                if abort_event.is_set():
                    raise AbortedError()
                images = [run_pipeline_array(graph, record.image_path) for record in batch]
                sources = [
                    image if evaluator.mean_image is not None else _as_image(_to_nchw(image))
                    for image in images
                ]
                reconstructions = evaluator.reconstruct_batch(images)
                for offset, _record in enumerate(batch):
                    error_map = _pixel_error_map(sources[offset], reconstructions[offset])
                    frame_max = float(np.max(error_map)) if error_map.size else 0.0
                    frame_max_errors.append(frame_max)
                    global_vmax = max(global_vmax, frame_max)
                    frame_path = frames_dir / f"frame_{done:05d}.png"
                    if shared:
                        np.save(tmp_dir / f"{done:05d}.npy", error_map.astype(np.float16))
                        Image.fromarray(_source_to_rgb_uint8(sources[offset])).save(
                            tmp_dir / f"{done:05d}.png", format="PNG"
                        )
                    else:
                        _write_frame_png(sources[offset], error_map, None, frame_path)
                    done += 1
                if batch_index % _COMMIT_EVERY == 0:
                    run.done_count = done
                    db.commit()

            run.done_count = done
            run.global_vmax = global_vmax
            run.frame_max_errors = frame_max_errors
            db.commit()

            # Pass 2 (shared only): composite with the global ceiling. Cheap (no model).
            if shared:
                for index in range(len(selected)):
                    if abort_event.is_set():
                        raise AbortedError()
                    error_map = np.load(tmp_dir / f"{index:05d}.npy").astype(np.float64)
                    source = np.asarray(Image.open(tmp_dir / f"{index:05d}.png"))
                    _write_frame_png(source, error_map, global_vmax, frames_dir / f"frame_{index:05d}.png")
                shutil.rmtree(tmp_dir, ignore_errors=True)

            run.status = "finished"
            run.ended_at = _utcnow()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            db.commit()
            logger.info("Heatmap range run %s finished (%s frames)", run_id, len(selected))
        except AbortedError:
            db.rollback()
            run = db.get(models.HeatmapRangeRun, run_id)
            if run is not None:
                run.status = "aborted"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = "Heatmap video aborted by user."
                db.commit()
            logger.info("Heatmap range run %s aborted", run_id)
        except Exception as exc:  # noqa: BLE001 - record any failure on the run row
            db.rollback()
            run = db.get(models.HeatmapRangeRun, run_id)
            if run is not None:
                run.status = "failed"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = str(exc)
                db.commit()
            logger.exception("Heatmap range run %s failed", run_id)
    finally:
        db.close()
