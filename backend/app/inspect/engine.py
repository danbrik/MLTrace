from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import models
from app.database import SessionLocal, data_dir
from app.preprocessing.pipeline import absolute_image_to_uint8, compile_pipeline
from app.schemas import PreprocessingGraph
from app.training.data import enumerate_training_dataset_image_records_for_range

logger = logging.getLogger("mltrace.inspect")


class AbortedError(Exception):
    """Raised internally when an abort signal is observed mid-render."""


def _utcnow():
    from datetime import datetime

    return datetime.utcnow()


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    display = absolute_image_to_uint8(image)
    if display.ndim == 2:
        return np.stack([display] * 3, axis=-1)
    if display.ndim == 3 and display.shape[2] == 1:
        return np.repeat(display, 3, axis=2)
    if display.ndim == 3 and display.shape[2] == 4:
        return display[..., :3]
    if display.ndim == 3 and display.shape[2] == 3:
        return display
    raise ValueError(f"Inspect output must be 2D, 1-channel, 3-channel, or 4-channel; got shape {display.shape}.")


def _write_frame(path: Path, image_rgb: np.ndarray) -> None:
    Image.fromarray(image_rgb).save(path, format="PNG")


def run_inspect(run_id: int, abort_event: threading.Event | None = None) -> None:
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    db = SessionLocal()
    writer = None
    try:
        run = db.get(models.InspectRun, run_id)
        if run is None:
            logger.error("Inspect run %s not found", run_id)
            return

        run.status = "running"
        run.started_at = run.started_at or _utcnow()
        run.device = "CPU"
        run.error_message = None
        run.done_count = 0
        db.commit()

        try:
            training_dataset = run.training_dataset
            preprocessing_pipeline = run.preprocessing_pipeline
            records = enumerate_training_dataset_image_records_for_range(
                training_dataset,
                run.start_timestamp,
                run.end_timestamp,
                extra_stride=max(1, run.stride),
            )
            if not records:
                raise ValueError("No images in selected range.")
            run.frame_count = len(records)
            db.commit()

            graph = PreprocessingGraph.model_validate(preprocessing_pipeline.graph)
            compiled = compile_pipeline(graph)

            artifact_dir = data_dir() / "inspect_runs" / str(run.id)
            frames_dir = artifact_dir / "frames"
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir.mkdir(parents=True, exist_ok=True)
            video_path = artifact_dir / "inspect.mp4"
            if video_path.exists():
                video_path.unlink()
            run.frames_dir = str(frames_dir)
            run.video_path = str(video_path)
            db.commit()

            expected_size: tuple[int, int] | None = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            for index, record in enumerate(records):
                if abort_event.is_set():
                    raise AbortedError()
                processed = compiled.run(record.file_path)
                rgb = _to_rgb_uint8(processed)
                height, width = rgb.shape[:2]
                if expected_size is None:
                    expected_size = (width, height)
                    writer = cv2.VideoWriter(str(video_path), fourcc, float(run.fps), expected_size)
                    if not writer.isOpened():
                        raise ValueError("Could not open MP4 video writer.")
                elif expected_size != (width, height):
                    raise ValueError(
                        "Preprocessing output size changed during Inspect run: "
                        f"expected {expected_size[0]}x{expected_size[1]}, got {width}x{height} "
                        f"for {record.file_name}."
                    )
                _write_frame(frames_dir / f"frame_{index:05d}.png", rgb)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                run.done_count = index + 1
                if index == 0 or (index + 1) % 10 == 0:
                    db.commit()

            if writer is not None:
                writer.release()
                writer = None
            run.status = "finished"
            run.ended_at = _utcnow()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            run.done_count = len(records)
            db.commit()
            logger.info("Inspect run %s finished (%s frames)", run.id, len(records))
        except AbortedError:
            db.rollback()
            run = db.get(models.InspectRun, run_id)
            if run is not None:
                run.status = "aborted"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = "Inspect run aborted by user."
                db.commit()
            logger.info("Inspect run %s aborted", run_id)
        except Exception as exc:  # noqa: BLE001 - persist all worker failures
            db.rollback()
            run = db.get(models.InspectRun, run_id)
            if run is not None:
                run.status = "failed"
                run.ended_at = _utcnow()
                run.duration_seconds = round(time.perf_counter() - started, 3)
                run.error_message = str(exc)
                db.commit()
            logger.exception("Inspect run %s failed", run_id)
    finally:
        if writer is not None:
            writer.release()
        db.close()
