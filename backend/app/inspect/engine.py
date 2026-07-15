from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import models
from app.database import SessionLocal, data_dir
from app.inspect.contrast import StreamingMovingAverage, enhance_to_uint8, to_intensity_16scale
from app.inspect.diagnostics import compute_diagnostic, write_diagnostic_artifacts
from app.preprocessing.pipeline import absolute_image_to_uint8, compile_pipeline
from app.schemas import PreprocessingGraph
from app.training.data import enumerate_training_dataset_image_records_for_range
from app.video import add_timestamp_watermark, finalize_browser_mp4

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


def _gray_to_rgb(gray_uint8: np.ndarray) -> np.ndarray:
    return np.stack([gray_uint8] * 3, axis=-1)


def _render_passthrough(
    run: models.InspectRun,
    db: Session,
    abort_event: threading.Event,
    compiled,
    records,
    frames_dir: Path,
    video_path: Path,
) -> "cv2.VideoWriter | None":
    """Stream each processed frame straight into the inspection video."""
    writer = None
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
        stamped = add_timestamp_watermark(rgb, record.timestamp_parsed)
        _write_frame(frames_dir / f"frame_{index:05d}.png", stamped)
        writer.write(cv2.cvtColor(stamped, cv2.COLOR_RGB2BGR))
        run.done_count = index + 1
        if index == 0 or (index + 1) % 10 == 0:
            db.commit()
    return writer


def _render_contrast(
    run: models.InspectRun,
    db: Session,
    abort_event: threading.Event,
    compiled,
    records,
    frames_dir: Path,
    video_path: Path,
) -> "cv2.VideoWriter | None":
    """Build a mean reference, then stream diff-enhanced frames to disk/video."""
    reference_frames = max(1, int(run.contrast_reference_frames or 1))
    shift = float(run.contrast_shift or 0.0)
    vmax = float(run.contrast_vmax or 0.0)
    ma_radius = max(0, int(run.contrast_ma_radius or 0))
    if vmax <= 0:
        raise ValueError("Contrast vmax must be greater than zero.")

    # Pass 1: only the first N frames are needed for the mean reference. Decode
    # them in parallel (chunked) and accumulate — keeps memory ~one chunk while
    # cutting the otherwise-serial first step. Progress is surfaced through the
    # run counters so the UI doesn't appear frozen at 0/N during this phase.
    ref_records = records[:reference_frames]
    reference_total = len(ref_records)
    run.frame_count = reference_total
    run.done_count = 0
    db.commit()

    reference_acc: np.ndarray | None = None
    reference_used = 0
    expected_shape: tuple[int, int] | None = None
    workers = min(8, os.cpu_count() or 1)
    phase_started = time.perf_counter()

    def _load_intensity(record):
        return record, to_intensity_16scale(compiled.run(record.file_path))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_start in range(0, reference_total, workers):
            if abort_event.is_set():
                raise AbortedError()
            chunk = ref_records[chunk_start : chunk_start + workers]
            for record, intensity in pool.map(_load_intensity, chunk):
                if expected_shape is None:
                    expected_shape = intensity.shape
                    reference_acc = np.zeros(expected_shape, dtype=np.float64)
                elif intensity.shape != expected_shape:
                    raise ValueError(
                        "Preprocessing output size changed during Inspect run: "
                        f"expected {expected_shape[1]}x{expected_shape[0]}, got "
                        f"{intensity.shape[1]}x{intensity.shape[0]} for {record.file_name}."
                    )
                reference_acc += intensity
                reference_used += 1
            run.done_count = reference_used
            db.commit()
            rate = reference_used / max(1e-6, time.perf_counter() - phase_started)
            logger.info(
                "Inspect run %s: building reference %s/%s (%.0f img/s)",
                run.id, reference_used, reference_total, rate,
            )

    if reference_acc is None or reference_used == 0:
        raise ValueError("No images available to build the contrast reference.")
    reference = (reference_acc / float(reference_used)).astype(np.float32)

    # Switch counters to the render phase so the progress bar restarts cleanly.
    run.frame_count = len(records)
    run.done_count = 0
    db.commit()

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = reference.shape[:2]
    writer = cv2.VideoWriter(str(video_path), fourcc, float(run.fps), (width, height))
    if not writer.isOpened():
        raise ValueError("Could not open MP4 video writer.")

    smoother = StreamingMovingAverage(ma_radius)
    written = 0

    def write_output(gray_frame: np.ndarray) -> None:
        nonlocal written
        rgb = _gray_to_rgb(gray_frame)
        stamped = add_timestamp_watermark(rgb, records[written].timestamp_parsed)
        _write_frame(frames_dir / f"frame_{written:05d}.png", stamped)
        writer.write(cv2.cvtColor(stamped, cv2.COLOR_RGB2BGR))
        written += 1
        run.done_count = written

    # Pass 2: stream all frames through diff -> shift -> clip -> smoothing.
    for index, record in enumerate(records):
        if abort_event.is_set():
            raise AbortedError()
        processed = compiled.run(record.file_path)
        intensity = to_intensity_16scale(processed)
        if intensity.shape != expected_shape:
            raise ValueError(
                "Preprocessing output size changed during Inspect run: "
                f"expected {expected_shape[1]}x{expected_shape[0]}, got "
                f"{intensity.shape[1]}x{intensity.shape[0]} for {record.file_name}."
            )
        gray = enhance_to_uint8(intensity, reference, shift, vmax)
        output = smoother.push(gray)
        if output is not None:
            write_output(output)
        if written == 1 or written % 10 == 0:
            db.commit()

    for output in smoother.flush():
        if abort_event.is_set():
            raise AbortedError()
        write_output(output)
        if written % 10 == 0:
            db.commit()

    run.done_count = written
    db.commit()
    return writer


def _render_diagnostic(
    run: models.InspectRun,
    db: Session,
    abort_event: threading.Event,
    compiled,
    records,
    artifact_dir: Path,
) -> None:
    """Stream temporal diagnostic scores to CSV/JSON and optional overlay video."""
    mode = run.analysis_mode
    roi = run.roi
    run.frame_count = max(0, len(records) - 1)
    run.done_count = 0
    db.commit()

    def _progress(done: int) -> None:
        run.done_count = done
        if done == 1 or done % 10 == 0 or done == run.frame_count:
            db.commit()

    try:
        result = compute_diagnostic(
            mode,
            compiled,
            records,
            run.analysis_config or {},
            roi,
            abort_event=abort_event,
            artifact_dir=artifact_dir,
            fps=int(run.fps),
            generate_video=bool(run.generate_video),
            progress_callback=_progress,
        )
    except RuntimeError as exc:
        if str(exc) == "aborted":
            raise AbortedError() from exc
        raise
    paths = write_diagnostic_artifacts(result, artifact_dir)
    run.csv_path = paths["csv_path"]
    run.summary_json_path = paths["summary_json_path"]
    run.plot_preview_path = paths["plot_preview_path"]
    if result.get("video_path"):
        run.video_path = str(result["video_path"])
        run.overlay_video_path = str(result["video_path"])
    if result.get("frames_dir"):
        run.frames_dir = str(result["frames_dir"])
    run.done_count = len(result["rows"])
    run.frame_count = len(result["rows"])
    db.commit()



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

            if run.analysis_mode in {"energy", "optical_flow"}:
                _render_diagnostic(run, db, abort_event, compiled, records, artifact_dir)
            elif run.contrast_enabled or run.analysis_mode == "contrast_enhanced":
                writer = _render_contrast(run, db, abort_event, compiled, records, frames_dir, video_path)
            else:
                writer = _render_passthrough(run, db, abort_event, compiled, records, frames_dir, video_path)

            if writer is not None:
                writer.release()
                writer = None
            if run.analysis_mode not in {"energy", "optical_flow"}:
                finalize_browser_mp4(video_path)
            run.status = "finished"
            run.ended_at = _utcnow()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            if run.analysis_mode not in {"energy", "optical_flow"}:
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
