from __future__ import annotations

import logging
import os
import hashlib
import json
import csv
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import SessionLocal, data_dir
from app.inspect.diagnostics import compute_diagnostic, result_to_preview_response
from app.inspect.contrast import enhance_to_uint8, to_intensity_16scale
from app.preprocessing.pipeline import (
    encode_absolute_image_data_url,
    image_metadata,
    compile_pipeline,
)
from app.schemas import (
    InspectArtifactRunPage,
    InspectArtifactRunRead,
    InspectCsvColumn,
    InspectCsvData,
    InspectPreviewRequest,
    InspectPreviewResponse,
    InspectRunCreate,
    InspectRunRead,
    PreprocessingGraph,
)
from app.training.data import (
    count_folder_range_images,
    enumerate_head_records_for_range,
)
from app.video import add_timestamp_watermark, write_mp4

logger = logging.getLogger("mltrace.inspect")

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_POLL_INTERVAL_SECONDS = 2.0
_PREVIEW_FRAME_LIMIT = 30
_PREVIEW_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60


def _utcnow() -> datetime:
    return datetime.utcnow()


def _preview_cache(
    payload: InspectPreviewRequest,
    training_dataset: models.TrainingDataset,
    preprocessing_pipeline: models.PreprocessingPipeline,
) -> tuple[str, Path, str]:
    signature = {
        "request": payload.model_dump(mode="json"),
        "dataset_updated_at": str(training_dataset.updated_at),
        "pipeline_updated_at": str(preprocessing_pipeline.updated_at),
    }
    token = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    root = data_dir() / "inspect_previews"
    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for child in root.iterdir():
        try:
            if now - child.stat().st_mtime > _PREVIEW_CACHE_MAX_AGE_SECONDS:
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
        except OSError:
            pass
    artifact_dir = root / token
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return token, artifact_dir, f"/api/inspect/previews/{token}.mp4"


def _worker_database_url(database_url: str) -> str:
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return database_url
    database_path = Path(url.database).expanduser()
    if database_path.is_absolute():
        return database_url
    return url.set(database=str(database_path.resolve())).render_as_string(hide_password=False)


def _load_training_dataset(db: Session, training_dataset_id: int) -> models.TrainingDataset | None:
    return db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset_id)
        .options(
            selectinload(models.TrainingDataset.rules)
            .selectinload(models.TrainingDatasetRule.folder)
            .selectinload(models.DatasetFolder.dataset)
        )
    )


def _load_preprocessing_pipeline(db: Session, preprocessing_pipeline_id: int) -> models.PreprocessingPipeline | None:
    return db.get(models.PreprocessingPipeline, preprocessing_pipeline_id)


_PREVIEW_DECODE_WORKERS = min(8, os.cpu_count() or 1)


def _parallel_map(fn, items: list):
    """Apply ``fn`` over items in parallel (ordered results). opencv/decode work
    releases the GIL, so a thread pool speeds up preview frame decoding."""
    if len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(_PREVIEW_DECODE_WORKERS, len(items))) as pool:
        return list(pool.map(fn, items))


def preview_inspect(db: Session, payload: InspectPreviewRequest) -> InspectPreviewResponse:
    training_dataset = _load_training_dataset(db, int(payload.training_dataset_id))
    if training_dataset is None:
        raise ValueError(f"Train/Test Dataset does not exist: {payload.training_dataset_id}")
    preprocessing_pipeline = _load_preprocessing_pipeline(db, int(payload.preprocessing_pipeline_id))
    if preprocessing_pipeline is None:
        raise ValueError(f"Preprocessing pipeline does not exist: {payload.preprocessing_pipeline_id}")
    if payload.end_timestamp < payload.start_timestamp:
        raise ValueError("end_timestamp must not be before start_timestamp.")
    analysis_mode = _analysis_mode(payload)
    roi = db.get(models.RoiDefinition, payload.roi_id) if payload.roi_id is not None else None
    if payload.roi_id is not None and roi is None:
        raise ValueError(f"ROI does not exist: {payload.roi_id}")

    # Cheap total count for display; only the head of records is actually rendered.
    total_selected = _estimate_frame_count(
        training_dataset, payload.start_timestamp, payload.end_timestamp, max(1, payload.stride)
    )
    if total_selected == 0:
        raise ValueError("No images in selected range.")

    limit = (
        max(max(1, int(payload.contrast_reference_frames)), _PREVIEW_FRAME_LIMIT)
        if payload.contrast_enabled
        else _PREVIEW_FRAME_LIMIT
    )
    records = enumerate_head_records_for_range(
        training_dataset,
        payload.start_timestamp,
        payload.end_timestamp,
        extra_stride=max(1, payload.stride),
        limit=limit,
    )
    if not records:
        raise ValueError("No images in selected range.")

    graph = PreprocessingGraph.model_validate(preprocessing_pipeline.graph)
    compiled = compile_pipeline(graph)
    token, preview_dir, preview_video_url = _preview_cache(
        payload, training_dataset, preprocessing_pipeline
    )
    preview_video_path = preview_dir / "inspect.mp4"

    if analysis_mode in {"energy", "optical_flow"}:
        result = compute_diagnostic(
            analysis_mode,
            compiled,
            records,
            payload.analysis_config,
            roi,
            preview_limit=_PREVIEW_FRAME_LIMIT,
            artifact_dir=None if preview_video_path.exists() else preview_dir,
            generate_video=not preview_video_path.exists(),
        )
        preview_data = result_to_preview_response(result)
        first = records[0]
        example = result["example_rgb"]
        width, height, channels, dtype, value_min, value_max = image_metadata(example)
        return InspectPreviewResponse(
            training_dataset_id=training_dataset.id,
            preprocessing_pipeline_id=preprocessing_pipeline.id,
            start_timestamp=payload.start_timestamp,
            end_timestamp=payload.end_timestamp,
            stride=max(1, payload.stride),
            matching_images=total_selected,
            selected_images=total_selected,
            first_image_path=first.file_path,
            first_timestamp=first.timestamp_parsed,
            width=width,
            height=height,
            channels=channels,
            dtype=dtype,
            value_min=value_min,
            value_max=value_max,
            preview_frame_count=len(result["rows"]),
            preview_frames=[],
            analysis_mode=analysis_mode,
            analysis_config=payload.analysis_config,
            roi_id=roi.id if roi else None,
            roi_name=roi.name if roi else None,
            generate_video=bool(payload.generate_video),
            image_data_url=preview_data["image_data_url"],
            plot_image_data_url=preview_data["plot_image_data_url"],
            preview_video_url=preview_video_url,
            diagnostic_columns=preview_data["diagnostic_columns"],
            diagnostic_series=preview_data["diagnostic_series"],
        )

    if analysis_mode == "contrast_enhanced":
        return _preview_contrast(
            payload,
            training_dataset,
            preprocessing_pipeline,
            records,
            compiled,
            total_selected,
            preview_video_path,
            preview_video_url,
        )

    head = records[:_PREVIEW_FRAME_LIMIT]
    images = _parallel_map(lambda record: compiled.run(record.file_path), head)
    preview_frames = [
        {
            "index": index,
            "timestamp": record.timestamp_parsed.isoformat(),
            "image_path": record.file_path,
            "image_data_url": encode_absolute_image_data_url(image),
        }
        for index, (record, image) in enumerate(zip(head, images))
    ]
    first = records[0]
    image = images[0]
    if not preview_video_path.exists():
        video_frames = [
            add_timestamp_watermark(_preview_rgb(value), record.timestamp_parsed)
            for record, value in zip(head, images)
        ]
        write_mp4(preview_video_path, video_frames, payload.fps)
    width, height, channels, dtype, value_min, value_max = image_metadata(image)
    return InspectPreviewResponse(
        training_dataset_id=training_dataset.id,
        preprocessing_pipeline_id=preprocessing_pipeline.id,
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        stride=max(1, payload.stride),
        matching_images=total_selected,
        selected_images=total_selected,
        first_image_path=first.file_path,
        first_timestamp=first.timestamp_parsed,
        width=width,
        height=height,
        channels=channels,
        dtype=dtype,
        value_min=value_min,
        value_max=value_max,
        image_data_url=encode_absolute_image_data_url(image),
        preview_frame_count=len(preview_frames),
        preview_frames=preview_frames,
        analysis_mode=analysis_mode,
        analysis_config=payload.analysis_config,
        roi_id=roi.id if roi else None,
        roi_name=roi.name if roi else None,
        generate_video=bool(payload.generate_video),
        preview_video_url=preview_video_url,
    )


def _preview_rgb(image: np.ndarray) -> np.ndarray:
    from app.preprocessing.pipeline import absolute_image_to_uint8

    value = absolute_image_to_uint8(image)
    if value.ndim == 2:
        return np.stack([value] * 3, axis=-1)
    if value.ndim == 3 and value.shape[2] == 1:
        return np.repeat(value, 3, axis=2)
    if value.ndim == 3 and value.shape[2] == 4:
        return value[..., :3]
    return value


def _analysis_mode(payload: InspectPreviewRequest | InspectRunCreate) -> str:
    if payload.contrast_enabled and payload.analysis_mode == "preprocessed_video":
        return "contrast_enhanced"
    return payload.analysis_mode


def _preview_contrast(
    payload: InspectPreviewRequest,
    training_dataset: models.TrainingDataset,
    preprocessing_pipeline: models.PreprocessingPipeline,
    records,
    compiled,
    total_selected: int,
    preview_video_path: Path,
    preview_video_url: str,
) -> InspectPreviewResponse:
    reference_frames = max(1, int(payload.contrast_reference_frames))
    shift = float(payload.contrast_shift)
    vmax = float(payload.contrast_vmax)
    if vmax <= 0:
        raise ValueError("Contrast vmax must be greater than zero.")

    reference_used = min(reference_frames, len(records))
    preview_count = min(_PREVIEW_FRAME_LIMIT, len(records))
    needed = max(reference_used, preview_count)

    # Decode the needed head frames in parallel, then accumulate the reference serially.
    all_intensities = _parallel_map(
        lambda record: to_intensity_16scale(compiled.run(record.file_path)),
        records[:needed],
    )
    expected_shape = all_intensities[0].shape
    reference_acc = np.zeros(expected_shape, dtype=np.float64)
    for index, intensity in enumerate(all_intensities):
        if intensity.shape != expected_shape:
            raise ValueError(
                "Preprocessing output size changed within the preview window: "
                f"expected {expected_shape[1]}x{expected_shape[0]}, got "
                f"{intensity.shape[1]}x{intensity.shape[0]} for {records[index].file_name}."
            )
        if index < reference_used:
            reference_acc += intensity

    intensities = all_intensities[:preview_count]
    reference = (reference_acc / float(reference_used)).astype(np.float32)

    preview_frames = []
    diff_min = float("inf")
    diff_max = float("-inf")
    first_display: np.ndarray | None = None
    video_frames: list[np.ndarray] = []
    for index in range(preview_count):
        record = records[index]
        diff = intensities[index] - reference
        diff_min = min(diff_min, float(diff.min()))
        diff_max = max(diff_max, float(diff.max()))
        display = enhance_to_uint8(intensities[index], reference, shift, vmax)
        if first_display is None:
            first_display = display
        video_frames.append(
            add_timestamp_watermark(np.stack([display] * 3, axis=-1), record.timestamp_parsed)
        )
        preview_frames.append(
            {
                "index": index,
                "timestamp": record.timestamp_parsed.isoformat(),
                "image_path": record.file_path,
                "image_data_url": encode_absolute_image_data_url(display),
            }
        )

    first = records[0]
    assert first_display is not None
    if not preview_video_path.exists():
        write_mp4(preview_video_path, video_frames, payload.fps)
    height, width = first_display.shape[:2]
    return InspectPreviewResponse(
        training_dataset_id=training_dataset.id,
        preprocessing_pipeline_id=preprocessing_pipeline.id,
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        stride=max(1, payload.stride),
        matching_images=total_selected,
        selected_images=total_selected,
        first_image_path=first.file_path,
        first_timestamp=first.timestamp_parsed,
        width=width,
        height=height,
        channels=1,
        dtype="uint8",
        value_min=float(first_display.min()),
        value_max=float(first_display.max()),
        image_data_url=encode_absolute_image_data_url(first_display),
        preview_frame_count=len(preview_frames),
        preview_frames=preview_frames,
        contrast_enabled=True,
        contrast_reference_frames_used=reference_used,
        contrast_diff_min=None if diff_min == float("inf") else diff_min,
        contrast_diff_max=None if diff_max == float("-inf") else diff_max,
        preview_video_url=preview_video_url,
    )


def _serialize(run: models.InspectRun) -> InspectRunRead:
    return InspectRunRead.model_validate(run)


def _estimate_frame_count(
    training_dataset: models.TrainingDataset,
    start_timestamp: datetime,
    end_timestamp: datetime,
    extra_stride: int,
) -> int:
    """Cheap selected-image estimate without building record objects.

    Sums each rule's selected count (folder ``image_count`` fast path, else the
    cached timestamp index), then applies the extra stride. Cross-rule overlaps
    may slightly overcount; the worker computes the exact value when it runs."""
    selected = 0
    for rule in training_dataset.rules:
        if rule.folder is None:
            continue
        clipped_start = max(rule.start_timestamp, start_timestamp)
        clipped_end = min(rule.end_timestamp, end_timestamp)
        if clipped_end < clipped_start:
            continue
        selected += count_folder_range_images(
            rule.folder, clipped_start, clipped_end, rule.stride
        ).selected_images
    stride = max(1, extra_stride)
    return (selected + stride - 1) // stride if selected else 0


def create_inspect_run(db: Session, payload: InspectRunCreate) -> InspectRunRead:
    if payload.content_mode != "final_preprocessed_output":
        raise ValueError("Only final_preprocessed_output is supported.")
    analysis_mode = _analysis_mode(payload)
    if analysis_mode == "contrast_enhanced" and payload.contrast_vmax <= 0:
        raise ValueError("Contrast vmax must be greater than zero.")
    if payload.end_timestamp < payload.start_timestamp:
        raise ValueError("end_timestamp must not be before start_timestamp.")
    if analysis_mode not in {"preprocessed_video", "contrast_enhanced", "energy", "optical_flow"}:
        raise ValueError(f"Unsupported Inspect analysis mode: {analysis_mode}.")
    training_dataset = _load_training_dataset(db, int(payload.training_dataset_id))
    if training_dataset is None:
        raise ValueError(f"Train/Test Dataset does not exist: {payload.training_dataset_id}")
    preprocessing_pipeline = _load_preprocessing_pipeline(db, int(payload.preprocessing_pipeline_id))
    if preprocessing_pipeline is None:
        raise ValueError(f"Preprocessing pipeline does not exist: {payload.preprocessing_pipeline_id}")
    roi = db.get(models.RoiDefinition, payload.roi_id) if payload.roi_id is not None else None
    if payload.roi_id is not None and roi is None:
        raise ValueError(f"ROI does not exist: {payload.roi_id}")

    # Enqueue must stay constant-time. Even the timestamp-index based counter can
    # be slow for large folders, so only validate that the requested range
    # overlaps at least one saved dataset rule. The worker computes the exact
    # frame count after the run is already visible in Inspect runs.
    has_overlapping_rule = any(
        rule.folder is not None
        and max(rule.start_timestamp, payload.start_timestamp)
        <= min(rule.end_timestamp, payload.end_timestamp)
        for rule in training_dataset.rules
    )
    if not has_overlapping_rule:
        raise ValueError("No images in selected range.")
    run = models.InspectRun(
        training_dataset_id=training_dataset.id,
        preprocessing_pipeline_id=preprocessing_pipeline.id,
        status="queued",
        enqueued_at=_utcnow(),
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        stride=max(1, payload.stride),
        fps=max(1, min(60, int(payload.fps))),
        content_mode=payload.content_mode,
        analysis_mode=analysis_mode,
        analysis_config=payload.analysis_config or {},
        roi_id=roi.id if roi else None,
        generate_video=bool(payload.generate_video),
        contrast_enabled=analysis_mode == "contrast_enhanced",
        contrast_reference_frames=max(1, int(payload.contrast_reference_frames)),
        contrast_shift=float(payload.contrast_shift),
        contrast_vmax=float(payload.contrast_vmax),
        contrast_ma_radius=max(0, int(payload.contrast_ma_radius)),
        frame_count=None,
        done_count=0,
        device="CPU",
        training_dataset_name=training_dataset.name,
        preprocessing_pipeline_name=preprocessing_pipeline.name,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    inspect_queue.wake()
    return _serialize(run)


def list_inspect_runs(db: Session) -> list[InspectRunRead]:
    rows = db.scalars(select(models.InspectRun).order_by(models.InspectRun.created_at.desc())).all()
    return [_serialize(row) for row in rows]


def get_inspect_run(db: Session, run_id: int) -> InspectRunRead | None:
    run = db.get(models.InspectRun, run_id)
    return _serialize(run) if run is not None else None


def abort_inspect_run(db: Session, run_id: int) -> InspectRunRead | None:
    run = db.get(models.InspectRun, run_id)
    if run is None:
        return None
    if run.status == "queued":
        run.status = "aborted"
        run.ended_at = _utcnow()
        run.error_message = "Aborted before it started."
        db.commit()
        db.refresh(run)
    elif run.status == "running":
        inspect_queue.request_abort(run.id, run.pid)
    else:
        raise ValueError("Only queued or running inspect runs can be aborted.")
    return _serialize(run)


def delete_inspect_run(db: Session, run_id: int) -> bool:
    run = db.get(models.InspectRun, run_id)
    if run is None:
        return False
    if run.status == "running":
        raise ValueError("Abort the inspect run before removing it.")
    shutil.rmtree(data_dir() / "inspect_runs" / str(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return True


def read_inspect_log(db: Session, run_id: int, max_lines: int = 400) -> str | None:
    run = db.get(models.InspectRun, run_id)
    if run is None:
        return None
    if not run.log_path:
        return ""
    try:
        with open(run.log_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return ""
    return "".join(lines[-max_lines:])


def inspect_frame_path(db: Session, run_id: int, index: int) -> Path | None:
    run = db.get(models.InspectRun, run_id)
    if run is None or not run.frames_dir:
        return None
    path = Path(run.frames_dir) / f"frame_{index:05d}.png"
    return path if path.exists() else None


def inspect_video_path(db: Session, run_id: int) -> Path | None:
    run = db.get(models.InspectRun, run_id)
    if run is None or not run.video_path:
        return None
    path = Path(run.video_path)
    return path if path.exists() else None


def inspect_csv_path(db: Session, run_id: int) -> Path | None:
    run = db.get(models.InspectRun, run_id)
    if run is None or not run.csv_path:
        return None
    path = Path(run.csv_path)
    return path if path.exists() else None


def inspect_summary_path(db: Session, run_id: int) -> Path | None:
    run = db.get(models.InspectRun, run_id)
    if run is None or not run.summary_json_path:
        return None
    path = Path(run.summary_json_path)
    return path if path.exists() else None


def inspect_plot_preview_path(db: Session, run_id: int) -> Path | None:
    run = db.get(models.InspectRun, run_id)
    if run is None or not run.plot_preview_path:
        return None
    path = Path(run.plot_preview_path)
    return path if path.exists() else None


def inspect_preview_video_path(token: str) -> Path | None:
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        return None
    path = data_dir() / "inspect_previews" / token / "inspect.mp4"
    return path if path.exists() else None


def _artifact_from_inspect(run: models.InspectRun) -> InspectArtifactRunRead:
    return InspectArtifactRunRead(
        kind="inspect",
        id=run.id,
        mode=run.analysis_mode,
        status=run.status,
        error_message=run.error_message,
        training_dataset_id=run.training_dataset_id,
        training_dataset_name=run.training_dataset_name,
        preprocessing_pipeline_id=run.preprocessing_pipeline_id,
        preprocessing_pipeline_name=run.preprocessing_pipeline_name,
        start_timestamp=run.start_timestamp,
        end_timestamp=run.end_timestamp,
        stride=run.stride,
        fps=run.fps,
        frame_count=run.frame_count,
        done_count=run.done_count,
        has_video=bool(run.video_path and Path(run.video_path).exists()),
        has_csv=bool(run.csv_path and Path(run.csv_path).exists()),
        has_summary=bool(run.summary_json_path and Path(run.summary_json_path).exists()),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _artifact_from_heatmap(run: models.HeatmapRangeRun) -> InspectArtifactRunRead | None:
    testing_run = run.testing_run
    training_run = testing_run.training_run if testing_run else None
    training_pipeline = training_run.training_pipeline if training_run else None
    preprocessing = training_pipeline.preprocessing_pipeline if training_pipeline else None
    if testing_run is None or preprocessing is None:
        return None
    return InspectArtifactRunRead(
        kind="heatmap",
        id=run.id,
        mode="heatmap",
        status=run.status,
        error_message=run.error_message,
        training_dataset_id=testing_run.training_dataset_id,
        training_dataset_name=testing_run.training_dataset_name,
        preprocessing_pipeline_id=preprocessing.id,
        preprocessing_pipeline_name=testing_run.preprocessing_pipeline_name,
        start_timestamp=run.start_timestamp,
        end_timestamp=run.end_timestamp,
        stride=run.stride,
        fps=run.fps,
        frame_count=run.frame_count,
        done_count=run.done_count,
        has_video=bool(run.video_path and Path(run.video_path).exists()),
        has_csv=False,
        has_summary=False,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def list_inspect_artifacts(
    db: Session,
    *,
    page: int = 1,
    training_dataset_id: int | None = None,
    preprocessing_pipeline_id: int | None = None,
    mode: str | None = None,
    status: str | None = None,
) -> InspectArtifactRunPage:
    inspect_rows = db.scalars(select(models.InspectRun)).all()
    heatmap_rows = db.scalars(
        select(models.HeatmapRangeRun).options(
            selectinload(models.HeatmapRangeRun.testing_run)
            .selectinload(models.TestingRun.training_run)
            .selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.preprocessing_pipeline)
        )
    ).all()
    items = [_artifact_from_inspect(row) for row in inspect_rows]
    items.extend(
        item for row in heatmap_rows if (item := _artifact_from_heatmap(row)) is not None
    )
    active_total = sum(item.status in {"queued", "running"} for item in items)
    if training_dataset_id is not None:
        items = [item for item in items if item.training_dataset_id == training_dataset_id]
    if preprocessing_pipeline_id is not None:
        items = [item for item in items if item.preprocessing_pipeline_id == preprocessing_pipeline_id]
    if mode:
        items = [item for item in items if item.mode == mode]
    if status:
        items = [item for item in items if item.status == status]
    items.sort(key=lambda item: (item.created_at, item.id), reverse=True)
    page_size = 15
    total = len(items)
    pages = max(1, math.ceil(total / page_size))
    page = min(max(1, page), pages)
    start = (page - 1) * page_size
    return InspectArtifactRunPage(
        items=items[start : start + page_size],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
        active_total=active_total,
    )


def read_inspect_csv_data(db: Session, run_id: int) -> InspectCsvData | None:
    path = inspect_csv_path(db, run_id)
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    names = list(raw_rows[0].keys()) if raw_rows else []
    columns: list[InspectCsvColumn] = []
    kinds: dict[str, str] = {}
    for name in names:
        values = [row.get(name, "").strip() for row in raw_rows if row.get(name, "").strip()]
        kind = "text"
        if values:
            try:
                for value in values:
                    float(value)
                kind = "number"
            except ValueError:
                try:
                    for value in values:
                        datetime.fromisoformat(value.replace("Z", "+00:00"))
                    kind = "datetime"
                except ValueError:
                    pass
        kinds[name] = kind
        columns.append(InspectCsvColumn(name=name, kind=kind))
    rows: list[dict] = []
    for raw in raw_rows:
        row: dict = {}
        for name in names:
            value = raw.get(name, "").strip()
            row[name] = None if not value else float(value) if kinds[name] == "number" else value
        rows.append(row)
    return InspectCsvData(columns=columns, rows=rows)


class InspectQueue:
    """Small CPU-only queue for inspect video workers.

    It intentionally does not use GPU scheduler slots because Inspect is a data
    visualization path and should not delay training/inference dispatch.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._run_id: int | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mltrace-inspect-queue", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def wake(self) -> None:
        self._wake.set()

    def request_abort(self, run_id: int, pid: int | None) -> None:
        with self._lock:
            proc = self._process if self._run_id == run_id else None
        target_pid = proc.pid if proc is not None else pid
        if target_pid is None:
            return
        try:
            os.killpg(target_pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(target_pid, signal.SIGTERM)
            except OSError:
                return

    def _loop(self) -> None:
        self._reconcile_startup()
        while not self._stop.is_set():
            self._tick()
            self._wake.wait(timeout=_POLL_INTERVAL_SECONDS)
            self._wake.clear()

    def _reconcile_startup(self) -> None:
        db = SessionLocal()
        try:
            for run in db.scalars(select(models.InspectRun).where(models.InspectRun.status == "running")).all():
                run.status = "failed"
                run.ended_at = _utcnow()
                run.error_message = run.error_message or "Inspect worker was interrupted by API restart."
            db.commit()
        finally:
            db.close()

    def _tick(self) -> None:
        db = SessionLocal()
        try:
            with self._lock:
                proc = self._process
                run_id = self._run_id
            if proc is not None and proc.poll() is not None:
                with self._lock:
                    self._process = None
                    self._run_id = None
                run = db.get(models.InspectRun, run_id)
                if run is not None and run.status == "running":
                    run.status = "failed"
                    run.ended_at = _utcnow()
                    run.error_message = run.error_message or "Worker exited without reporting a result."
                    db.commit()

            with self._lock:
                busy = self._process is not None
            if busy:
                return

            run = db.scalar(
                select(models.InspectRun)
                .where(models.InspectRun.status == "queued")
                .order_by(models.InspectRun.enqueued_at.asc(), models.InspectRun.id.asc())
            )
            if run is None:
                return
            self._launch(db, run)
        finally:
            db.close()

    def _launch(self, db: Session, run: models.InspectRun) -> None:
        from app.config import get_settings

        artifact_dir = data_dir() / "inspect_runs" / str(run.id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_dir / "worker.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["DATABASE_URL"] = _worker_database_url(get_settings().database_url)

        with open(log_path, "a", encoding="utf-8") as parent_log:
            parent_log.write(f"{_utcnow().isoformat()} inspect queue: launching run {run.id} on CPU\n")
            parent_log.flush()

        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - child owns fd
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "app.inspect.worker", str(run.id)],
                cwd=str(_BACKEND_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()

        run.status = "running"
        run.started_at = _utcnow()
        run.device = "CPU"
        run.pid = proc.pid
        run.log_path = str(log_path)
        run.error_message = None
        db.commit()

        with self._lock:
            self._process = proc
            self._run_id = run.id
        logger.info("Launched inspect run %s on CPU (pid %s)", run.id, proc.pid)


inspect_queue = InspectQueue()
