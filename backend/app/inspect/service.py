from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import SessionLocal, data_dir
from app.preprocessing.pipeline import (
    encode_absolute_image_data_url,
    image_metadata,
    run_pipeline_array,
)
from app.schemas import InspectPreviewRequest, InspectPreviewResponse, InspectRunCreate, InspectRunRead, PreprocessingGraph
from app.training.data import enumerate_training_dataset_image_records_for_range

logger = logging.getLogger("mltrace.inspect")

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_POLL_INTERVAL_SECONDS = 2.0
_PREVIEW_FRAME_LIMIT = 30


def _utcnow() -> datetime:
    return datetime.utcnow()


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


def _resolve_selected_records(
    db: Session,
    payload: InspectPreviewRequest | InspectRunCreate | models.InspectRun,
):
    training_dataset_id = int(payload.training_dataset_id)
    preprocessing_pipeline_id = int(payload.preprocessing_pipeline_id)
    training_dataset = _load_training_dataset(db, training_dataset_id)
    if training_dataset is None:
        raise ValueError(f"Train/Test Dataset does not exist: {training_dataset_id}")
    preprocessing_pipeline = _load_preprocessing_pipeline(db, preprocessing_pipeline_id)
    if preprocessing_pipeline is None:
        raise ValueError(f"Preprocessing pipeline does not exist: {preprocessing_pipeline_id}")
    if payload.end_timestamp < payload.start_timestamp:
        raise ValueError("end_timestamp must not be before start_timestamp.")
    records = enumerate_training_dataset_image_records_for_range(
        training_dataset,
        payload.start_timestamp,
        payload.end_timestamp,
        extra_stride=max(1, int(payload.stride)),
    )
    return training_dataset, preprocessing_pipeline, records


def preview_inspect(db: Session, payload: InspectPreviewRequest) -> InspectPreviewResponse:
    training_dataset, preprocessing_pipeline, records = _resolve_selected_records(db, payload)
    if not records:
        raise ValueError("No images in selected range.")
    graph = PreprocessingGraph.model_validate(preprocessing_pipeline.graph)
    preview_frames = []
    first_image = None
    for index, record in enumerate(records[:_PREVIEW_FRAME_LIMIT]):
        image = run_pipeline_array(graph, record.file_path)
        if first_image is None:
            first_image = image
        preview_frames.append(
            {
                "index": index,
                "timestamp": record.timestamp_parsed.isoformat(),
                "image_path": record.file_path,
                "image_data_url": encode_absolute_image_data_url(image),
            }
        )
    first = records[0]
    image = first_image
    if image is None:
        raise ValueError("No images in selected range.")
    width, height, channels, dtype, value_min, value_max = image_metadata(image)
    return InspectPreviewResponse(
        training_dataset_id=training_dataset.id,
        preprocessing_pipeline_id=preprocessing_pipeline.id,
        start_timestamp=payload.start_timestamp,
        end_timestamp=payload.end_timestamp,
        stride=max(1, payload.stride),
        matching_images=len(records),
        selected_images=len(records),
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
    )


def _serialize(run: models.InspectRun) -> InspectRunRead:
    return InspectRunRead.model_validate(run)


def create_inspect_run(db: Session, payload: InspectRunCreate) -> InspectRunRead:
    if payload.content_mode != "final_preprocessed_output":
        raise ValueError("Only final_preprocessed_output is supported.")
    training_dataset, preprocessing_pipeline, records = _resolve_selected_records(db, payload)
    if not records:
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
        frame_count=len(records),
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
