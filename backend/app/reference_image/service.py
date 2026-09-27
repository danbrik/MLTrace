from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import threading
import time

import numpy as np
from PIL import Image
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app import models
from app.database import data_dir
from app.preprocessing.pipeline import compile_pipeline, absolute_image_to_uint8
from app.schemas import PreprocessingGraph
from app.training.data import enumerate_training_dataset_image_records
from app.training.scheduler import next_queue_rank, scheduler
from app.reference_image.engine import select_records, grayscale, render_difference
from app.reference_image.schemas import ReferenceImageConfig, ReferenceImageRunRead
from app.video import add_timestamp_watermark, write_mp4

logger = logging.getLogger(__name__)


class AbortedError(Exception):
    pass


def artifact_dir(run_id: int) -> Path:
    return data_dir() / "reference_image_runs" / str(run_id)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare(db: Session, config: ReferenceImageConfig):
    dataset = db.get(models.TrainingDataset, config.training_dataset_id)
    pipeline = db.get(models.PreprocessingPipeline, config.preprocessing_pipeline_id)
    if dataset is None or pipeline is None:
        raise ValueError("Datensatz oder Preprocessing-Pipeline nicht gefunden.")
    if not dataset.rules or any(rule.folder is None for rule in dataset.rules):
        raise ValueError("Der Datensatz enthält keine gültige vollständige Regelauswahl.")
    first = min(rule.start_timestamp for rule in dataset.rules)
    last = max(rule.end_timestamp for rule in dataset.rules)
    for interval in (config.reference, config.anomaly):
        if interval.start < first or interval.end > last:
            raise ValueError("Die Zeiträume müssen innerhalb der Datensatzgrenzen liegen.")
    compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
    records = enumerate_training_dataset_image_records(dataset)
    samples, preview = select_records(records, config)
    return dataset, pipeline, samples, preview


def preview(db: Session, config: ReferenceImageConfig):
    _, pipeline, samples, result = prepare(db, config)
    compiled = compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
    shape = None
    for role in ("reference", "anomaly"):
        if samples[role]:
            output = grayscale(compiled.run(samples[role][0]["file_path"]), shape)
            shape = output.shape
    return result


def enqueue(db: Session, config: ReferenceImageConfig, *, wake_scheduler: bool = True):
    dataset, pipeline, samples, selection = prepare(db, config)
    if selection.errors:
        raise ValueError(" ".join(selection.errors))
    # Freeze the selected file identities at enqueue time, not at worker dispatch.
    for sample in [*samples["reference"], *samples["anomaly"]]:
        info = Path(sample["file_path"]).stat()
        sample["size_bytes"] = info.st_size
        sample["mtime_ns"] = info.st_mtime_ns
    snapshot = {"id": dataset.id, "name": dataset.name, "rules": [{
        "id": rule.id, "start": rule.start_timestamp.isoformat(), "end": rule.end_timestamp.isoformat(),
        "stride": rule.stride, "folder_id": rule.folder_id, "folder": rule.folder.relative_path,
        "root_path": rule.folder.dataset.root_path, "timestamp_regex": rule.folder.dataset.timestamp_regex,
        "timestamp_format": rule.folder.dataset.timestamp_format,
    } for rule in dataset.rules]}
    run = models.ReferenceImageRun(
        training_dataset_id=dataset.id, training_dataset_name=dataset.name,
        config=config.model_dump(mode="json"), dataset_snapshot=snapshot,
        pipeline_snapshot={"id": pipeline.id, "name": pipeline.name, "graph": deepcopy(pipeline.graph)},
        status="queued", current_step="queued", enqueued_at=models.utc_now(), queue_rank=next_queue_rank(db),
    )
    db.add(run)
    directory = None
    try:
        db.flush()
        directory = artifact_dir(run.id)
        write_json(directory / "manifest.json", {"samples": samples, "selection": selection.model_dump()})
        db.commit()
    except Exception:
        db.rollback()
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    db.refresh(run)
    if wake_scheduler:
        scheduler.wake()
    return ReferenceImageRunRead.model_validate(run)


def list_runs(db: Session):
    return [ReferenceImageRunRead.model_validate(run) for run in db.scalars(
        select(models.ReferenceImageRun).order_by(models.ReferenceImageRun.id.desc()))]


def get_run(db: Session, run_id: int):
    run = db.get(models.ReferenceImageRun, run_id)
    return ReferenceImageRunRead.model_validate(run) if run else None


def abort_run(db: Session, run_id: int):
    run = db.get(models.ReferenceImageRun, run_id)
    if run is None:
        return None
    if run.status not in {"queued", "running"}:
        raise ValueError("Nur wartende oder laufende Analysen können abgebrochen werden.")
    # A persisted flag also covers cancellation racing with worker startup.
    db.execute(update(models.ReferenceImageRun).where(
        models.ReferenceImageRun.id == run_id, models.ReferenceImageRun.status.in_(["queued", "running"])
    ).values(cancel_requested=True))
    db.execute(update(models.ReferenceImageRun).where(
        models.ReferenceImageRun.id == run_id, models.ReferenceImageRun.status == "queued"
    ).values(status="aborted", current_step="aborted", ended_at=models.utc_now()))
    db.commit()
    db.refresh(run)
    if run.status == "running" and run.pid is not None:
        scheduler.request_abort("reference_image", run.id, run.pid)
    return ReferenceImageRunRead.model_validate(run)


def delete_run(db: Session, run_id: int):
    run = db.get(models.ReferenceImageRun, run_id)
    if run is None:
        return False
    if run.status in {"running", "queued"}:
        raise ValueError("Analyse vor dem Löschen abbrechen und auf das Ende warten.")
    db.delete(run)
    db.commit()
    shutil.rmtree(artifact_dir(run_id), ignore_errors=True)
    return True


def artifact_path(db: Session, run_id: int, name: str):
    run = db.get(models.ReferenceImageRun, run_id)
    if run is None or run.status != "finished":
        return None
    allowed = name in {"reference.png", "video.mp4", "results.json"}
    if name.startswith("frame_") and name.endswith(".png"):
        number = name[6:-4]
        allowed = 1 <= len(number) <= 12 and number.isascii() and number.isdigit() and name == f"frame_{int(number):06d}.png" and int(number) < (run.result or {}).get("frame_count", 0)
    if not allowed:
        return None
    path = artifact_dir(run_id) / name
    return path if path.is_file() else None


def results(db: Session, run_id: int):
    path = artifact_path(db, run_id, "results.json")
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def read_log(db: Session, run_id: int):
    if db.get(models.ReferenceImageRun, run_id) is None:
        return None
    path = artifact_dir(run_id) / "worker.log"
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=400))


def run_scheduled(run_id: int, abort_event: threading.Event | None = None):
    from app.database import SessionLocal
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    db = SessionLocal()
    stopped = threading.Event()
    heartbeat_thread = None
    try:
        run = db.get(models.ReferenceImageRun, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return
        if run.cancel_requested or abort_event.is_set():
            raise AbortedError()
        run.status = "running"
        run.started_at = run.started_at or models.utc_now()
        run.device = run.device or "CPU"
        db.commit()
        factory = sessionmaker(bind=db.get_bind())

        def heartbeat():
            while not stopped.wait(2):
                try:
                    with factory() as session:
                        current = session.get(models.ReferenceImageRun, run_id)
                        if current and current.status == "running":
                            if current.cancel_requested:
                                abort_event.set()
                            current.heartbeat_at = models.utc_now()
                            session.commit()
                except Exception:
                    logger.warning("Could not update analysis heartbeat", exc_info=True)

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        def report(step, done, total):
            db.refresh(run)
            if abort_event.is_set() or run.cancel_requested:
                raise AbortedError()
            if run.current_step != step:
                logger.info("Analysis phase: %s", step)
            run.current_step = step
            run.processed_images = done
            run.total_images = total
            run.heartbeat_at = models.utc_now()
            db.commit()

        result = calculate(run, report, abort_event)
        report("finished", result["frame_count"], result["frame_count"])
        db.execute(update(models.ReferenceImageRun).where(
            models.ReferenceImageRun.id == run_id, models.ReferenceImageRun.cancel_requested.is_(False)
        ).values(status="finished", result=result, ended_at=models.utc_now(),
                 duration_seconds=round(time.perf_counter() - started, 3)))
        db.commit()
        db.refresh(run)
        if run.cancel_requested:
            raise AbortedError()
    except Exception as exc:
        db.rollback()
        run = db.get(models.ReferenceImageRun, run_id)
        if run:
            aborted = isinstance(exc, AbortedError) or abort_event.is_set() or run.cancel_requested
            run.status = run.current_step = "aborted" if aborted else "failed"
            run.error_message = "Analyse abgebrochen." if aborted else str(exc)
            run.ended_at = models.utc_now()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            db.commit()
            if not aborted:
                logger.exception("Reference image analysis failed")
    finally:
        stopped.set()
        if heartbeat_thread:
            heartbeat_thread.join(timeout=3)
        db.close()


def calculate(run, report, abort_event):
    config = ReferenceImageConfig.model_validate(run.config)
    directory = artifact_dir(run.id)
    temporary = directory / "differences"
    temporary.mkdir(parents=True, exist_ok=True)
    samples = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["samples"]
    pipeline = compile_pipeline(PreprocessingGraph.model_validate(run.pipeline_snapshot["graph"]))
    shape = None
    source_dtype = None

    def load(sample):
        nonlocal shape, source_dtype
        path = Path(sample["file_path"])
        before = path.stat()
        if before.st_size != sample["size_bytes"] or before.st_mtime_ns != sample["mtime_ns"]:
            raise ValueError(f"Quelldatei seit Erstellung verändert: {path.name}")
        raw = pipeline.run(str(path))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Quelldatei während des Lesens verändert: {path.name}")
        array = grayscale(raw, shape)
        shape = array.shape
        source_dtype = source_dtype or raw.dtype
        return array

    try:
        reference = None
        count = len(samples["reference"])
        report("reference", 0, count)
        for index, sample in enumerate(samples["reference"]):
            report("reference", index, count)
            array = load(sample)
            if reference is None:
                reference = np.zeros(array.shape, dtype=np.float64)
            reference += array / count
        if reference is None or not np.isfinite(reference).all():
            raise ValueError("Kein gültiges Referenzbild berechnet.")
        np.save(directory / "reference.npy", reference)
        # Preserve the input intensity range for display, without image-wise stretching.
        if np.issubdtype(source_dtype, np.integer):
            info = np.iinfo(source_dtype)
            display = np.rint(np.clip((reference - info.min) / float(info.max - info.min), 0, 1) * 255).astype(np.uint8)
        else:
            display = absolute_image_to_uint8(reference)
        Image.fromarray(display).save(directory / "reference.png")

        frames = []
        maximum = 0.0
        count = len(samples["anomaly"])
        if count == 0:
            raise ValueError("Keine Anomaliebilder ausgewählt.")
        for index, sample in enumerate(samples["anomaly"]):
            report("difference", index, count)
            difference = load(sample) - reference
            if not np.isfinite(difference).all():
                raise ValueError("Die Differenz enthält nicht endliche Pixelwerte.")
            maximum = max(maximum, float(np.abs(difference).max()))
            distance = float(np.mean(np.abs(difference)))
            if not np.isfinite(distance):
                raise ValueError("Der Bildabstand ist nicht endlich.")
            np.save(temporary / f"{index}.npy", difference)
            frames.append({"index": index, "timestamp": sample["timestamp"], "distance": distance})
        limit = maximum if config.scale_mode == "auto" else config.scale_limit

        def rendered_frames():
            for frame in frames:
                index = frame["index"]
                report("rendering", index, count)
                difference = np.load(temporary / f"{index}.npy", allow_pickle=False)
                gray = render_difference(difference, limit)
                rgb = np.repeat(gray[:, :, None], 3, axis=2)
                # Pad before saving so PNG and MP4 use the same even-sized canvas.
                rgb = np.pad(rgb, ((0, rgb.shape[0] % 2), (0, rgb.shape[1] % 2), (0, 0)), mode="edge")
                rgb = add_timestamp_watermark(rgb, datetime.fromisoformat(frame["timestamp"]))
                Image.fromarray(rgb).save(directory / f"frame_{index:06d}.png")
                (temporary / f"{index}.npy").unlink()
                yield rgb
            report("encoding", count, count)

        write_mp4(directory / "video.mp4", rendered_frames(), config.fps, cancelled=abort_event.is_set)
        result = {"frame_count": count, "reference_count": len(samples["reference"]),
                  "scale_limit": limit, "maximum_difference": maximum, "fps": config.fps}
        write_json(directory / "results.json", {"summary": result, "frames": frames})
        return result
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
