from __future__ import annotations

from collections import deque
from copy import deepcopy
import csv
import json
import logging
from pathlib import Path
import shutil
import threading
import time

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app import models
from app.database import data_dir
from app.analysis.dinov3_engine import DinoEncoder, analyze_features, evaluate_clusters, model_rgb, sample_records, unit_rgb
from app.analysis.dinov3_schemas import RepresentationConfig, RepresentationRunRead
from app.preprocessing.pipeline import compile_pipeline, encode_png_data_url
from app.schemas import PreprocessingGraph
from app.training.data import enumerate_training_dataset_image_records
from app.training.scheduler import next_queue_rank, scheduler

logger = logging.getLogger(__name__)
ARTIFACTS = {"samples.csv", "features.npz", "analysis.json", "manifest.json", "points.json"}


class AbortedError(Exception):
    pass


def artifact_dir(run_id: int) -> Path:
    return data_dir() / "representation_runs" / str(run_id)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare(db: Session, config: RepresentationConfig):
    dataset = db.get(models.TrainingDataset, config.training_dataset_id)
    pipeline = db.get(models.PreprocessingPipeline, config.preprocessing_pipeline_id)
    if dataset is None or pipeline is None:
        raise ValueError("Datensatz oder Preprocessing-Pipeline nicht gefunden.")
    if not dataset.rules or any(rule.folder is None for rule in dataset.rules):
        raise ValueError("Der Datensatz enthält keine gültige vollständige Regelauswahl.")
    compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
    records = enumerate_training_dataset_image_records(dataset)
    samples, preview = sample_records(records, config)
    return dataset, pipeline, samples, preview


def preview(db: Session, config: RepresentationConfig):
    _, pipeline, samples, result = prepare(db, config)
    if samples:
        output = compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph)).run(samples[0]["file_path"])
        result.image = encode_png_data_url((unit_rgb(output) * 255).round().astype(np.uint8))
        result.model_image = encode_png_data_url((model_rgb(output) * 255).round().astype(np.uint8))
        result.timestamp = samples[0]["timestamp"]
    return result


def enqueue(db: Session, config: RepresentationConfig, *, wake_scheduler: bool = True):
    dataset, pipeline, samples, selection = prepare(db, config)
    if selection.errors:
        raise ValueError(" ".join(selection.errors))
    # Freeze the selected file identities at enqueue time, not at worker dispatch.
    for sample in samples:
        info = Path(sample["file_path"]).stat()
        sample["size_bytes"] = info.st_size
        sample["mtime_ns"] = info.st_mtime_ns
    snapshot = {"id": dataset.id, "name": dataset.name, "rules": [{
        "id": rule.id, "start": rule.start_timestamp.isoformat(), "end": rule.end_timestamp.isoformat(),
        "stride": rule.stride, "folder_id": rule.folder_id, "folder": rule.folder.relative_path,
        "root_path": rule.folder.dataset.root_path, "timestamp_regex": rule.folder.dataset.timestamp_regex,
        "timestamp_format": rule.folder.dataset.timestamp_format,
    } for rule in dataset.rules]}
    run = models.RepresentationRun(
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
    return RepresentationRunRead.model_validate(run)


def list_runs(db: Session):
    return [RepresentationRunRead.model_validate(run) for run in db.scalars(
        select(models.RepresentationRun).order_by(models.RepresentationRun.id.desc()))]


def get_run(db: Session, run_id: int):
    run = db.get(models.RepresentationRun, run_id)
    return RepresentationRunRead.model_validate(run) if run else None


def abort_run(db: Session, run_id: int):
    run = db.get(models.RepresentationRun, run_id)
    if run is None:
        return None
    if run.status not in {"queued", "running"}:
        raise ValueError("Nur wartende oder laufende Analysen können abgebrochen werden.")
    # A persisted flag also covers cancellation racing with worker startup.
    db.execute(update(models.RepresentationRun).where(
        models.RepresentationRun.id == run_id, models.RepresentationRun.status.in_(["queued", "running"])
    ).values(cancel_requested=True))
    db.execute(update(models.RepresentationRun).where(
        models.RepresentationRun.id == run_id, models.RepresentationRun.status == "queued"
    ).values(status="aborted", current_step="aborted", ended_at=models.utc_now()))
    db.commit()
    db.refresh(run)
    if run.status == "running" and run.pid is not None:
        scheduler.request_abort("dinov3_analysis", run.id, run.pid)
    return RepresentationRunRead.model_validate(run)


def delete_run(db: Session, run_id: int):
    run = db.get(models.RepresentationRun, run_id)
    if run is None:
        return False
    if run.status in {"running", "queued"}:
        raise ValueError("Analyse vor dem Löschen abbrechen und auf das Ende warten.")
    db.delete(run)
    db.commit()
    shutil.rmtree(artifact_dir(run_id), ignore_errors=True)
    return True


def artifact_path(db: Session, run_id: int, name: str):
    run = db.get(models.RepresentationRun, run_id)
    if run is None or name not in ARTIFACTS:
        return None
    if name != "manifest.json" and run.status != "finished":
        return None
    path = artifact_dir(run_id) / name
    return path if path.is_file() else None


def read_log(db: Session, run_id: int):
    if db.get(models.RepresentationRun, run_id) is None:
        return None
    path = artifact_dir(run_id) / "worker.log"
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=400))


def calculate(run, report, save_model, encoder_factory=DinoEncoder):
    config = RepresentationConfig.model_validate(run.config)
    directory = artifact_dir(run.id)
    report("selection", 0, None)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    pipeline = compile_pipeline(PreprocessingGraph.model_validate(run.pipeline_snapshot["graph"]))
    report("loading_model", 0, None)
    encoder = encoder_factory("cuda:0" if (run.device or "").startswith("GPU") else "cpu")
    save_model(encoder.snapshot)
    features = []
    report("features", 0, len(samples))
    for start in range(0, len(samples), 8):
        batch = samples[start:start + 8]
        images = []
        for sample in batch:
            path = Path(sample["file_path"])
            info = path.stat()
            if info.st_size != sample["size_bytes"] or info.st_mtime_ns != sample["mtime_ns"]:
                raise ValueError(f"Quelldatei seit Erstellung verändert: {path.name}")
            images.append(pipeline.run(str(path)))
        encoded = np.asarray(encoder.encode(images), dtype=np.float32)
        if encoded.ndim != 2 or len(encoded) != len(batch) or not np.isfinite(encoded).all():
            raise ValueError("Der Encoder lieferte ungültige Featurevektoren.")
        features.append(encoded)
        report("features", start + len(batch), len(samples))
    matrix = np.concatenate(features)
    pca_xy, umap_xy, clusters, metrics = analyze_features(matrix, config, report)
    report("evaluation", 0, None)
    metrics = evaluate_clusters([sample["label"] for sample in samples], clusters, metrics)
    points = [{**{key: sample[key] for key in ("file_path", "timestamp", "interval_id", "interval_name", "label", "event_id")},
               "pca_x": float(pca_xy[index, 0]), "pca_y": float(pca_xy[index, 1]),
               "umap_x": float(umap_xy[index, 0]), "umap_y": float(umap_xy[index, 1]),
               "cluster": int(clusters[index])} for index, sample in enumerate(samples)]
    report("saving", 0, None)
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "umap-learn")}
    write_json(directory / "points.json", points)
    write_json(directory / "analysis.json", {
        "config": run.config, "dataset": run.dataset_snapshot, "pipeline": run.pipeline_snapshot,
        "model": encoder.snapshot, "metrics": metrics, "versions": versions,
        "umap": {"n_neighbors": min(15, len(samples) - 1), "min_dist": 0.1, "metric": "euclidean", "init": "random"},
        "kmeans": {"n_init": 10},
    })
    with (directory / "features.npz.part").open("wb") as handle:
        np.savez_compressed(handle, features=matrix, file_paths=np.array([row["file_path"] for row in samples]),
                            timestamps=np.array([row["timestamp"] for row in samples]))
    (directory / "features.npz.part").replace(directory / "features.npz")
    with (directory / "samples.csv.part").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (directory / "samples.csv.part").replace(directory / "samples.csv")
    report("saving", len(samples), len(samples))
    return metrics


def run_scheduled(run_id: int, abort_event: threading.Event | None = None):
    from app.database import SessionLocal
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    db = SessionLocal()
    stopped = threading.Event()
    heartbeat_thread = None
    try:
        run = db.get(models.RepresentationRun, run_id)
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
                        current = session.get(models.RepresentationRun, run_id)
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

        def save_model(snapshot):
            run.model_snapshot = snapshot
            db.commit()

        result = calculate(run, report, save_model)
        report("finished", result["sample_count"], result["sample_count"])
        db.execute(update(models.RepresentationRun).where(
            models.RepresentationRun.id == run_id, models.RepresentationRun.cancel_requested.is_(False)
        ).values(status="finished", result=result, ended_at=models.utc_now(),
                 duration_seconds=round(time.perf_counter() - started, 3)))
        db.commit()
        db.refresh(run)
        if run.cancel_requested:
            raise AbortedError()
    except Exception as exc:
        db.rollback()
        run = db.get(models.RepresentationRun, run_id)
        if run:
            aborted = isinstance(exc, AbortedError) or abort_event.is_set() or run.cancel_requested
            run.status = run.current_step = "aborted" if aborted else "failed"
            run.error_message = "Analyse abgebrochen." if aborted else str(exc)
            run.ended_at = models.utc_now()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            db.commit()
            if not aborted:
                logger.exception("Representation analysis failed")
    finally:
        stopped.set()
        if heartbeat_thread:
            heartbeat_thread.join(timeout=3)
        db.close()
