"""Project-local, scheduled ROI preparation and immutable export revisions."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app import models
from app.database import data_dir
from app.image_selection import read_frozen_image
from app.mean_variance import service
from app.mean_variance.engine import OnlineMoments
from app.mean_variance.roi_engine import RoiConfig, export_roi, finish_basis, load_basis
from app.preprocessing.pipeline import compile_pipeline
from app.reference_image.engine import grayscale
from app.schemas import PreprocessingGraph
from app.training.scheduler import scheduler, next_queue_rank


def artifact_dir(job_id):
    return data_dir() / "variance_roi_jobs" / str(job_id)


def basis_dir(parent_id, db=None):
    original = service.artifact_dir(parent_id) / "roi_data"
    if (original / "basis.json").is_file() or db is None:
        return original
    prepared = db.scalar(select(models.VarianceRoiJob).where(
        models.VarianceRoiJob.parent_run_id == parent_id,
        models.VarianceRoiJob.operation == "prepare", models.VarianceRoiJob.status == "finished"
    ).order_by(models.VarianceRoiJob.id.desc()).limit(1))
    return artifact_dir(prepared.id) / "roi_data" if prepared else original


def job_read(job):
    return {column.name: getattr(job, column.name) for column in models.VarianceRoiJob.__table__.columns} if job else None


def list_jobs(db):
    return [job_read(job) for job in db.scalars(select(models.VarianceRoiJob).order_by(models.VarianceRoiJob.id.desc()))]


def parent_required(db, parent_id):
    parent = db.get(models.MeanVarianceRun, parent_id)
    if parent is None:
        raise LookupError("Varianzvergleich nicht gefunden.")
    if parent.status != "finished":
        raise ValueError("Die ROI-Auswertung ist erst nach einem fertigen Varianzvergleich verfügbar.")
    return parent


def state(db, parent_id):
    parent_required(db, parent_id)
    latest = db.scalar(select(models.VarianceRoiJob).where(models.VarianceRoiJob.parent_run_id == parent_id)
                       .order_by(models.VarianceRoiJob.id.desc()).limit(1))
    successful = db.scalar(select(models.VarianceRoiJob).where(models.VarianceRoiJob.parent_run_id == parent_id,
                           models.VarianceRoiJob.operation == "evaluate", models.VarianceRoiJob.status == "finished")
                           .order_by(models.VarianceRoiJob.id.desc()).limit(1))
    directory = basis_dir(parent_id, db)
    ready = (directory / "basis.json").is_file()
    return {"ready": ready, "basis": load_basis(directory) if ready else None,
            "job": job_read(latest), "saved": job_read(successful)}


def enqueue(db, parent_id, operation, config=None, *, wake_scheduler=True):
    parent = parent_required(db, parent_id)
    current = state(db, parent_id)
    if current["job"] and current["job"]["status"] in {"queued", "running"}:
        raise ValueError("Für diesen Vergleich läuft bereits ein ROI-Auftrag.")
    if operation == "prepare":
        if current["ready"]:
            raise ValueError("Die ROI-Daten sind bereits vorhanden.")
        payload = {}
    else:
        if not current["ready"]:
            raise ValueError("Bitte zuerst die ROI-Daten nachberechnen.")
        config.roi.validate_bounds(current["basis"]["width"], current["basis"]["height"])
        payload = config.model_dump()
    job = models.VarianceRoiJob(parent_run_id=parent_id, operation=operation,
                               training_dataset_name=parent.training_dataset_name, config=payload,
                               status="queued", current_step="queued", queue_rank=next_queue_rank(db), enqueued_at=models.utc_now())
    db.add(job)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("Für diesen Vergleich läuft bereits ein ROI-Auftrag.") from exc
    db.refresh(job)
    if wake_scheduler:
        scheduler.wake()
    return job_read(job)


def abort_job(db, job_id):
    job = db.get(models.VarianceRoiJob, job_id)
    if job is None:
        raise LookupError("ROI-Auftrag nicht gefunden.")
    if job.status not in {"queued", "running"}:
        raise ValueError("Nur wartende oder laufende ROI-Aufträge können abgebrochen werden.")
    db.execute(update(models.VarianceRoiJob).where(models.VarianceRoiJob.id == job_id,
               models.VarianceRoiJob.status.in_(["queued", "running"])).values(cancel_requested=True))
    db.execute(update(models.VarianceRoiJob).where(models.VarianceRoiJob.id == job_id,
               models.VarianceRoiJob.status == "queued").values(status="aborted", current_step="aborted", ended_at=models.utc_now()))
    db.commit(); db.refresh(job)
    if job.status == "running" and job.pid is not None:
        scheduler.request_abort("variance_roi", job.id, job.pid)
    return job_read(job)


def delete_job(db, job_id):
    job = db.get(models.VarianceRoiJob, job_id)
    if job is None:
        return False
    if job.status in {"queued", "running"}:
        raise ValueError("ROI-Auftrag vor dem Löschen abbrechen und auf das Ende warten.")
    if job.operation == "prepare" and job.status == "finished" and (artifact_dir(job_id) / "roi_data").is_dir():
        active = db.scalar(select(models.VarianceRoiJob.id).where(
            models.VarianceRoiJob.parent_run_id == job.parent_run_id,
            models.VarianceRoiJob.status.in_(["queued", "running"])).limit(1))
        if active is not None:
            raise ValueError("Die Nachbereitungsdaten werden noch von einem aktiven ROI-Auftrag verwendet.")
        original = service.artifact_dir(job.parent_run_id) / "roi_data"
        if not original.exists():
            (artifact_dir(job_id) / "roi_data").replace(original)
    db.delete(job); db.commit()
    shutil.rmtree(artifact_dir(job_id), ignore_errors=True)
    return True


def delete_children(db, parent_id):
    jobs = list(db.scalars(select(models.VarianceRoiJob).where(models.VarianceRoiJob.parent_run_id == parent_id)))
    if any(job.status in {"queued", "running"} for job in jobs):
        raise ValueError("Aktive ROI-Aufträge vor dem Löschen des Vergleichs abbrechen und auf das Ende warten.")
    for job in jobs:
        db.delete(job)
        shutil.rmtree(artifact_dir(job.id), ignore_errors=True)
    db.flush()


def job_log(db, job_id):
    from collections import deque
    if db.get(models.VarianceRoiJob, job_id) is None:
        raise LookupError("ROI-Auftrag nicht gefunden.")
    path = artifact_dir(job_id) / "worker.log"
    if not path.exists():
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=400))


def prepare_basis(parent, output, report, abort_event):
    samples = json.loads((service.artifact_dir(parent.id) / "manifest.json").read_text(encoding="utf-8"))["samples"]
    pipeline = compile_pipeline(PreprocessingGraph.model_validate(parent.pipeline_snapshot["graph"]))
    count = len(parent.config["pairs"]) if parent.config.get("version") == 2 else 1
    total, done, shape = sum(len(group) for group in samples.values()), 0, None
    output.mkdir(parents=True)
    for index in range(count):
        normal_variance = None
        for role in ("reference", "anomaly"):
            key = f"{index}_{role}" if parent.config.get("version") == 2 else role
            moments = OnlineMoments()
            for sample in samples[key]:
                report(f"u{index + 1}_{role}", done, total)
                if abort_event.is_set():
                    raise service.AbortedError()
                values = grayscale(read_frozen_image(sample, pipeline), shape)
                shape = values.shape
                moments.add(values); done += 1
            mean, variance = moments.finish()
            if role == "reference":
                np.save(output / f"{index}_mean.npy", mean)
                normal_variance = variance
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = variance - normal_variance
                np.save(output / f"{index}_difference.npy", difference)
            del moments, mean, variance, values
        del normal_variance, difference
    result = parent.result or json.loads((service.artifact_dir(parent.id) / "results.json").read_text(encoding="utf-8"))
    finish_basis(output, parent.config, result, parent.training_dataset_name, parent.pipeline_snapshot["name"],
                 lambda: report("preparing", total, total))
    report("exporting", total, total)
    return total


def calculate(job, report, abort_event):
    from app.database import SessionLocal
    with SessionLocal() as db:
        parent = parent_required(db, job.parent_run_id)
        basis_path = basis_dir(parent.id, db)
        db.expunge(parent)
    directory = artifact_dir(job.id)
    temporary = directory / "exporting"
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        if job.operation == "prepare":
            total = prepare_basis(parent, temporary / "roi_data", report, abort_event)
            report("exporting", total, total)
            (temporary / "roi_data").replace(directory / "roi_data")
            return {"total_images": total}
        config = RoiConfig.model_validate(job.config)
        basis = load_basis(basis_path)
        total = len(basis["pairs"])
        def check_abort():
            report("rendering", 0, total)
        result = export_roi(basis_path, temporary, config, check_abort)
        result["total_images"] = total
        service.write_json(temporary / "results.json", result)
        report("exporting", total, total)
        # Files remain inaccessible until this immutable job revision is finished.
        for name in ("roi_comparison.png", "roi_table.png", "results.json"):
            (temporary / name).replace(directory / name)
        return result
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def run_scheduled(job_id, abort_event=None):
    service.run_scheduled(job_id, abort_event, job_model=models.VarianceRoiJob, calculator=calculate)
