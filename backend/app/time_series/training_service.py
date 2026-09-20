from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from sqlalchemy import select, func, update, or_
from sqlalchemy.exc import IntegrityError

from app import models
from app.database import data_dir
from app.time_series.data import prepare, SUBSETS
from app.time_series.definitions import DEFINITIONS, definition, defaults, provenance, validate_values, representation_metadata


class Conflict(ValueError):
    pass


def require(db, model, record_id):
    record = db.get(model, record_id)
    if record is None:
        raise LookupError("Eintrag nicht gefunden.")
    return record


def record_dict(record, fields):
    return {field: getattr(record, field) for field in fields.split()}


def model_read(record):
    return record_dict(record, "id name kind config provenance template_key created_at updated_at")


def ensure_models(db):
    for kind in DEFINITIONS:
        if db.scalar(select(models.TimeSeriesModel.id).where(models.TimeSeriesModel.template_key == kind)):
            continue
        config = defaults(kind, "architecture")
        db.add(models.TimeSeriesModel(name=definition(kind)["label"], kind=kind, config=config,
                                     provenance=provenance(kind, config), template_key=kind))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()  # Another project request seeded the same unique templates.


def list_models(db):
    ensure_models(db)
    return [model_read(row) for row in db.scalars(select(models.TimeSeriesModel).where(models.TimeSeriesModel.name != '').order_by(models.TimeSeriesModel.id))]


def save_model(db, payload, record_id=None):
    config = validate_values(payload.kind, "architecture", payload.config)
    if record_id:
        row = require(db, models.TimeSeriesModel, record_id)
        if row.kind != payload.kind:
            raise ValueError("Der Modelltyp einer gespeicherten Variante bleibt fest.")
    else:
        row = models.TimeSeriesModel(kind=payload.kind)
        db.add(row)
    row.name, row.config = payload.name, config
    row.provenance = provenance(payload.kind, config)
    db.commit()
    db.refresh(row)
    return model_read(row)


def dataset_snapshot(dataset):
    return record_dict(dataset, "id name filename selected_columns timestamp_column timestamp_format label_column")


def resolve_preview(db, payload):
    split = require(db, models.TimeSeriesSplit, payload.split_id)
    dataset = require(db, models.TimeSeriesDataset, split.dataset_id)
    prepared = prepare(dataset.source_csv, dataset_snapshot(dataset), split.intervals, payload.window_length)
    summary = deepcopy(prepared.summary)
    if payload.model_id:
        model = require(db, models.TimeSeriesModel, payload.model_id)
        validate_architecture(model.kind, model.config, prepared.window_length, len(prepared.columns))
        training = validate_values(model.kind, "training", payload.training)
        summary["model"] = model_read(model)
        summary["training"] = training
        summary["representation"] = representation_metadata(model.kind, prepared.window_length, model.config)
        summary["provenance"] = provenance(model.kind, model.config, training)
        for key, value in {"window_length": prepared.window_length, "step": 1, "scaling": "minmax", "gap_factor": 1.5}.items():
            spec = summary["provenance"]["input"][key]
            spec.update(value=value, overridden=value != spec["default"])
    return dataset, split, prepared, summary


def validate_architecture(kind, config, length, channels):
    validate_values(kind, "architecture", config)
    if kind == "tcn_ae" and config["pooling_factor"] > length:
        raise ValueError("TCN-Poolingfaktor darf L nicht überschreiten. Bitte Poolingfaktor anpassen; L bleibt unverändert.")
    if kind == "usad":
        n = length * channels
        a, b = max(1, int(n * config["hidden_ratio_1"])), max(1, int(n * config["hidden_ratio_2"]))
        count = 3 * (n * a + a * b + b * config["latent_dim"])
        if count > 100_000_000:
            raise ValueError("USAD-Konfiguration überschreitet 100 Mio. Gewichte. Fenster oder Encoderbreiten verkleinern.")


def pipeline_read(row):
    return record_dict(row, "id name dataset_id split_id model_id window_length training snapshot created_at updated_at")


def save_pipeline(db, payload, record_id=None):
    dataset, split, prepared, summary = resolve_preview(db, payload)
    if summary["errors"]:
        raise ValueError(" ".join(summary["errors"]))
    model = require(db, models.TimeSeriesModel, payload.model_id)
    snapshot = dict(dataset=dataset_snapshot(dataset), source_hash=hashlib.sha256(dataset.source_csv).hexdigest(),
                    split=dict(id=split.id, name=split.name, tags=deepcopy(split.tags), intervals=deepcopy(split.intervals)),
                    model=dict(id=model.id, name=model.name, kind=model.kind, config=deepcopy(model.config), version="1"),
                    training=summary["training"], provenance=summary["provenance"], preview=prepared.summary,
                    window_length=payload.window_length, step=1, scaling="minmax", representation=summary["representation"], architecture_version=definition(model.kind)["version"])
    row = require(db, models.TimeSeriesPipeline, record_id) if record_id else models.TimeSeriesPipeline()
    row.name, row.dataset_id, row.split_id, row.model_id = payload.name, dataset.id, split.id, model.id
    row.window_length, row.training, row.snapshot = payload.window_length, summary["training"], snapshot
    db.add(row)
    db.commit()
    db.refresh(row)
    return pipeline_read(row)


def list_pipelines(db):
    return [pipeline_read(row) for row in db.scalars(select(models.TimeSeriesPipeline).order_by(models.TimeSeriesPipeline.id.desc()))]


def artifact_dir(run_id):
    return data_dir() / "time_series_runs" / str(run_id)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def run_read(row, *, detail=True):
    result = record_dict(row, "id pipeline_id name kind status current_step epoch epochs train_loss val_loss checkpoint_selection selected_epoch selected_metric checkpoint result cancel_requested processed_windows total_windows error_message queue_rank enqueued_at started_at ended_at heartbeat_at duration_seconds device gpu_index pid created_at")
    if detail:
        result["snapshot"] = row.snapshot
    return result


def list_runs(db):
    return [run_read(row) for row in db.scalars(select(models.TimeSeriesRun).order_by(models.TimeSeriesRun.id.desc()))]


def enqueue(db, pipeline_id, *, wake_scheduler=True):
    if importlib.util.find_spec("torch") is None:
        raise ValueError("PyTorch fehlt. Bitte die ML-Abhängigkeiten von MLTrace installieren.")
    from app.training.scheduler import next_queue_rank, scheduler
    pipeline = require(db, models.TimeSeriesPipeline, pipeline_id)
    dataset = require(db, models.TimeSeriesDataset, pipeline.dataset_id)
    snapshot = deepcopy(pipeline.snapshot)
    if hashlib.sha256(dataset.source_csv).hexdigest() != snapshot["source_hash"]:
        raise Conflict("Die Quelldatei stimmt nicht mehr mit der gespeicherten Pipeline überein.")
    prepared = prepare(dataset.source_csv, snapshot["dataset"], snapshot["split"]["intervals"], pipeline.window_length)
    if prepared.summary["errors"]:
        raise ValueError(" ".join(prepared.summary["errors"]))
    snapshot["preview"] = prepared.summary
    snapshot["pipeline"] = dict(id=pipeline.id, name=pipeline.name)
    has_val = prepared.summary["counts"]["validation"]["windows"] > 0
    kind = snapshot["model"]["kind"]
    selection = ("validation_reconstruction_score" if kind == "usad" else "validation_loss") if has_val else "final_epoch"
    snapshot["checkpoint_metric"] = dict(selection=selection, subset="validation" if has_val else None,
        definition="mean_window_score" if kind == "usad" else "mean_logcosh" if kind == "tcn_ae" else "mean_gaussian_nll_plus_weighted_mean_kl",
        alpha=snapshot["training"].get("alpha"), beta=1 - snapshot["training"]["alpha"] if kind == "usad" else None,
        kl_weight=snapshot["training"].get("kl_weight"), noise_std=0, latent_sampling=False, tie_break="earliest_epoch")
    row = models.TimeSeriesRun(pipeline_id=pipeline.id, name=pipeline.name, kind=kind, snapshot=snapshot,
        epochs=snapshot["training"]["epochs"], checkpoint_selection=selection, status="queued", current_step="queued",
        enqueued_at=models.utc_now(), queue_rank=next_queue_rank(db), total_windows=len(prepared.endpoints))
    db.add(row)
    directory = None
    try:
        db.flush()
        directory = artifact_dir(row.id)
        directory.mkdir(parents=True, exist_ok=True)
        write_npz(directory / "input.npz", values=prepared.values, scaled=prepared.scaled, timestamps=prepared.timestamps,
                  groups=prepared.groups, interval_indices=prepared.interval_indices, segment_ids=prepared.segment_ids, endpoints=prepared.endpoints)
        write_json(directory / "snapshot.json", snapshot)
        db.commit()
    except Exception:
        db.rollback()
        if directory:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    db.refresh(row)
    if wake_scheduler:
        scheduler.wake()
    return run_read(row)


def abort_run(db, record_id):
    from app.training.scheduler import scheduler
    row = require(db, models.TimeSeriesRun, record_id)
    if row.status in {"finished", "failed", "aborted"}:
        return run_read(row)
    # Compare-and-set prevents a queued cancellation overwriting a dispatch.
    db.execute(update(models.TimeSeriesRun).where(models.TimeSeriesRun.id == record_id,
        models.TimeSeriesRun.status == "queued").values(status="aborted", current_step="aborted", cancel_requested=True, ended_at=models.utc_now()))
    db.execute(update(models.TimeSeriesRun).where(models.TimeSeriesRun.id == record_id,
        models.TimeSeriesRun.status == "running").values(cancel_requested=True))
    db.commit()
    db.refresh(row)
    # Workers poll the persisted flag, so cancellation is safe across projects
    # and needs no signaling of an unverified or reused PID.
    scheduler.wake()
    return run_read(row)


def guard_delete(db, kind, record_id):
    if kind == "dataset":
        predicate = or_(models.TimeSeriesPipeline.dataset_id == record_id,
            models.TimeSeriesPipeline.split_id.in_(select(models.TimeSeriesSplit.id).where(models.TimeSeriesSplit.dataset_id == record_id)))
    elif kind == "split":
        predicate = models.TimeSeriesPipeline.split_id == record_id
    elif kind == "model":
        predicate = models.TimeSeriesPipeline.model_id == record_id
    else:
        count = db.scalar(select(func.count()).select_from(models.TimeSeriesRun).where(models.TimeSeriesRun.pipeline_id == record_id))
        if count:
            raise Conflict(f"Pipeline wird von {count} Läufen verwendet. Zuerst diese Läufe löschen.")
        return
    names = db.scalars(select(models.TimeSeriesPipeline.name).where(predicate)).all()
    if names:
        raise Conflict("Eintrag wird von Trainingspipelines verwendet: " + ", ".join(names))


def delete_run(db, record_id):
    row = require(db, models.TimeSeriesRun, record_id)
    if row.status in {"queued", "running"}:
        raise Conflict("Aktive Läufe zuerst abbrechen; Löschen ist erst nach Abschluss möglich.")
    db.execute(models.TimeSeriesEpochMetric.__table__.delete().where(models.TimeSeriesEpochMetric.run_id == record_id))
    db.delete(row)
    db.commit()
    shutil.rmtree(artifact_dir(record_id), ignore_errors=True)
