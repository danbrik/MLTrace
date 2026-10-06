from collections import OrderedDict, deque
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil

import numpy as np
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.orm import object_session

from app import models
from app.database import data_dir
from app.image_selection import resolve_source, freeze_samples, read_frozen_image
from app.preprocessing.pipeline import compile_pipeline
from app.schemas import PreprocessingGraph
from app.training.scheduler import next_queue_rank, scheduler
from app.temporal_difference.engine import ROLES, LABELS, select_pairs, absolute_change, grayscale, statistics
from app.temporal_difference.schemas import PlotSettings, TemporalDifferenceConfig, TemporalDifferenceRunRead

from app.temporal_difference.units import convert_row, to_percent

Run = models.TemporalDifferenceRun
Value = models.TemporalDifferenceValue
Summary = models.TemporalDifferenceSummary
SUMMARY_FIELDS = ("role", "delta_seconds", "pair_count", "median", "q1", "q3", "iqr")
VALUE_FIELDS = ("id", "role", "delta_seconds", "first_file", "second_file", "first_timestamp", "second_timestamp", "first_utc", "second_utc", "value")


def artifact_dir(run_id):
    return data_dir() / "temporal_difference_runs" / str(run_id)


def prepare(db, config):
    dataset, pipeline, records = resolve_source(db, config)
    first = min(rule.start_timestamp for rule in dataset.rules)
    last = max(rule.end_timestamp for rule in dataset.rules)
    for role in ROLES:
        period = getattr(config, role)
        if period.start < first or period.end > last:
            raise ValueError(f"{LABELS[role]}: Der Zeitraum muss innerhalb der Datensatzgrenzen liegen.")
    samples, pairs, selection = select_pairs(records, config)
    return dataset, pipeline, samples, pairs, selection


def preview(db, config):
    _, pipeline, samples, _, selection = prepare(db, config)
    compiled = compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
    shape = None
    for role in ROLES:
        if samples[role]:
            shape = grayscale(compiled.run(samples[role][0]["file_path"]), shape).shape
    return selection


def enqueue(db, config, *, wake_scheduler=True):
    dataset, pipeline, samples, pairs, selection = prepare(db, config)
    if selection["errors"]:
        raise ValueError(" ".join(selection["errors"]))
    freeze_samples(samples)
    snapshot = {"id": dataset.id, "name": dataset.name, "rules": [{
        "id": rule.id, "start": rule.start_timestamp.isoformat(), "end": rule.end_timestamp.isoformat(),
        "stride": rule.stride, "folder_id": rule.folder_id, "folder": rule.folder.relative_path,
        "root_path": rule.folder.dataset.root_path, "timestamp_regex": rule.folder.dataset.timestamp_regex,
        "timestamp_format": rule.folder.dataset.timestamp_format,
    } for rule in dataset.rules]}
    run = Run(training_dataset_id=dataset.id, training_dataset_name=dataset.name, config=config.model_dump(mode="json"),
              dataset_snapshot=snapshot, pipeline_snapshot={"id": pipeline.id, "name": pipeline.name, "graph": deepcopy(pipeline.graph)},
              plot_settings=PlotSettings(unit_version=2, y_title="Mittlere Pixeländerung (%)").model_dump(), status="queued", current_step="queued",
              enqueued_at=models.utc_now(), queue_rank=next_queue_rank(db))
    db.add(run)
    directory = None
    try:
        db.flush()
        directory = artifact_dir(run.id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "manifest.json.part"
        temporary.write_text(json.dumps({"samples": samples, "pairs": pairs, "selection": selection}, allow_nan=False), encoding="utf-8")
        temporary.replace(directory / "manifest.json")
        db.commit()
    except Exception:
        db.rollback()
        if directory:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    db.refresh(run)
    if wake_scheduler:
        scheduler.wake()
    return TemporalDifferenceRunRead.model_validate(run)


def list_runs(db):
    return [TemporalDifferenceRunRead.model_validate(row) for row in db.scalars(select(Run).order_by(Run.id.desc()))]


def get_run(db, run_id):
    row = db.get(Run, run_id)
    return TemporalDifferenceRunRead.model_validate(row) if row else None


def finished(db, run_id):
    row = db.get(Run, run_id)
    return row if row and row.status == "finished" else None


def summaries(db, run_id, unit="raw"):
    if not finished(db, run_id):
        return None
    return [convert_row(dict(zip(SUMMARY_FIELDS, row)), unit, ("median", "q1", "q3", "iqr")) for row in db.execute(select(*(getattr(Summary, key) for key in SUMMARY_FIELDS))
        .where(Summary.run_id == run_id).order_by(Summary.delta_seconds, Summary.role))]


def values(db, run_id, offset=0, limit=50, role=None, delta=None, unit="raw"):
    if not finished(db, run_id):
        return None
    criteria = [Value.run_id == run_id]
    if role:
        criteria.append(Value.role == role)
    if delta is not None:
        criteria.append(Value.delta_seconds == delta)
    count = db.scalar(select(func.count()).select_from(Value).where(*criteria))
    rows = db.execute(select(*(getattr(Value, key) for key in VALUE_FIELDS)).where(*criteria).order_by(Value.id).offset(offset).limit(limit))
    return {"total": count, "items": [convert_row(dict(zip(VALUE_FIELDS, row)), unit, ("value",)) for row in rows]}


def save_plot(db, run_id, settings):
    run = finished(db, run_id)
    if run is None:
        return None
    run.plot_settings = settings.model_dump()
    db.commit()
    return settings


def csv_path(db, run_id, kind):
    if not finished(db, run_id):
        return None
    path = artifact_dir(run_id) / f"{kind}.csv"
    return path if path.is_file() else None


def read_log(db, run_id):
    if db.get(Run, run_id) is None:
        return None
    path = artifact_dir(run_id) / "worker.log"
    if not path.exists():
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=400))


def abort_run(db, run_id):
    run = db.get(Run, run_id)
    if run is None:
        return None
    if run.status not in {"queued", "running"}:
        raise ValueError("Nur wartende oder laufende Analysen können abgebrochen werden.")
    db.execute(update(Run).where(Run.id == run_id, Run.status.in_(["queued", "running"])).values(cancel_requested=True))
    db.execute(update(Run).where(Run.id == run_id, Run.status == "queued").values(status="aborted", current_step="aborted", ended_at=models.utc_now()))
    db.commit()
    db.refresh(run)
    if run.status == "running" and run.pid is not None:
        scheduler.request_abort("temporal_difference", run.id, run.pid)
    return TemporalDifferenceRunRead.model_validate(run)


def delete_run(db, run_id):
    run = db.get(Run, run_id)
    if run is None:
        return False
    if run.status in {"queued", "running"}:
        raise ValueError("Analyse vor dem Löschen abbrechen und auf das Ende warten.")
    for model in (Value, Summary):
        db.execute(delete(model).where(model.run_id == run_id))
    db.delete(run)
    db.commit()
    shutil.rmtree(artifact_dir(run_id), ignore_errors=True)
    return True


def check_file(sample):
    path = Path(sample["file_path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (sample["size_bytes"], sample["mtime_ns"]):
        raise ValueError(f"Quelldatei seit Erstellung verändert: {path.name}")


def calculate(run, report, abort_event):
    from app.mean_variance.service import AbortedError
    config = TemporalDifferenceConfig.model_validate(run.config)
    directory = artifact_dir(run.id)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    samples, pairs = manifest["samples"], manifest["pairs"]
    pipeline = compile_pipeline(PreprocessingGraph.model_validate(run.pipeline_snapshot["graph"]))
    total, shape, cache_bytes = len(pairs), None, 0
    cache = OrderedDict()
    temporary = directory / "exporting"
    temporary.mkdir(exist_ok=True)
    db = object_session(run)
    def check_abort():
        if abort_event.is_set():
            raise AbortedError()
    def read(sample):
        nonlocal shape, cache_bytes
        check_abort()
        check_file(sample)
        key = sample["file_path"]
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        image = grayscale(read_frozen_image(sample, pipeline), shape)
        shape = image.shape
        while cache and cache_bytes + image.nbytes > 128 * 1024 * 1024:
            _, previous = cache.popitem(last=False)
            cache_bytes -= previous.nbytes
        if image.nbytes <= 128 * 1024 * 1024:
            cache[key] = image
            cache_bytes += image.nbytes
        return image
    try:
        report("validating", 0, total)
        for group in samples.values():
            for sample in group:
                check_abort()
                check_file(sample)
        for model in (Value, Summary):
            db.execute(delete(model).where(model.run_id == run.id))
        db.commit()
        batch = []
        # Visit all lags at each start image together for bounded cache reuse.
        pairs.sort(key=lambda pair: (pair["role"], pair["first"], pair["delta_seconds"]))
        for index, pair in enumerate(pairs):
            check_abort()
            first, second = (samples[pair["role"]][pair[key]] for key in ("first", "second"))
            value = absolute_change(read(first), read(second))
            batch.append(dict(run_id=run.id, role=pair["role"], delta_seconds=pair["delta_seconds"],
                first_file=first["file_path"], second_file=second["file_path"], first_timestamp=first["timestamp"],
                second_timestamp=second["timestamp"], first_utc=first["utc"], second_utc=second["utc"], value=value))
            if len(batch) >= 256 or index == total - 1:
                db.execute(insert(Value), batch)
                db.commit()
                batch.clear()
                report("calculating", index + 1, total)
        cache.clear()
        report("summarizing", total, total)
        summary_rows = []
        for role in ROLES:
            for delta in config.deltas_seconds:
                check_abort()
                numbers = np.fromiter(db.scalars(select(Value.value).where(Value.run_id == run.id, Value.role == role, Value.delta_seconds == delta)), dtype=np.float64)
                summary_rows.append(dict(run_id=run.id, role=role, delta_seconds=delta, **statistics(numbers)))
        db.execute(insert(Summary), summary_rows)
        db.commit()
        report("exporting", total, total)
        with (temporary / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(VALUE_FIELDS)
            rows = db.execute(select(*(getattr(Value, key) for key in VALUE_FIELDS)).where(Value.run_id == run.id).order_by(Value.id).execution_options(yield_per=256))
            for row in rows:
                check_abort()
                writer.writerow(row)
        with (temporary / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["delta_seconds"] + [f"{role}_{key}" for role in ROLES for key in ("pair_count", "median", "q1", "q3", "iqr")]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for delta in config.deltas_seconds:
                row = {"delta_seconds": delta}
                for summary in summary_rows:
                    if summary["delta_seconds"] == delta:
                        row.update({f"{summary['role']}_{key}": summary[key] for key in ("pair_count", "median", "q1", "q3", "iqr")})
                writer.writerow(row)
        for group in samples.values():
            for sample in group:
                check_abort()
                check_file(sample)
        for kind in ("summary", "pairs"):
            (temporary / f"{kind}.csv").replace(directory / f"{kind}.csv")
        return {"total_images": total, "total_pairs": total, "width": shape[1], "height": shape[0], "selection": manifest["selection"]}
    except Exception:
        db.rollback()
        for model in (Value, Summary):
            db.execute(delete(model).where(model.run_id == run.id))
        db.commit()
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def run_scheduled(run_id, abort_event=None):
    # Shared heartbeat, cancellation and completion protocol of image analyses.
    from app.mean_variance.service import run_scheduled as run_analysis
    run_analysis(run_id, abort_event, job_model=Run, calculator=calculate)


def percent_csv(path, kind):
    """Stream conversions from the completed raw export without rewriting it."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        numeric = {key for key in fields if key == "value" or key.endswith(("_median", "_q1", "_q3", "_iqr"))}
        names = {key: (key.replace("reference_", "normal_").replace("comparison_", "anomaly_") + "_percent" if key in numeric else key.replace("reference_", "normal_").replace("comparison_", "anomaly_")) for key in fields}
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([names[key] for key in fields])
        yield buffer.getvalue()
        for row in reader:
            buffer.seek(0); buffer.truncate(0)
            writer.writerow([to_percent(row[key]) if key in numeric and row[key] else LABELS.get(row[key], row[key]) if key == "role" else row[key] for key in fields])
            yield buffer.getvalue()
