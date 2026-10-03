from __future__ import annotations

from collections import deque
from copy import deepcopy
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
from app.preprocessing.pipeline import compile_pipeline
from app.schemas import PreprocessingGraph
from app.image_selection import prepare, resolve_source, select_prepared, freeze_samples, read_frozen_image
from app.training.scheduler import next_queue_rank, scheduler
from app.reference_image.engine import grayscale
from app.mean_variance.engine import OnlineMoments, difference_maps, render_heatmap, render_comparison, map_statistics
from app.mean_variance.schemas import MeanVarianceConfig, MeanVarianceRunRead, VarianceConfig, VariancePreview, StoredConfig, parse_config

logger = logging.getLogger(__name__)


class AbortedError(Exception):
    pass


def artifact_dir(run_id: int) -> Path:
    return data_dir() / "mean_variance_runs" / str(run_id)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare_comparison(db, config):
    if not isinstance(config, VarianceConfig):
        return prepare(db, config)
    dataset, pipeline, records = resolve_source(db, config)
    samples, previews, errors = {}, [], []
    for index, pair in enumerate(config.pairs):
        try:
            selected, preview = select_prepared(dataset, records, config.selection_config(pair))
        except ValueError as exc:
            raise ValueError(f"u{index + 1}: {exc}") from exc
        previews.append(preview)
        errors.extend(f"u{index + 1}: {message.replace('Referenz:', 'Normalzustand:')}" for message in preview.errors)
        samples.update({f"{index}_{role}": group for role, group in selected.items()})
    return dataset, pipeline, samples, VariancePreview(pairs=previews, errors=errors)


def preview(db: Session, config):
    _, pipeline, samples, result = prepare_comparison(db, config)
    compiled = compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
    shape = None
    for role in samples:
        if samples[role]:
            output = grayscale(compiled.run(samples[role][0]["file_path"]), shape)
            shape = output.shape
    result.errors = [message.replace("Referenz:", "Normalphase:") for message in result.errors]
    return result


def enqueue(db: Session, config: StoredConfig, *, wake_scheduler: bool = True):
    dataset, pipeline, samples, selection = prepare_comparison(db, config)
    if selection.errors:
        raise ValueError(" ".join(selection.errors).replace("Referenz:", "Normalphase:"))
    # Freeze the selected file identities at enqueue time, not at worker dispatch.
    freeze_samples(samples)
    snapshot = {"id": dataset.id, "name": dataset.name, "rules": [{
        "id": rule.id, "start": rule.start_timestamp.isoformat(), "end": rule.end_timestamp.isoformat(),
        "stride": rule.stride, "folder_id": rule.folder_id, "folder": rule.folder.relative_path,
        "root_path": rule.folder.dataset.root_path, "timestamp_regex": rule.folder.dataset.timestamp_regex,
        "timestamp_format": rule.folder.dataset.timestamp_format,
    } for rule in dataset.rules]}
    run = models.MeanVarianceRun(
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
    return MeanVarianceRunRead.model_validate(run)


def list_runs(db: Session):
    return [MeanVarianceRunRead.model_validate(run) for run in db.scalars(
        select(models.MeanVarianceRun).order_by(models.MeanVarianceRun.id.desc()))]


def get_run(db: Session, run_id: int):
    run = db.get(models.MeanVarianceRun, run_id)
    return MeanVarianceRunRead.model_validate(run) if run else None


def abort_run(db: Session, run_id: int):
    run = db.get(models.MeanVarianceRun, run_id)
    if run is None:
        return None
    if run.status not in {"queued", "running"}:
        raise ValueError("Nur wartende oder laufende Analysen können abgebrochen werden.")
    # A persisted flag also covers cancellation racing with worker startup.
    db.execute(update(models.MeanVarianceRun).where(
        models.MeanVarianceRun.id == run_id, models.MeanVarianceRun.status.in_(["queued", "running"])
    ).values(cancel_requested=True))
    db.execute(update(models.MeanVarianceRun).where(
        models.MeanVarianceRun.id == run_id, models.MeanVarianceRun.status == "queued"
    ).values(status="aborted", current_step="aborted", ended_at=models.utc_now()))
    db.commit()
    db.refresh(run)
    if run.status == "running" and run.pid is not None:
        scheduler.request_abort("mean_variance", run.id, run.pid)
    return MeanVarianceRunRead.model_validate(run)


def delete_run(db: Session, run_id: int):
    run = db.get(models.MeanVarianceRun, run_id)
    if run is None:
        return False
    if run.status in {"running", "queued"}:
        raise ValueError("Analyse vor dem Löschen abbrechen und auf das Ende warten.")
    db.delete(run)
    db.commit()
    shutil.rmtree(artifact_dir(run_id), ignore_errors=True)
    return True


def artifact_path(db: Session, run_id: int, name: str):
    run = db.get(models.MeanVarianceRun, run_id)
    if run is None or run.status != "finished":
        return None
    if name not in {"mean_difference.png", "variance_difference.png", "variance_comparison.png", "results.json"}:
        return None
    path = artifact_dir(run_id) / name
    return path if path.is_file() else None


def results(db: Session, run_id: int):
    path = artifact_path(db, run_id, "results.json")
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def read_log(db: Session, run_id: int):
    if db.get(models.MeanVarianceRun, run_id) is None:
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
        run = db.get(models.MeanVarianceRun, run_id)
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
                        current = session.get(models.MeanVarianceRun, run_id)
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
        report("finished", result["total_images"], result["total_images"])
        db.execute(update(models.MeanVarianceRun).where(
            models.MeanVarianceRun.id == run_id, models.MeanVarianceRun.cancel_requested.is_(False)
        ).values(status="finished", result=result, ended_at=models.utc_now(),
                 duration_seconds=round(time.perf_counter() - started, 3)))
        db.commit()
        db.refresh(run)
        if run.cancel_requested:
            raise AbortedError()
    except Exception as exc:
        db.rollback()
        run = db.get(models.MeanVarianceRun, run_id)
        if run:
            aborted = isinstance(exc, AbortedError) or abort_event.is_set() or run.cancel_requested
            run.status = run.current_step = "aborted" if aborted else "failed"
            run.error_message = "Analyse abgebrochen." if aborted else str(exc)
            run.ended_at = models.utc_now()
            run.duration_seconds = round(time.perf_counter() - started, 3)
            db.commit()
            if not aborted:
                logger.exception("Variance comparison failed")
    finally:
        stopped.set()
        if heartbeat_thread:
            heartbeat_thread.join(timeout=3)
        db.close()


def calculate(run, report, abort_event):
    config = parse_config(run.config)
    if isinstance(config, VarianceConfig):
        return calculate_pairs(run, config, report, abort_event)
    directory = artifact_dir(run.id)
    temporary = directory / "exporting"
    samples = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["samples"]
    pipeline = compile_pipeline(PreprocessingGraph.model_validate(run.pipeline_snapshot["graph"]))
    counts = {role: len(samples[role]) for role in ("reference", "anomaly")}
    total = sum(counts.values())
    shape = None
    done = 0
    moments = {}
    try:
        temporary.mkdir(parents=True, exist_ok=True)
        for role in ("reference", "anomaly"):
            accumulator = OnlineMoments()
            for sample in samples[role]:
                report(role, done, total)
                if abort_event.is_set():
                    raise AbortedError()
                array = grayscale(read_frozen_image(sample, pipeline), shape)
                shape = array.shape
                accumulator.add(array)
                done += 1
            moments[role] = accumulator.finish()
        report("difference", total, total)
        mean, variance = difference_maps(moments["reference"], moments["anomaly"])
        del moments
        maps = {}
        for key, values, scale, signed in (
            ("mean", mean, config.mean_scale, False),
            ("variance", variance, config.variance_scale, True),
        ):
            report("rendering", total, total)
            maps[key] = render_heatmap(values, scale, signed, temporary / f"{key}_difference.png",
                                       config, counts, run.training_dataset_name)
        report("exporting", total, total)
        warnings = [f"{'Normalphase' if role == 'reference' else 'Anomaliephase'}: Nur ein Bild; Varianz 0, keine zeitliche Vergleichsbasis."
                    for role, count in counts.items() if count == 1]
        result = {"total_images": total, "reference_count": counts["reference"], "anomaly_count": counts["anomaly"],
                  "width": shape[1], "height": shape[0], "ddof": 0, "maps": maps, "warnings": warnings}
        write_json(temporary / "results.json", result)
        for name in ("mean_difference.png", "variance_difference.png", "results.json"):
            (temporary / name).replace(directory / name)
        return result
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def calculate_pairs(run, config, report, abort_event):
    directory = artifact_dir(run.id)
    temporary = directory / "exporting"
    samples = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["samples"]
    pipeline = compile_pipeline(PreprocessingGraph.model_validate(run.pipeline_snapshot["graph"]))
    total = sum(len(group) for group in samples.values())
    done, shape = 0, None
    pairs, warnings = [], []
    def check_abort():
        if abort_event.is_set():
            raise AbortedError()
    try:
        temporary.mkdir(parents=True, exist_ok=True)
        for index, pair in enumerate(config.pairs):
            check_abort()
            maps, counts = {}, {}
            for role, sample_role in (("normal", "reference"), ("anomaly", "anomaly")):
                accumulator = OnlineMoments()
                group = samples[f"{index}_{sample_role}"]
                for sample in group:
                    check_abort()
                    report(f"u{index + 1}_{role}", done, total)
                    array = grayscale(read_frozen_image(sample, pipeline), shape)
                    shape = array.shape
                    accumulator.add(array)
                    done += 1
                _, values = accumulator.finish()
                counts[role] = len(group)
                path = temporary / f"{index}_{role}.npy"
                np.save(path, values)
                maps[role] = {**map_statistics(values), "path": str(path)}
                del accumulator, values, array, _
                if len(group) == 1:
                    warnings.append(f"u{index + 1} {'Normalzustand' if role == 'normal' else 'Anomaliephase'}: Nur ein Bild; Varianz 0, keine zeitliche Vergleichsbasis.")
            normal = np.load(maps["normal"]["path"], mmap_mode="r")
            anomaly = np.load(maps["anomaly"]["path"], mmap_mode="r")
            with np.errstate(over="ignore", invalid="ignore"):
                difference = anomaly - normal
            if not np.isfinite(difference).all():
                raise ValueError(f"u{index + 1}: Die Differenz enthält nicht endliche Werte.")
            path = temporary / f"{index}_difference.npy"
            np.save(path, difference)
            maps["difference"] = {**map_statistics(difference), "path": str(path)}
            for role, stats in maps.items():
                if stats["all_zero"]:
                    warnings.append(f"u{index + 1} {dict(normal='Normalzustand', anomaly='Anomaliephase', difference='Differenz')[role]}: Alle Pixelwerte sind 0.")
            pairs.append({"label": f"u{index + 1}", "periods": pair.model_dump(mode="json"), "counts": counts, "maps": maps})
            del normal, anomaly, difference
        check_abort()
        report("rendering", total, total)
        result = render_comparison(pairs, config, temporary / "variance_comparison.png",
                                   run.training_dataset_name, run.pipeline_snapshot["name"], shape, check_abort)
        result.update(total_images=total, warnings=warnings)
        write_json(temporary / "results.json", result)
        check_abort()
        report("exporting", total, total)
        for name in ("variance_comparison.png", "results.json"):
            (temporary / name).replace(directory / name)
        return result
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
