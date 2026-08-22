"""Model-centred persistence and calculations for evaluation methods A and B."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app import models, schemas
from app.artifact_signatures import artifact_signature as calculate_artifact_signature
from app.continuity import continuity_segments, source_group
from app.evaluation.metrics import DEFAULT_EPSILON, empirical_wasserstein_1

SCORE_COLUMNS = {
    "score": models.TestingRunResult.score,
    "full_mse": models.TestingRunResult.full_mse,
    "roi_mse": models.TestingRunResult.roi_mse,
}


def _iso(value: datetime) -> str:
    return value.isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _workspace(db: Session, training_run_id: int) -> models.EvaluationModelWorkspace:
    training = db.execute(
        select(models.TrainingRun).where(models.TrainingRun.id == training_run_id).with_for_update()
    ).scalar_one_or_none()
    if training is None:
        raise LookupError("Training run not found.")
    if training.status == "finished" and not training.artifact_signature and training.artifact_path:
        training.artifact_signature = calculate_artifact_signature(training.artifact_path)
        if training.artifact_signature:
            db.add(training)
            db.flush()
    if training.status != "finished" or not training.artifact_signature:
        raise ValueError("The selected model requires a finished, signed artifact.")
    row = db.scalar(select(models.EvaluationModelWorkspace).where(
        models.EvaluationModelWorkspace.training_run_id == training.id,
        models.EvaluationModelWorkspace.artifact_signature == training.artifact_signature,
    ))
    if row is None:
        row = models.EvaluationModelWorkspace(
            training_run_id=training.id, artifact_signature=training.artifact_signature
        )
        db.add(row)
        db.flush()
    return row


def _locked_source(
    db: Session, workspace: models.EvaluationModelWorkspace, testing_run_id: int
) -> models.TestingRun:
    run = db.execute(
        select(models.TestingRun).where(models.TestingRun.id == testing_run_id).with_for_update()
    ).scalar_one_or_none()
    if run is None:
        raise LookupError("Testing run not found.")
    if run.status != "finished":
        raise ValueError("Only finished inference runs can be evaluated.")
    if not run.artifact_signature and run.artifact_path:
        run.artifact_signature = calculate_artifact_signature(run.artifact_path)
        if run.artifact_signature:
            db.add(run)
            db.flush()
    if run.training_run_id != workspace.training_run_id or run.artifact_signature != workspace.artifact_signature:
        raise ValueError("Inference run does not belong to the selected model artifact.")
    return run


def _scores(
    db: Session, run_id: int, score_series: str, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    column = SCORE_COLUMNS.get(score_series)
    if column is None:
        raise ValueError(f"Unsupported score series '{score_series}'.")
    rows = db.execute(
        select(
            models.TestingRunResult.position,
            models.TestingRunResult.timestamp,
            models.TestingRunResult.image_path,
            column.label("value"),
        ).where(
            models.TestingRunResult.testing_run_id == run_id,
            models.TestingRunResult.timestamp >= start,
            models.TestingRunResult.timestamp <= end,
        ).order_by(models.TestingRunResult.timestamp, models.TestingRunResult.position)
    ).all()
    if not rows:
        raise ValueError("The selected range contains no score points.")
    if any(row.value is None or not math.isfinite(float(row.value)) for row in rows):
        raise ValueError(f"Score series '{score_series}' contains missing or non-finite values.")
    segments = continuity_segments(
        [row.timestamp for row in rows], [source_group(row.image_path) for row in rows]
    )
    return [
        {"timestamp": row.timestamp, "position": row.position, "score": float(row.value), "segment": segment}
        for row, segment in zip(rows, segments, strict=True)
    ]


def _outdated(db: Session, calculation) -> bool:
    source = db.get(models.TestingRun, calculation.testing_run_id)
    return source is None or source.result_revision != calculation.source_result_revision or source.status != "finished"


def _mark_stale(db: Session, workspace: models.EvaluationModelWorkspace) -> None:
    """A separation result only ever reflects the current layout and run, so an outdated
    one is dropped instead of kept around; drift keeps its activatable history."""
    changed = False
    for calculation in db.scalars(select(models.EvaluationSeparationCalculation).where(
        models.EvaluationSeparationCalculation.workspace_id == workspace.id,
    )).all():
        if _outdated(db, calculation):
            db.delete(calculation)
            changed = True
    for calculation in db.scalars(select(models.EvaluationDriftCalculation).where(
        models.EvaluationDriftCalculation.workspace_id == workspace.id,
        models.EvaluationDriftCalculation.stale.is_(False),
    )).all():
        if _outdated(db, calculation):
            calculation.stale = True
            calculation.active = False
            changed = True
    if changed:
        db.flush()
        _reaggregate(db, workspace)


def _drop_layout_calculations(db: Session, calculation_model, layout_id: int) -> None:
    """Editing a layout invalidates every calculation that used it, in any workspace."""
    rows = db.scalars(select(calculation_model).where(calculation_model.layout_id == layout_id)).all()
    if not rows:
        return
    workspace_ids = {row.workspace_id for row in rows}
    for row in rows:
        db.delete(row)
    db.flush()
    for workspace in db.scalars(select(models.EvaluationModelWorkspace).where(
        models.EvaluationModelWorkspace.id.in_(workspace_ids)
    )).all():
        _reaggregate(db, workspace)


def _layout_calculation_counts(db: Session, calculation_model, layout_ids: list[int]) -> dict[int, int]:
    if not layout_ids:
        return {}
    rows = db.execute(
        select(calculation_model.layout_id, func.count(calculation_model.id))
        .where(calculation_model.layout_id.in_(layout_ids))
        .group_by(calculation_model.layout_id)
    ).all()
    return {layout_id: count for layout_id, count in rows}


def _reaggregate(db: Session, workspace: models.EvaluationModelWorkspace) -> None:
    separations = list(db.scalars(
        select(models.EvaluationSeparationResult.separation)
        .join(models.EvaluationSeparationCalculation)
        .where(
            models.EvaluationSeparationCalculation.workspace_id == workspace.id,
            models.EvaluationSeparationResult.included.is_(True),
        )
    ))
    workspace.sep_median = float(statistics.median(separations)) if separations else None
    workspace.sep_min = float(min(separations)) if separations else None
    active = db.scalar(select(models.EvaluationDriftCalculation).where(
        models.EvaluationDriftCalculation.workspace_id == workspace.id,
        models.EvaluationDriftCalculation.active.is_(True),
        models.EvaluationDriftCalculation.stale.is_(False),
    ).order_by(models.EvaluationDriftCalculation.created_at.desc()))
    workspace.active_drift_calculation_id = active.id if active else None
    workspace.drift_mean = active.drift_mean if active else None
    workspace.drift_max = active.drift_max if active else None
    db.add(workspace)
    db.flush()


def _summary(db: Session, workspace: models.EvaluationModelWorkspace) -> dict[str, Any]:
    _mark_stale(db, workspace)
    _reaggregate(db, workspace)
    included_count = db.scalar(
        select(func.count(models.EvaluationSeparationResult.id))
        .join(models.EvaluationSeparationCalculation)
        .where(
            models.EvaluationSeparationCalculation.workspace_id == workspace.id,
            models.EvaluationSeparationResult.included.is_(True),
        )
    ) or 0
    active = db.get(models.EvaluationDriftCalculation, workspace.active_drift_calculation_id) if workspace.active_drift_calculation_id else None
    return {
        "workspace_id": workspace.id,
        "training_run_id": workspace.training_run_id,
        "artifact_signature": workspace.artifact_signature,
        "sep_median": workspace.sep_median, "sep_min": workspace.sep_min,
        "d_mean": workspace.drift_mean, "d_max": workspace.drift_max,
        "included_separation_results": int(included_count),
        "active_drift_calculation_id": workspace.active_drift_calculation_id,
        "active_drift_testing_run_id": active.testing_run_id if active else None,
    }


def list_models(db: Session) -> list[dict[str, Any]]:
    rows = db.scalars(select(models.TrainingRun).where(
        models.TrainingRun.status == "finished", models.TrainingRun.artifact_path.is_not(None)
    ).order_by(models.TrainingRun.ended_at.desc())).all()
    result = []
    for training in rows:
        try:
            workspace = _workspace(db, training.id)
        except ValueError:
            # A legacy row whose artifact has disappeared cannot be evaluated.
            continue
        summary = _summary(db, workspace)
        summary.update({
            "name": training.training_pipeline_name,
            "method_type": training.method_type,
            "method_family": training.method_family,
            "preprocessing_pipeline_name": training.preprocessing_pipeline_name,
            "training_dataset_names": list(training.dataset_names or []),
            "ended_at": training.ended_at,
        })
        result.append(summary)
    db.commit()
    return result


def get_summary(db: Session, training_run_id: int) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id)
    result = _summary(db, workspace)
    db.commit()
    return result


def available_testing_runs(db: Session, training_run_id: int) -> list[models.TestingRun]:
    workspace = _workspace(db, training_run_id)
    candidates = db.scalars(select(models.TestingRun).where(
        models.TestingRun.training_run_id == training_run_id,
        models.TestingRun.status == "finished",
    ).order_by(models.TestingRun.ended_at.desc())).all()
    rows = []
    for run in candidates:
        if not run.artifact_signature and run.artifact_path:
            run.artifact_signature = calculate_artifact_signature(run.artifact_path)
            if run.artifact_signature:
                db.add(run)
        if run.artifact_signature == workspace.artifact_signature:
            rows.append(run)
    db.commit()
    return rows


def _validate_anomaly_overlap(pairs: list[schemas.EvaluationSeparationPairInput]) -> None:
    if len({pair.pair_key for pair in pairs}) != len(pairs):
        raise ValueError("Pair keys must be unique within a layout.")
    ordered = sorted(pairs, key=lambda pair: (pair.anomaly_start, pair.anomaly_end))
    for previous, current in zip(ordered, ordered[1:]):
        if current.anomaly_start <= previous.anomaly_end:
            raise ValueError(f"Anomaly ranges '{previous.name}' and '{current.name}' overlap.")


def _sep_layout_dict(row: models.EvaluationSeparationLayout) -> dict[str, Any]:
    return {"id": row.id, "training_dataset_id": row.training_dataset_id, "name": row.name,
            "description": row.description, "version": row.version,
            "pairs": [{"pair_key": p.pair_key, "name": p.name, "normal_start": p.normal_start,
                       "normal_end": p.normal_end, "anomaly_start": p.anomaly_start,
                       "anomaly_end": p.anomaly_end} for p in row.pairs],
            "created_at": row.created_at, "updated_at": row.updated_at}


def list_separation_layouts(db: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = db.scalars(select(models.EvaluationSeparationLayout).options(selectinload(models.EvaluationSeparationLayout.pairs)).where(
        models.EvaluationSeparationLayout.training_dataset_id == dataset_id
    ).order_by(models.EvaluationSeparationLayout.name)).all()
    counts = _layout_calculation_counts(db, models.EvaluationSeparationCalculation, [row.id for row in rows])
    return [{**_sep_layout_dict(row), "calculation_count": counts.get(row.id, 0)} for row in rows]


def save_separation_layout(db: Session, payload: schemas.EvaluationSeparationLayoutInput, layout_id: int | None = None) -> dict[str, Any]:
    _validate_anomaly_overlap(payload.pairs)
    duplicate = db.scalar(select(models.EvaluationSeparationLayout.id).where(
        models.EvaluationSeparationLayout.training_dataset_id == payload.training_dataset_id,
        models.EvaluationSeparationLayout.name == payload.name.strip(),
        models.EvaluationSeparationLayout.id != (layout_id or -1),
    ))
    if duplicate is not None:
        raise ValueError("A separation layout with this name already exists for the dataset.")
    row = db.get(models.EvaluationSeparationLayout, layout_id) if layout_id else None
    if layout_id and row is None:
        raise LookupError("Separation layout not found.")
    if row is None:
        row = models.EvaluationSeparationLayout(training_dataset_id=payload.training_dataset_id, name=payload.name)
        db.add(row); db.flush()
    elif row.training_dataset_id != payload.training_dataset_id:
        raise ValueError("A layout cannot be moved to another inference dataset.")
    else:
        row.version += 1
    row.name, row.description = payload.name.strip(), payload.description
    # The keys usually survive an edit, so the orphans must be gone before the
    # replacements are inserted or the (layout_id, pair_key) constraint trips.
    row.pairs.clear(); db.flush()
    for index, pair in enumerate(payload.pairs):
        row.pairs.append(models.EvaluationSeparationPair(position=index, **pair.model_dump()))
    _drop_layout_calculations(db, models.EvaluationSeparationCalculation, row.id)
    db.commit(); db.refresh(row)
    return {**_sep_layout_dict(row), "calculation_count": 0}


def delete_separation_layout(db: Session, layout_id: int) -> bool:
    row = db.get(models.EvaluationSeparationLayout, layout_id)
    if row is None: return False
    db.delete(row); db.commit(); return True


def calculate_separation(db: Session, training_run_id: int, payload: schemas.EvaluationSeparationCalculateRequest) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id)
    run = _locked_source(db, workspace, payload.testing_run_id)
    layout = db.scalar(select(models.EvaluationSeparationLayout).options(selectinload(models.EvaluationSeparationLayout.pairs)).where(models.EvaluationSeparationLayout.id == payload.layout_id))
    if layout is None: raise LookupError("Separation layout not found.")
    if layout.training_dataset_id != run.training_dataset_id: raise ValueError("Layout and inference dataset do not match.")
    selected = [p for p in layout.pairs if p.pair_key in set(payload.pair_keys)]
    if len(selected) != len(set(payload.pair_keys)): raise ValueError("One or more selected pairs do not exist.")
    snapshot = _sep_layout_dict(layout)
    calculation = db.scalar(select(models.EvaluationSeparationCalculation).where(
        models.EvaluationSeparationCalculation.workspace_id == workspace.id,
        models.EvaluationSeparationCalculation.testing_run_id == run.id,
        models.EvaluationSeparationCalculation.layout_id == layout.id,
        models.EvaluationSeparationCalculation.layout_version == layout.version,
        models.EvaluationSeparationCalculation.score_series == payload.score_series,
        models.EvaluationSeparationCalculation.source_result_revision == run.result_revision,
    ))
    if calculation is None:
        calculation = models.EvaluationSeparationCalculation(
            workspace_id=workspace.id, testing_run_id=run.id, layout_id=layout.id,
            layout_version=layout.version, layout_snapshot=_json_safe(snapshot), score_series=payload.score_series,
            artifact_signature=workspace.artifact_signature, source_result_revision=run.result_revision,
        ); db.add(calculation); db.flush()
    existing = {result.pair_key: result for result in calculation.results}
    for pair in selected:
        points = _scores(db, run.id, payload.score_series, min(pair.normal_start, pair.anomaly_start), max(pair.normal_end, pair.anomaly_end))
        normal = [p for p in points if pair.normal_start <= p["timestamp"] < pair.normal_end]
        anomaly = [p for p in points if pair.anomaly_start <= p["timestamp"] <= pair.anomaly_end]
        if not normal or not anomaly: raise ValueError(f"Pair '{pair.name}' contains an empty score range.")
        if len({p["segment"] for p in normal}) > 1 or len({p["segment"] for p in anomaly}) > 1:
            raise ValueError(f"Pair '{pair.name}' crosses a continuity gap.")
        normal_values = np.asarray([p["score"] for p in normal], dtype=float)
        anomaly_values = np.asarray([p["score"] for p in anomaly], dtype=float)
        median = float(np.median(normal_values)); mad = float(np.median(np.abs(normal_values - median)))
        scale = 1.4826 * mad + DEFAULT_EPSILON
        z = (anomaly_values - median) / scale
        values = dict(pair_name=pair.name, normal_start=pair.normal_start, normal_end=pair.normal_end,
                      anomaly_start=pair.anomaly_start, anomaly_end=pair.anomaly_end,
                      normal_median=median, normal_mad=mad, robust_scale=scale,
                      normal_point_count=len(normal), anomaly_point_count=len(anomaly),
                      separation=float(np.median(z)), separation_p95=float(np.quantile(z, .95, method="linear")))
        result = existing.get(pair.pair_key)
        if result is None:
            result = models.EvaluationSeparationResult(calculation_id=calculation.id, pair_key=pair.pair_key, **values)
            db.add(result)
        else:
            for key, value in values.items(): setattr(result, key, value)
    _reaggregate(db, workspace); db.commit()
    return get_summary(db, training_run_id)


def list_separation_results(db: Session, training_run_id: int) -> list[dict[str, Any]]:
    workspace = _workspace(db, training_run_id); _mark_stale(db, workspace)
    rows = db.execute(select(models.EvaluationSeparationResult, models.EvaluationSeparationCalculation)
        .join(models.EvaluationSeparationCalculation)
        .where(models.EvaluationSeparationCalculation.workspace_id == workspace.id)
        .order_by(models.EvaluationSeparationCalculation.created_at.desc(), models.EvaluationSeparationResult.id)).all()
    db.commit()
    return [{"id": r.id, "calculation_id": c.id, "testing_run_id": c.testing_run_id,
             "layout_version": c.layout_version, "score_series": c.score_series,
             "source_result_revision": c.source_result_revision,
             "pair_key": r.pair_key, "pair_name": r.pair_name, "normal_start": r.normal_start,
             "normal_end": r.normal_end, "anomaly_start": r.anomaly_start, "anomaly_end": r.anomaly_end,
             "normal_median": r.normal_median, "normal_mad": r.normal_mad, "robust_scale": r.robust_scale,
             "normal_point_count": r.normal_point_count, "anomaly_point_count": r.anomaly_point_count,
             "separation": r.separation, "separation_p95": r.separation_p95, "included": r.included}
            for r, c in rows]


def update_separation_result(db: Session, training_run_id: int, result_id: int, included: bool) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id)
    row = db.scalar(select(models.EvaluationSeparationResult).join(models.EvaluationSeparationCalculation).where(
        models.EvaluationSeparationResult.id == result_id,
        models.EvaluationSeparationCalculation.workspace_id == workspace.id))
    if row is None: raise LookupError("Separation result not found.")
    row.included = included; _reaggregate(db, workspace); db.commit(); return get_summary(db, training_run_id)


def delete_separation_result(db: Session, training_run_id: int, result_id: int) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id)
    row = db.scalar(select(models.EvaluationSeparationResult).join(models.EvaluationSeparationCalculation).where(
        models.EvaluationSeparationResult.id == result_id,
        models.EvaluationSeparationCalculation.workspace_id == workspace.id))
    if row is None: raise LookupError("Separation result not found.")
    db.delete(row); _reaggregate(db, workspace); db.commit(); return get_summary(db, training_run_id)


def _drift_layout_dict(row: models.EvaluationDriftLayout) -> dict[str, Any]:
    return {"id": row.id, "training_dataset_id": row.training_dataset_id, "name": row.name,
            "description": row.description, "version": row.version,
            "reference_start": row.reference_start, "reference_end": row.reference_end,
            "analysis_start": row.analysis_start, "analysis_end": row.analysis_end,
            "bucket_seconds": row.bucket_seconds, "reference_exclusion_action": row.reference_exclusion_action,
            "exclusions": [{"exclusion_key": x.exclusion_key, "name": x.name, "start_timestamp": x.start_timestamp,
                            "end_timestamp": x.end_timestamp} for x in row.exclusions],
            "buckets": [{"bucket_key": b.bucket_key, "start_timestamp": b.start_timestamp,
                         "end_timestamp": b.end_timestamp, "decision": b.decision} for b in row.buckets],
            "created_at": row.created_at, "updated_at": row.updated_at}


def _generated_buckets(layout: schemas.EvaluationDriftLayoutInput) -> list[schemas.EvaluationDriftBucketInput]:
    decisions = {bucket.bucket_key: bucket.decision for bucket in layout.buckets}
    result = []; cursor = layout.analysis_start; index = 0; step = timedelta(seconds=layout.bucket_seconds)
    while cursor + step <= layout.analysis_end:
        end = cursor + step; key = f"bucket-{index}"
        result.append(schemas.EvaluationDriftBucketInput(bucket_key=key, start_timestamp=cursor, end_timestamp=end,
                                                         decision=decisions.get(key, "include")))
        cursor = end; index += 1
    if cursor < layout.analysis_end:
        result.append(schemas.EvaluationDriftBucketInput(bucket_key=f"remainder-{index}", start_timestamp=cursor,
                                                         end_timestamp=layout.analysis_end, decision="drop_bucket"))
    return result


def list_drift_layouts(db: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = db.scalars(select(models.EvaluationDriftLayout).options(selectinload(models.EvaluationDriftLayout.exclusions), selectinload(models.EvaluationDriftLayout.buckets)).where(
        models.EvaluationDriftLayout.training_dataset_id == dataset_id).order_by(models.EvaluationDriftLayout.name)).all()
    counts = _layout_calculation_counts(db, models.EvaluationDriftCalculation, [row.id for row in rows])
    return [{**_drift_layout_dict(row), "calculation_count": counts.get(row.id, 0)} for row in rows]


def save_drift_layout(db: Session, payload: schemas.EvaluationDriftLayoutInput, layout_id: int | None = None) -> dict[str, Any]:
    if len({item.exclusion_key for item in payload.exclusions}) != len(payload.exclusions):
        raise ValueError("Exclusion keys must be unique within a layout.")
    duplicate = db.scalar(select(models.EvaluationDriftLayout.id).where(
        models.EvaluationDriftLayout.training_dataset_id == payload.training_dataset_id,
        models.EvaluationDriftLayout.name == payload.name.strip(),
        models.EvaluationDriftLayout.id != (layout_id or -1),
    ))
    if duplicate is not None:
        raise ValueError("A drift layout with this name already exists for the dataset.")
    row = db.get(models.EvaluationDriftLayout, layout_id) if layout_id else None
    if layout_id and row is None: raise LookupError("Drift layout not found.")
    if row is None:
        row = models.EvaluationDriftLayout(training_dataset_id=payload.training_dataset_id, name=payload.name,
                                           reference_start=payload.reference_start, reference_end=payload.reference_end,
                                           analysis_start=payload.analysis_start, analysis_end=payload.analysis_end,
                                           bucket_seconds=payload.bucket_seconds)
        db.add(row); db.flush()
    elif row.training_dataset_id != payload.training_dataset_id: raise ValueError("A layout cannot be moved to another inference dataset.")
    else: row.version += 1
    row.name, row.description = payload.name.strip(), payload.description
    row.reference_start, row.reference_end = payload.reference_start, payload.reference_end
    row.analysis_start, row.analysis_end, row.bucket_seconds = payload.analysis_start, payload.analysis_end, payload.bucket_seconds
    row.reference_exclusion_action = payload.reference_exclusion_action
    row.exclusions.clear(); row.buckets.clear(); db.flush()
    for exclusion in payload.exclusions: row.exclusions.append(models.EvaluationDriftExclusion(**exclusion.model_dump()))
    for index, bucket in enumerate(_generated_buckets(payload)):
        row.buckets.append(models.EvaluationDriftBucket(position=index, **bucket.model_dump()))
    _drop_layout_calculations(db, models.EvaluationDriftCalculation, row.id)
    db.commit(); db.refresh(row); return {**_drift_layout_dict(row), "calculation_count": 0}


def delete_drift_layout(db: Session, layout_id: int) -> bool:
    row = db.get(models.EvaluationDriftLayout, layout_id)
    if row is None: return False
    db.delete(row); db.commit(); return True


def _overlaps(start: datetime, end: datetime, other_start: datetime, other_end: datetime) -> bool:
    return start <= other_end and other_start < end


def drift_preview(db: Session, workspace: models.EvaluationModelWorkspace, run: models.TestingRun, layout: schemas.EvaluationDriftLayoutInput, score_series: str) -> dict[str, Any]:
    minimum = min(layout.reference_start, layout.analysis_start); maximum = max(layout.reference_end, layout.analysis_end)
    points = _scores(db, run.id, score_series, minimum, maximum)
    exclusions = [(x.start_timestamp, x.end_timestamp) for x in layout.exclusions]
    def blocked(point): return any(start <= point["timestamp"] <= end for start, end in exclusions)
    reference_all = [p for p in points if layout.reference_start <= p["timestamp"] < layout.reference_end]
    reference_overlap = any(_overlaps(layout.reference_start, layout.reference_end, a, b) for a, b in exclusions)
    if reference_overlap and layout.reference_exclusion_action == "drop_reference":
        raise ValueError("The reference was removed; select a new reference range.")
    reference = [p for p in reference_all if not blocked(p)] if reference_overlap else reference_all
    if not reference: raise ValueError("The reference contains no usable score points.")
    ref_values = np.asarray([p["score"] for p in reference], dtype=float)
    ref_iqr = float(np.quantile(ref_values, .75, method="linear") - np.quantile(ref_values, .25, method="linear"))
    bucket_rows = []
    for bucket in _generated_buckets(layout):
        values = [p for p in points if bucket.start_timestamp <= p["timestamp"] < bucket.end_timestamp]
        overlap = any(_overlaps(bucket.start_timestamp, bucket.end_timestamp, a, b) for a, b in exclusions)
        status = "ready"; reason = None; used = values
        if bucket.bucket_key.startswith("remainder-"):
            status, reason, used = "excluded", "incomplete remainder", []
        elif _overlaps(bucket.start_timestamp, bucket.end_timestamp, layout.reference_start, layout.reference_end):
            status, reason, used = "excluded", "overlaps reference", []
        elif len({p["segment"] for p in values}) > 1:
            status, reason, used = "excluded", "continuity gap", []
        elif bucket.decision == "drop_bucket":
            status, reason, used = "removed", "removed by user", []
        elif overlap and bucket.decision == "include":
            status, reason, used = "conflict", "choose drop bucket or filter points", []
        elif overlap and bucket.decision == "filter_points":
            used = [p for p in values if not blocked(p)]
            if not used: status, reason = "excluded", "no points after exclusion filtering"
        elif not values:
            status, reason = "excluded", "no score points"
        w1 = d = None
        if status == "ready":
            w1 = float(empirical_wasserstein_1(ref_values, [p["score"] for p in used]))
            d = w1 / (ref_iqr + DEFAULT_EPSILON)
        bucket_rows.append({"bucket_key": bucket.bucket_key, "start_timestamp": bucket.start_timestamp,
                            "end_timestamp": bucket.end_timestamp, "decision": bucket.decision,
                            "original_point_count": len(values), "used_point_count": len(used),
                            "exclusion_overlap": overlap, "status": status, "reason": reason,
                            "wasserstein_1": w1, "normalized_drift": d})
    return {"reference_point_count": len(reference), "reference_original_point_count": len(reference_all),
            "reference_iqr": ref_iqr, "reference_exclusion_overlap": reference_overlap,
            "near_zero_iqr": ref_iqr <= DEFAULT_EPSILON, "buckets": bucket_rows}


def preview_drift(db: Session, training_run_id: int, payload: schemas.EvaluationDriftPreviewRequest) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id); run = _locked_source(db, workspace, payload.testing_run_id)
    if payload.layout.training_dataset_id != run.training_dataset_id: raise ValueError("Layout and inference dataset do not match.")
    result = drift_preview(db, workspace, run, payload.layout, payload.score_series); db.rollback(); return result


def calculate_drift(db: Session, training_run_id: int, payload: schemas.EvaluationDriftCalculateRequest) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id); run = _locked_source(db, workspace, payload.testing_run_id)
    row = db.scalar(select(models.EvaluationDriftLayout).options(selectinload(models.EvaluationDriftLayout.exclusions), selectinload(models.EvaluationDriftLayout.buckets)).where(models.EvaluationDriftLayout.id == payload.layout_id))
    if row is None: raise LookupError("Drift layout not found.")
    if row.training_dataset_id != run.training_dataset_id: raise ValueError("Layout and inference dataset do not match.")
    layout = schemas.EvaluationDriftLayoutInput.model_validate(_drift_layout_dict(row))
    preview = drift_preview(db, workspace, run, layout, payload.score_series)
    conflicts = [bucket for bucket in preview["buckets"] if bucket["status"] == "conflict"]
    if conflicts: raise ValueError("Every exclusion conflict needs a persisted bucket decision.")
    valid = [bucket for bucket in preview["buckets"] if bucket["status"] == "ready"]
    if not valid: raise ValueError("No valid drift bucket remains.")
    db.execute(select(models.EvaluationDriftCalculation).where(models.EvaluationDriftCalculation.workspace_id == workspace.id).with_for_update())
    for previous in db.scalars(select(models.EvaluationDriftCalculation).where(models.EvaluationDriftCalculation.workspace_id == workspace.id, models.EvaluationDriftCalculation.active.is_(True))): previous.active = False
    d_values = [bucket["normalized_drift"] for bucket in valid]
    calculation = models.EvaluationDriftCalculation(
        workspace_id=workspace.id, testing_run_id=run.id, layout_id=row.id, layout_version=row.version,
        layout_snapshot=_json_safe(_drift_layout_dict(row)), score_series=payload.score_series,
        artifact_signature=workspace.artifact_signature, source_result_revision=run.result_revision,
        reference_iqr=preview["reference_iqr"], reference_point_count=preview["reference_point_count"],
        drift_mean=float(np.mean(d_values)), drift_max=float(max(d_values)), active=True,
    ); db.add(calculation); db.flush()
    for bucket in preview["buckets"]:
        db.add(models.EvaluationDriftBucketResult(calculation_id=calculation.id, **{k: v for k, v in bucket.items() if k != "decision"}, included=bucket["status"] == "ready"))
    _reaggregate(db, workspace); db.commit(); return get_summary(db, training_run_id)


def list_drift_calculations(db: Session, training_run_id: int) -> list[dict[str, Any]]:
    workspace = _workspace(db, training_run_id); _mark_stale(db, workspace)
    rows = db.scalars(select(models.EvaluationDriftCalculation).options(selectinload(models.EvaluationDriftCalculation.results)).where(
        models.EvaluationDriftCalculation.workspace_id == workspace.id).order_by(models.EvaluationDriftCalculation.created_at.desc())).all(); db.commit()
    return [{"id": row.id, "testing_run_id": row.testing_run_id, "layout_id": row.layout_id,
             "layout_version": row.layout_version, "layout_snapshot": row.layout_snapshot,
             "score_series": row.score_series, "source_result_revision": row.source_result_revision,
             "reference_iqr": row.reference_iqr, "reference_point_count": row.reference_point_count,
             "d_mean": row.drift_mean, "d_max": row.drift_max, "stale": row.stale, "active": row.active,
             "created_at": row.created_at,
             "buckets": [{"id": b.id, "bucket_key": b.bucket_key, "start_timestamp": b.start_timestamp,
                          "end_timestamp": b.end_timestamp, "original_point_count": b.original_point_count,
                          "used_point_count": b.used_point_count, "exclusion_overlap": b.exclusion_overlap,
                          "status": b.status, "reason": b.reason, "wasserstein_1": b.wasserstein_1,
                          "normalized_drift": b.normalized_drift, "included": b.included} for b in row.results]}
            for row in rows]


def activate_drift(db: Session, training_run_id: int, calculation_id: int) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id); _mark_stale(db, workspace)
    target = db.scalar(select(models.EvaluationDriftCalculation).where(models.EvaluationDriftCalculation.id == calculation_id,
        models.EvaluationDriftCalculation.workspace_id == workspace.id).with_for_update())
    if target is None: raise LookupError("Drift calculation not found.")
    if target.stale: raise ValueError("A stale drift calculation cannot be activated.")
    for row in db.scalars(select(models.EvaluationDriftCalculation).where(models.EvaluationDriftCalculation.workspace_id == workspace.id)): row.active = row.id == target.id
    _reaggregate(db, workspace); db.commit(); return get_summary(db, training_run_id)


def delete_drift_calculation(db: Session, training_run_id: int, calculation_id: int) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id)
    row = db.scalar(select(models.EvaluationDriftCalculation).where(models.EvaluationDriftCalculation.id == calculation_id,
        models.EvaluationDriftCalculation.workspace_id == workspace.id))
    if row is None: raise LookupError("Drift calculation not found.")
    was_active = row.active; db.delete(row); db.flush()
    if was_active:
        replacement = db.scalar(select(models.EvaluationDriftCalculation).where(models.EvaluationDriftCalculation.workspace_id == workspace.id,
            models.EvaluationDriftCalculation.stale.is_(False)).order_by(models.EvaluationDriftCalculation.created_at.desc()))
        if replacement: replacement.active = True
    _reaggregate(db, workspace); db.commit(); return get_summary(db, training_run_id)


def update_drift_bucket(db: Session, training_run_id: int, result_id: int, included: bool) -> dict[str, Any]:
    workspace = _workspace(db, training_run_id)
    row = db.scalar(select(models.EvaluationDriftBucketResult).join(models.EvaluationDriftCalculation).where(
        models.EvaluationDriftBucketResult.id == result_id, models.EvaluationDriftCalculation.workspace_id == workspace.id))
    if row is None: raise LookupError("Drift bucket result not found.")
    calculation = db.get(models.EvaluationDriftCalculation, row.calculation_id)
    row.included = included
    included_values = [b.normalized_drift for b in calculation.results if b.included and b.status == "ready" and b.normalized_drift is not None]
    calculation.drift_mean = float(np.mean(included_values)) if included_values else None
    calculation.drift_max = float(max(included_values)) if included_values else None
    _reaggregate(db, workspace); db.commit(); return get_summary(db, training_run_id)
