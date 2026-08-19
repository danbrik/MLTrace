from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import uuid
from datetime import UTC, datetime
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.continuity import continuity_segments, source_group
from app.schemas import (
    EvaluationLabelCsvImportRequest,
    EvaluationLabelCsvPreviewRead,
    EvaluationLabelCsvPreviewRequest,
    EvaluationLabelEventInput,
    EvaluationLabelEventRead,
    EvaluationLabelSetCreate,
    EvaluationLabelSetRead,
    EvaluationProfileCreate,
    EvaluationScorePreviewPoint,
    EvaluationScorePreviewRead,
    ModelEvaluationCreate,
    ModelEvaluationDuplicateRequest,
    ModelEvaluationRead,
    ModelEvaluationUpdate,
)


PROFILE_FIELDS = (
    "normal_window_duration_seconds",
    "normal_window_buffer_seconds",
    "drift_window_seconds",
    "false_alarm_horizon_seconds",
    "anticipation_seconds",
    "epsilon",
)
PROFILE_STAGE_FIELDS = {
    "separation": {"normal_window_duration_seconds", "normal_window_buffer_seconds", "epsilon"},
    "drift": {"drift_window_seconds", "epsilon"},
    "detection": {"false_alarm_horizon_seconds", "anticipation_seconds"},
}
STAGES = ("separation", "drift", "detection")
SCORE_COLUMNS = {
    "score": models.TestingRunResult.score,
    "full_mse": models.TestingRunResult.full_mse,
    "roi_mse": models.TestingRunResult.roi_mse,
}
CSV_COLUMNS = (
    "event_id",
    "type",
    "name",
    "category",
    "start_timestamp",
    "end_timestamp",
    "notes",
)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _clean_name(value: str, label: str = "Name") -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{label} is required.")
    return clean


def _dataset_local_timestamp(value: datetime) -> datetime:
    """Normalize API timestamps to MLTrace's dataset-local wall-clock time.

    Offsets are intentionally discarded without UTC conversion: image timestamps
    and every existing MLTrace range are timezone-naive local values.
    """
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value.replace(tzinfo=None)
    return value


def _hash_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _profile_snapshot(profile: models.EvaluationProfile) -> dict:
    return {field: float(getattr(profile, field)) for field in PROFILE_FIELDS}


def _event_dict(event: models.EvaluationLabelEvent) -> dict:
    return {
        "event_id": event.event_id,
        "type": event.type,
        "name": event.name,
        "category": event.category,
        "start_timestamp": event.start_timestamp.isoformat(),
        "end_timestamp": event.end_timestamp.isoformat(),
        "notes": event.notes,
    }


def _label_snapshot(label_set: models.EvaluationLabelSet) -> dict:
    return {
        "label_set_id": label_set.id,
        "training_dataset_id": label_set.training_dataset_id,
        "name": label_set.name,
        "version": label_set.version,
        "events": [_event_dict(event) for event in label_set.events],
    }


def _serialize_label_set(label_set: models.EvaluationLabelSet) -> EvaluationLabelSetRead:
    return EvaluationLabelSetRead(
        id=label_set.id,
        training_dataset_id=label_set.training_dataset_id,
        name=label_set.name,
        description=label_set.description,
        version=label_set.version,
        events=[EvaluationLabelEventRead.model_validate(event) for event in label_set.events],
        categories=sorted(
            {
                event.category
                for event in label_set.events
                if event.type == "target" and event.category
            }
        ),
        created_at=label_set.created_at,
        updated_at=label_set.updated_at,
    )


def _serialize_evaluation(row: models.ModelEvaluation) -> ModelEvaluationRead:
    values = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    values["selected_categories"] = values["selected_categories"] or []
    values["normal_window_overrides"] = values["normal_window_overrides"] or {}
    values["profile_overrides"] = values["profile_overrides"] or {}
    values["warnings"] = values["warnings"] or []
    return ModelEvaluationRead.model_validate(values)


def _profile_name_exists(db: Session, name: str, exclude_id: int | None = None) -> bool:
    query = select(models.EvaluationProfile.id).where(
        func.lower(models.EvaluationProfile.name) == name.casefold()
    )
    if exclude_id is not None:
        query = query.where(models.EvaluationProfile.id != exclude_id)
    return db.scalar(query) is not None


def list_profiles(db: Session) -> list[models.EvaluationProfile]:
    return list(db.scalars(select(models.EvaluationProfile).order_by(models.EvaluationProfile.name)))


def get_profile(db: Session, profile_id: int) -> models.EvaluationProfile | None:
    return db.get(models.EvaluationProfile, profile_id)


def create_profile(db: Session, payload: EvaluationProfileCreate) -> models.EvaluationProfile:
    name = _clean_name(payload.name, "Profile name")
    if _profile_name_exists(db, name):
        raise ValueError("An evaluation profile with this name already exists.")
    values = payload.model_dump()
    values["name"] = name
    row = models.EvaluationProfile(**values)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _mark_stale(row: models.ModelEvaluation, stages: Iterable[str]) -> None:
    for stage in stages:
        status_field = f"{stage}_status"
        if getattr(row, status_field) != "not_calculated":
            setattr(row, status_field, "stale")
        setattr(row, f"{stage}_error", None)


def update_profile(
    db: Session, profile_id: int, payload: EvaluationProfileCreate
) -> models.EvaluationProfile | None:
    row = db.get(models.EvaluationProfile, profile_id)
    if row is None:
        return None
    name = _clean_name(payload.name, "Profile name")
    if _profile_name_exists(db, name, profile_id):
        raise ValueError("An evaluation profile with this name already exists.")
    values = payload.model_dump()
    values["name"] = name
    changed = {field for field, value in values.items() if getattr(row, field) != value}
    for field, value in values.items():
        setattr(row, field, value)
    if changed & set(PROFILE_FIELDS):
        for evaluation in db.scalars(
            select(models.ModelEvaluation).where(
                models.ModelEvaluation.profile_id == profile_id,
                models.ModelEvaluation.status == "draft",
            )
        ):
            before = {
                stage: _stage_config_payload(evaluation, stage) for stage in STAGES
            }
            evaluation.profile_snapshot = _profile_snapshot(row)
            affected = {
                stage
                for stage in STAGES
                if before[stage] != _stage_config_payload(evaluation, stage)
            }
            _mark_stale(evaluation, affected)
            _refresh_config_signature(evaluation)
    db.commit()
    db.refresh(row)
    return row


def delete_profile(db: Session, profile_id: int) -> bool:
    row = db.get(models.EvaluationProfile, profile_id)
    if row is None:
        return False
    references = db.scalar(
        select(func.count(models.ModelEvaluation.id)).where(
            models.ModelEvaluation.profile_id == profile_id
        )
    ) or 0
    if references:
        raise ValueError(
            "Evaluation profile is used by saved evaluations. Delete or duplicate those evaluations first."
        )
    db.delete(row)
    db.commit()
    return True


def _load_label_set(db: Session, label_set_id: int) -> models.EvaluationLabelSet | None:
    return db.scalar(
        select(models.EvaluationLabelSet)
        .where(models.EvaluationLabelSet.id == label_set_id)
        .options(selectinload(models.EvaluationLabelSet.events))
    )


def list_label_sets(
    db: Session, *, training_dataset_id: int | None = None
) -> list[EvaluationLabelSetRead]:
    query = select(models.EvaluationLabelSet).options(
        selectinload(models.EvaluationLabelSet.events)
    )
    if training_dataset_id is not None:
        query = query.where(models.EvaluationLabelSet.training_dataset_id == training_dataset_id)
    query = query.order_by(models.EvaluationLabelSet.updated_at.desc(), models.EvaluationLabelSet.id.desc())
    return [_serialize_label_set(row) for row in db.scalars(query).all()]


def get_label_set(db: Session, label_set_id: int) -> EvaluationLabelSetRead | None:
    row = _load_label_set(db, label_set_id)
    return _serialize_label_set(row) if row else None


def _label_name_exists(
    db: Session, training_dataset_id: int, name: str, exclude_id: int | None = None
) -> bool:
    query = select(models.EvaluationLabelSet.id).where(
        models.EvaluationLabelSet.training_dataset_id == training_dataset_id,
        func.lower(models.EvaluationLabelSet.name) == name.casefold(),
    )
    if exclude_id is not None:
        query = query.where(models.EvaluationLabelSet.id != exclude_id)
    return db.scalar(query) is not None


def _dataset_rule_ranges(db: Session, training_dataset_id: int) -> list[tuple[datetime, datetime]]:
    rules = db.execute(
        select(
            models.TrainingDatasetRule.start_timestamp,
            models.TrainingDatasetRule.end_timestamp,
        )
        .where(models.TrainingDatasetRule.training_dataset_id == training_dataset_id)
        .order_by(models.TrainingDatasetRule.start_timestamp)
    ).all()
    merged: list[list[datetime]] = []
    for start, end in rules:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def _validate_events(
    db: Session,
    training_dataset_id: int,
    events: list[EvaluationLabelEventInput],
) -> list[dict]:
    if db.get(models.TrainingDataset, training_dataset_id) is None:
        raise ValueError("Inference dataset not found.")
    allowed_ranges = _dataset_rule_ranges(db, training_dataset_id)
    if not allowed_ranges:
        raise ValueError("The inference dataset has no timestamp rules.")
    normalized: list[dict] = []
    seen_ids: set[str] = set()
    for event in events:
        event.start_timestamp = _dataset_local_timestamp(event.start_timestamp)
        event.end_timestamp = _dataset_local_timestamp(event.end_timestamp)
        event_id = str(uuid.uuid4()) if event.event_id is None else event.event_id.strip()
        if not event_id:
            raise ValueError("event_id must not be blank.")
        if event_id in seen_ids:
            raise ValueError(f"Duplicate event_id '{event_id}'.")
        seen_ids.add(event_id)
        if not any(
            start <= event.start_timestamp and event.end_timestamp <= end
            for start, end in allowed_ranges
        ):
            raise ValueError(
                f"Event '{event_id}' is outside the inference dataset timestamp rules."
            )
        normalized.append(
            {
                "event_id": event_id,
                "type": event.type,
                "name": event.name.strip() if event.name else None,
                "category": event.category.strip() if event.category else None,
                "start_timestamp": event.start_timestamp,
                "end_timestamp": event.end_timestamp,
                "notes": event.notes.strip() if event.notes else None,
            }
        )
    targets = sorted(
        (event for event in normalized if event["type"] == "target"),
        key=lambda event: (event["start_timestamp"], event["end_timestamp"]),
    )
    for previous, current in zip(targets, targets[1:]):
        if current["start_timestamp"] <= previous["end_timestamp"]:
            raise ValueError(
                f"Target events '{previous['event_id']}' and '{current['event_id']}' overlap."
            )
    return normalized


def _replace_events(
    label_set: models.EvaluationLabelSet, normalized: list[dict]
) -> None:
    existing = {event.event_id: event for event in label_set.events}
    replacement: list[models.EvaluationLabelEvent] = []
    for values in normalized:
        event = existing.get(values["event_id"])
        if event is None:
            event = models.EvaluationLabelEvent(**values)
        else:
            for field, value in values.items():
                setattr(event, field, value)
        replacement.append(event)
    label_set.events[:] = replacement


def _touch_label_dependents(db: Session, label_set: models.EvaluationLabelSet) -> None:
    snapshot = _label_snapshot(label_set)
    for evaluation in db.scalars(
        select(models.ModelEvaluation).where(
            models.ModelEvaluation.label_set_id == label_set.id,
            models.ModelEvaluation.status == "draft",
        )
    ):
        evaluation.label_snapshot = snapshot
        _mark_stale(evaluation, STAGES)
        _refresh_config_signature(evaluation)


def create_label_set(db: Session, payload: EvaluationLabelSetCreate) -> EvaluationLabelSetRead:
    name = _clean_name(payload.name, "Label-set name")
    if _label_name_exists(db, payload.training_dataset_id, name):
        raise ValueError("A label set with this name already exists for the inference dataset.")
    normalized = _validate_events(db, payload.training_dataset_id, payload.events)
    row = models.EvaluationLabelSet(
        training_dataset_id=payload.training_dataset_id,
        name=name,
        description=payload.description,
    )
    _replace_events(row, normalized)
    db.add(row)
    db.commit()
    row = _load_label_set(db, row.id)
    assert row is not None
    return _serialize_label_set(row)


def update_label_set(
    db: Session, label_set_id: int, payload: EvaluationLabelSetCreate
) -> EvaluationLabelSetRead | None:
    row = _load_label_set(db, label_set_id)
    if row is None:
        return None
    if payload.training_dataset_id != row.training_dataset_id:
        references = db.scalar(
            select(func.count(models.ModelEvaluation.id)).where(
                models.ModelEvaluation.label_set_id == label_set_id
            )
        ) or 0
        if references:
            raise ValueError(
                "A label set already used by saved evaluations cannot be moved to another inference dataset."
            )
    name = _clean_name(payload.name, "Label-set name")
    if _label_name_exists(db, payload.training_dataset_id, name, label_set_id):
        raise ValueError("A label set with this name already exists for the inference dataset.")
    normalized = _validate_events(db, payload.training_dataset_id, payload.events)
    current_events = sorted(
        (
            {
                "event_id": event.event_id,
                "type": event.type,
                "name": event.name,
                "category": event.category,
                "start_timestamp": event.start_timestamp,
                "end_timestamp": event.end_timestamp,
                "notes": event.notes,
            }
            for event in row.events
        ),
        key=lambda event: event["event_id"],
    )
    incoming_events = sorted(normalized, key=lambda event: event["event_id"])
    if (
        payload.training_dataset_id == row.training_dataset_id
        and name == row.name
        and payload.description == row.description
        and incoming_events == current_events
    ):
        return _serialize_label_set(row)
    row.training_dataset_id = payload.training_dataset_id
    row.name = name
    row.description = payload.description
    row.version += 1
    _replace_events(row, normalized)
    db.flush()
    _touch_label_dependents(db, row)
    db.commit()
    row = _load_label_set(db, label_set_id)
    assert row is not None
    return _serialize_label_set(row)


def add_label_event(
    db: Session, label_set_id: int, payload: EvaluationLabelEventInput
) -> EvaluationLabelSetRead | None:
    row = _load_label_set(db, label_set_id)
    if row is None:
        return None
    inputs = [
        EvaluationLabelEventInput(**_event_dict(event)) for event in row.events
    ] + [payload]
    normalized = _validate_events(db, row.training_dataset_id, inputs)
    row.version += 1
    _replace_events(row, normalized)
    db.flush()
    _touch_label_dependents(db, row)
    db.commit()
    row = _load_label_set(db, label_set_id)
    assert row is not None
    return _serialize_label_set(row)


def update_label_event(
    db: Session,
    label_set_id: int,
    event_id: str,
    payload: EvaluationLabelEventInput,
) -> EvaluationLabelSetRead | None:
    row = _load_label_set(db, label_set_id)
    if row is None:
        return None
    if not any(event.event_id == event_id for event in row.events):
        raise KeyError(event_id)
    if payload.event_id is not None and payload.event_id != event_id:
        raise ValueError("event_id is immutable and must match the event id in the route.")
    values = []
    for event in row.events:
        if event.event_id == event_id:
            replacement = payload.model_copy(update={"event_id": event_id})
            values.append(replacement)
        else:
            values.append(EvaluationLabelEventInput(**_event_dict(event)))
    normalized = _validate_events(db, row.training_dataset_id, values)
    row.version += 1
    _replace_events(row, normalized)
    db.flush()
    _touch_label_dependents(db, row)
    db.commit()
    row = _load_label_set(db, label_set_id)
    assert row is not None
    return _serialize_label_set(row)


def delete_label_event(db: Session, label_set_id: int, event_id: str) -> bool:
    row = _load_label_set(db, label_set_id)
    if row is None:
        return False
    event = next((item for item in row.events if item.event_id == event_id), None)
    if event is None:
        return False
    row.events.remove(event)
    row.version += 1
    db.flush()
    _touch_label_dependents(db, row)
    db.commit()
    return True


def delete_label_set(db: Session, label_set_id: int) -> bool:
    row = _load_label_set(db, label_set_id)
    if row is None:
        return False
    references = db.scalar(
        select(func.count(models.ModelEvaluation.id)).where(
            models.ModelEvaluation.label_set_id == label_set_id
        )
    ) or 0
    if references:
        raise ValueError(
            "Evaluation label set is used by saved evaluations. Delete or duplicate those evaluations first."
        )
    db.delete(row)
    db.commit()
    return True


def preview_label_csv(
    db: Session, payload: EvaluationLabelCsvPreviewRequest
) -> EvaluationLabelCsvPreviewRead:
    errors: list[dict] = []
    events: list[EvaluationLabelEventInput] = []
    try:
        reader = csv.DictReader(io.StringIO(payload.csv_text.lstrip("\ufeff")))
        missing = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            return EvaluationLabelCsvPreviewRead(
                valid=False,
                errors=[{"row": 1, "message": f"Missing CSV columns: {', '.join(missing)}."}],
            )
        for row_number, raw in enumerate(reader, start=2):
            try:
                events.append(
                    EvaluationLabelEventInput(
                        event_id=(raw.get("event_id") or "").strip() or None,
                        type=(raw.get("type") or "").strip(),
                        name=(raw.get("name") or "").strip() or None,
                        category=(raw.get("category") or "").strip() or None,
                        start_timestamp=datetime.fromisoformat(
                            (raw.get("start_timestamp") or "").strip()
                        ),
                        end_timestamp=datetime.fromisoformat(
                            (raw.get("end_timestamp") or "").strip()
                        ),
                        notes=(raw.get("notes") or "").strip() or None,
                    )
                )
            except (ValueError, TypeError) as exc:
                errors.append({"row": row_number, "message": str(exc)})
    except csv.Error as exc:
        errors.append({"row": 1, "message": f"Invalid CSV: {exc}"})

    normalized: list[dict] = []
    if not errors:
        try:
            normalized = _validate_events(db, payload.training_dataset_id, events)
        except ValueError as exc:
            errors.append({"row": 0, "message": str(exc)})
    return EvaluationLabelCsvPreviewRead(
        valid=not errors,
        events=[EvaluationLabelEventRead(**event) for event in normalized],
        errors=errors,
    )


def import_label_csv(
    db: Session,
    label_set_id: int,
    payload: EvaluationLabelCsvImportRequest,
) -> EvaluationLabelSetRead | None:
    row = _load_label_set(db, label_set_id)
    if row is None:
        return None
    if payload.training_dataset_id != row.training_dataset_id:
        raise ValueError("CSV inference dataset does not match the label set.")
    preview = preview_label_csv(db, payload)
    if not preview.valid:
        raise ValueError("CSV validation failed: " + "; ".join(error.message for error in preview.errors))
    imported = [EvaluationLabelEventInput(**event.model_dump()) for event in preview.events]
    if payload.mode == "append":
        imported = [EvaluationLabelEventInput(**_event_dict(event)) for event in row.events] + imported
    normalized = _validate_events(db, row.training_dataset_id, imported)
    row.version += 1
    _replace_events(row, normalized)
    db.flush()
    _touch_label_dependents(db, row)
    db.commit()
    row = _load_label_set(db, label_set_id)
    assert row is not None
    return _serialize_label_set(row)


def export_label_csv(db: Session, label_set_id: int) -> str | None:
    row = _load_label_set(db, label_set_id)
    if row is None:
        return None
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for event in row.events:
        writer.writerow(_event_dict(event))
    return output.getvalue()


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _source_row_snapshot(db: Session, run: models.TestingRun | None) -> dict | None:
    if run is None:
        return None
    result_count, latest_result_id, latest_result_at = db.execute(
        select(
            func.count(models.TestingRunResult.id),
            func.max(models.TestingRunResult.id),
            func.max(models.TestingRunResult.created_at),
        ).where(models.TestingRunResult.testing_run_id == run.id)
    ).one()
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "training_run_id": run.training_run_id,
        "training_dataset_id": run.training_dataset_id,
        "training_dataset_name": run.training_dataset_name,
        "artifact_signature": run.artifact_signature,
        "artifact_path": run.artifact_path,
        "preprocessing_pipeline_name": run.preprocessing_pipeline_name,
        "method_type": run.method_type,
        "method_family": run.method_family,
        "roi_geometry": run.roi_geometry,
        "inference_config": run.inference_config,
        "run_updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "result_count": int(result_count or 0),
        "latest_result_id": latest_result_id,
        "latest_result_at": latest_result_at.isoformat() if latest_result_at else None,
    }


def _update_source_snapshot(db: Session, row: models.ModelEvaluation) -> None:
    row.source_snapshot = {
        "evaluation": _source_row_snapshot(
            db,
            db.get(models.TestingRun, row.evaluation_testing_run_id)
            if row.evaluation_testing_run_id is not None
            else None
        ),
        "reference": _source_row_snapshot(
            db,
            db.get(models.TestingRun, row.reference_testing_run_id)
            if row.reference_testing_run_id is not None
            else None
        ),
        "calibration": _source_row_snapshot(
            db,
            db.get(models.TestingRun, row.calibration_testing_run_id)
            if row.calibration_testing_run_id is not None
            else None
        ),
    }


def _sync_source_revisions(db: Session, row: models.ModelEvaluation) -> bool:
    before = row.source_snapshot or {}
    previous = {key: before.get(key) for key in ("evaluation", "reference", "calibration")}
    _update_source_snapshot(db, row)
    current = row.source_snapshot or {}
    changed = False
    role_stages = {
        "evaluation": STAGES,
        "reference": ("drift",),
        "calibration": ("detection",),
    }
    for role, stages in role_stages.items():
        # An empty snapshot is expected on a brand-new draft and is not stale.
        if previous[role] is not None and previous[role] != current.get(role):
            _mark_stale(row, stages)
            changed = True
    if changed:
        _refresh_config_signature(row)
    return changed


def _effective_profile(row: models.ModelEvaluation) -> dict:
    values = dict(row.profile_snapshot or {})
    values.update(row.profile_overrides or {})
    return values


def _config_payload(row: models.ModelEvaluation) -> dict:
    return {
        "evaluation_testing_run_id": row.evaluation_testing_run_id,
        "reference_testing_run_id": row.reference_testing_run_id,
        "calibration_testing_run_id": row.calibration_testing_run_id,
        "score_series": row.score_series,
        "evaluation_start_timestamp": row.evaluation_start_timestamp,
        "evaluation_end_timestamp": row.evaluation_end_timestamp,
        "reference_start_timestamp": row.reference_start_timestamp,
        "reference_end_timestamp": row.reference_end_timestamp,
        "calibration_start_timestamp": row.calibration_start_timestamp,
        "calibration_end_timestamp": row.calibration_end_timestamp,
        "selected_categories": row.selected_categories or [],
        "normal_window_overrides": row.normal_window_overrides or {},
        "profile": _effective_profile(row),
        "label_version": (row.label_snapshot or {}).get("version"),
        "sources": row.source_snapshot,
    }


def _stage_config_payload(row: models.ModelEvaluation, stage: str) -> dict:
    common = {
        "evaluation_testing_run_id": row.evaluation_testing_run_id,
        "score_series": row.score_series,
        "evaluation_start_timestamp": row.evaluation_start_timestamp,
        "evaluation_end_timestamp": row.evaluation_end_timestamp,
        "selected_categories": row.selected_categories or [],
        "label_version": (row.label_snapshot or {}).get("version"),
    }
    profile = _effective_profile(row)
    if stage == "separation":
        common.update(
            {
                "normal_window_duration_seconds": profile.get("normal_window_duration_seconds"),
                "normal_window_buffer_seconds": profile.get("normal_window_buffer_seconds"),
                "epsilon": profile.get("epsilon"),
                "normal_window_overrides": row.normal_window_overrides or {},
            }
        )
    elif stage == "drift":
        common.update(
            {
                "reference_testing_run_id": row.reference_testing_run_id,
                "reference_start_timestamp": row.reference_start_timestamp,
                "reference_end_timestamp": row.reference_end_timestamp,
                "drift_window_seconds": profile.get("drift_window_seconds"),
                "epsilon": profile.get("epsilon"),
            }
        )
    elif stage == "detection":
        common.update(
            {
                "calibration_testing_run_id": row.calibration_testing_run_id,
                "calibration_start_timestamp": row.calibration_start_timestamp,
                "calibration_end_timestamp": row.calibration_end_timestamp,
                "false_alarm_horizon_seconds": profile.get("false_alarm_horizon_seconds"),
                "anticipation_seconds": profile.get("anticipation_seconds"),
            }
        )
    return common


def _refresh_config_signature(row: models.ModelEvaluation) -> None:
    row.config_signature = _hash_json(_config_payload(row))


def _validate_profile_overrides(values: dict[str, float]) -> None:
    unknown = set(values) - set(PROFILE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown profile override(s): {', '.join(sorted(unknown))}.")
    for key, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"Profile override '{key}' must be finite.")
        if key in {"normal_window_buffer_seconds", "anticipation_seconds"}:
            if value < 0:
                raise ValueError(f"Profile override '{key}' must be non-negative.")
        elif value <= 0:
            raise ValueError(f"Profile override '{key}' must be greater than zero.")


def _validate_normal_window_overrides(values: dict) -> None:
    for event_id, interval in values.items():
        if not isinstance(interval, dict):
            raise ValueError(f"Normal-window override '{event_id}' must be a time range.")
        start = interval.get("start_timestamp")
        end = interval.get("end_timestamp")
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValueError(
                f"Normal-window override '{event_id}' requires start_timestamp and end_timestamp."
            )
        start = _dataset_local_timestamp(start)
        end = _dataset_local_timestamp(end)
        if end <= start:
            raise ValueError(f"Normal-window override '{event_id}' end must be after start.")
        interval["start_timestamp"] = start.isoformat()
        interval["end_timestamp"] = end.isoformat()


def _assert_evaluation_references(db: Session, row: models.ModelEvaluation) -> None:
    for field in (
        "evaluation_testing_run_id",
        "reference_testing_run_id",
        "calibration_testing_run_id",
    ):
        value = getattr(row, field)
        if value is not None and db.get(models.TestingRun, value) is None:
            raise ValueError(f"Testing run #{value} not found.")
    if row.profile_id is not None:
        profile = db.get(models.EvaluationProfile, row.profile_id)
        if profile is None:
            raise ValueError("Evaluation profile not found.")
        row.profile_snapshot = _profile_snapshot(profile)
    else:
        row.profile_snapshot = None
    if row.label_set_id is not None:
        label_set = _load_label_set(db, row.label_set_id)
        if label_set is None:
            raise ValueError("Evaluation label set not found.")
        if row.evaluation_testing_run_id is not None:
            run = db.get(models.TestingRun, row.evaluation_testing_run_id)
            assert run is not None
            if label_set.training_dataset_id != run.training_dataset_id:
                raise ValueError("The label set belongs to a different inference dataset.")
        row.label_snapshot = _label_snapshot(label_set)
    else:
        row.label_snapshot = None
    _validate_profile_overrides(row.profile_overrides or {})
    _validate_normal_window_overrides(row.normal_window_overrides or {})
    for role in ("evaluation", "reference", "calibration"):
        start = getattr(row, f"{role}_start_timestamp")
        end = getattr(row, f"{role}_end_timestamp")
        if start is not None:
            start = _dataset_local_timestamp(start)
            setattr(row, f"{role}_start_timestamp", start)
        if end is not None:
            end = _dataset_local_timestamp(end)
            setattr(row, f"{role}_end_timestamp", end)
        if (start is None) != (end is None):
            continue
        if start is not None and end <= start:
            raise ValueError(f"{role.title()} end timestamp must be after its start timestamp.")
    _assert_disjoint_same_run_ranges(row)


def _assert_disjoint_same_run_ranges(row: models.ModelEvaluation) -> None:
    roles = ("evaluation", "reference", "calibration")
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if getattr(row, f"{left}_testing_run_id") != getattr(
                row, f"{right}_testing_run_id"
            ):
                continue
            run_id = getattr(row, f"{left}_testing_run_id")
            if run_id is None:
                continue
            left_start = getattr(row, f"{left}_start_timestamp")
            left_end = getattr(row, f"{left}_end_timestamp")
            right_start = getattr(row, f"{right}_start_timestamp")
            right_end = getattr(row, f"{right}_end_timestamp")
            if None in (left_start, left_end, right_start, right_end):
                continue
            if left_start <= right_end and right_start <= left_end:
                raise ValueError(
                    f"{left.title()} and {right} ranges must be disjoint when they use the same run."
                )


def create_evaluation(db: Session, payload: ModelEvaluationCreate) -> ModelEvaluationRead:
    values = payload.model_dump()
    values["name"] = _clean_name(payload.name, "Evaluation name")
    values["selected_categories"] = values.get("selected_categories") or []
    values["normal_window_overrides"] = _json_value(values.get("normal_window_overrides") or {})
    values["profile_overrides"] = values.get("profile_overrides") or {}
    row = models.ModelEvaluation(**values)
    _assert_evaluation_references(db, row)
    _update_source_snapshot(db, row)
    _refresh_config_signature(row)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _serialize_evaluation(row)


def list_evaluations(
    db: Session,
    *,
    status: str | None = None,
    stale: bool | None = None,
    category: str | None = None,
    score_series: str | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    search: str | None = None,
) -> list[ModelEvaluationRead]:
    query = select(models.ModelEvaluation)
    if status:
        query = query.where(models.ModelEvaluation.status == status)
    if stale is True:
        query = query.where(
            (models.ModelEvaluation.separation_status == "stale")
            | (models.ModelEvaluation.drift_status == "stale")
            | (models.ModelEvaluation.detection_status == "stale")
        )
    elif stale is False:
        query = query.where(
            models.ModelEvaluation.separation_status != "stale",
            models.ModelEvaluation.drift_status != "stale",
            models.ModelEvaluation.detection_status != "stale",
        )
    if score_series:
        query = query.where(models.ModelEvaluation.score_series == score_series)
    if created_from:
        query = query.where(models.ModelEvaluation.created_at >= created_from)
    if created_to:
        query = query.where(models.ModelEvaluation.created_at <= created_to)
    if search:
        query = query.where(models.ModelEvaluation.name.ilike(f"%{search.strip()}%"))
    rows = db.scalars(
        query.order_by(models.ModelEvaluation.updated_at.desc(), models.ModelEvaluation.id.desc())
    ).all()
    source_changed = False
    for row in rows:
        if row.status == "draft":
            source_changed = _sync_source_revisions(db, row) or source_changed
    if source_changed:
        db.commit()
    serialized = [_serialize_evaluation(row) for row in rows]
    if category:
        serialized = [
            row
            for row in serialized
            if category in row.selected_categories
            or category in ((row.label_snapshot or {}).get("categories") or [])
            or any(
                event.get("category") == category
                for event in ((row.label_snapshot or {}).get("events") or [])
            )
        ]
    return serialized


def get_evaluation(db: Session, evaluation_id: int) -> ModelEvaluationRead | None:
    row = db.get(models.ModelEvaluation, evaluation_id)
    if row is not None and row.status == "draft" and _sync_source_revisions(db, row):
        db.commit()
        db.refresh(row)
    return _serialize_evaluation(row) if row else None


def _changed_stages(before: dict, after: dict) -> set[str]:
    stages: set[str] = set()
    all_fields = {
        "evaluation_testing_run_id",
        "score_series",
        "evaluation_start_timestamp",
        "evaluation_end_timestamp",
        "label_set_id",
        "selected_categories",
    }
    if any(before.get(field) != after.get(field) for field in all_fields):
        stages.update(STAGES)
    if before.get("normal_window_overrides") != after.get("normal_window_overrides"):
        stages.add("separation")
    if any(
        before.get(field) != after.get(field)
        for field in (
            "reference_testing_run_id",
            "reference_start_timestamp",
            "reference_end_timestamp",
        )
    ):
        stages.add("drift")
    if any(
        before.get(field) != after.get(field)
        for field in (
            "calibration_testing_run_id",
            "calibration_start_timestamp",
            "calibration_end_timestamp",
        )
    ):
        stages.add("detection")
    if before.get("profile_id") != after.get("profile_id"):
        stages.update(STAGES)
    old_overrides = before.get("profile_overrides") or {}
    new_overrides = after.get("profile_overrides") or {}
    changed_profile_fields = {
        field for field in PROFILE_FIELDS if old_overrides.get(field) != new_overrides.get(field)
    }
    for stage, fields in PROFILE_STAGE_FIELDS.items():
        if changed_profile_fields & fields:
            stages.add(stage)
    return stages


def update_evaluation(
    db: Session, evaluation_id: int, payload: ModelEvaluationUpdate
) -> ModelEvaluationRead | None:
    row = db.get(models.ModelEvaluation, evaluation_id)
    if row is None:
        return None
    if row.status == "finalized":
        raise ValueError("Finalized evaluations are immutable. Duplicate it to make changes.")
    _sync_source_revisions(db, row)
    before = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    values = payload.model_dump(exclude_unset=True)
    if "name" in values:
        values["name"] = _clean_name(values["name"], "Evaluation name")
    if "normal_window_overrides" in values:
        values["normal_window_overrides"] = _json_value(values["normal_window_overrides"] or {})
    for field, value in values.items():
        setattr(row, field, value)
    _assert_evaluation_references(db, row)
    _update_source_snapshot(db, row)
    after = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    _mark_stale(row, _changed_stages(before, after))
    if "active_quantile" in values:
        if row.detection_status == "current":
            set_active_quantile(db, row, float(values["active_quantile"]))
        else:
            row.active_quantile = float(values["active_quantile"])
    _refresh_config_signature(row)
    db.commit()
    db.refresh(row)
    return _serialize_evaluation(row)


def delete_evaluation(db: Session, evaluation_id: int) -> bool:
    row = db.get(models.ModelEvaluation, evaluation_id)
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def duplicate_evaluation(
    db: Session,
    evaluation_id: int,
    payload: ModelEvaluationDuplicateRequest | None = None,
) -> ModelEvaluationRead | None:
    source = db.get(models.ModelEvaluation, evaluation_id)
    if source is None:
        return None
    copied_fields = (
        "evaluation_testing_run_id",
        "reference_testing_run_id",
        "calibration_testing_run_id",
        "profile_id",
        "label_set_id",
        "score_series",
        "evaluation_start_timestamp",
        "evaluation_end_timestamp",
        "reference_start_timestamp",
        "reference_end_timestamp",
        "calibration_start_timestamp",
        "calibration_end_timestamp",
        "selected_categories",
        "normal_window_overrides",
        "profile_overrides",
        "active_quantile",
    )
    name = payload.name if payload and payload.name else f"{source.name} (Copy)"
    values = {field: getattr(source, field) for field in copied_fields}
    values["name"] = _clean_name(name, "Evaluation name")
    row = models.ModelEvaluation(**values)
    _assert_evaluation_references(db, row)
    _update_source_snapshot(db, row)
    _refresh_config_signature(row)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _serialize_evaluation(row)


def _assert_complete_range(row: models.ModelEvaluation, role: str) -> tuple[int, datetime, datetime]:
    run_id = getattr(row, f"{role}_testing_run_id")
    start = getattr(row, f"{role}_start_timestamp")
    end = getattr(row, f"{role}_end_timestamp")
    if run_id is None or start is None or end is None:
        raise ValueError(f"{role.title()} source and time range are required for this stage.")
    if end <= start:
        raise ValueError(f"{role.title()} end timestamp must be after its start timestamp.")
    return run_id, start, end


def _assert_compatible_runs(
    db: Session, evaluation_run: models.TestingRun, other_run: models.TestingRun
) -> None:
    for run in (evaluation_run, other_run):
        if run.status != "finished":
            raise ValueError(f"Testing run #{run.id} is not finished.")
    if not evaluation_run.artifact_signature or not other_run.artifact_signature:
        raise ValueError(
            "Evaluation source is missing an artifact signature. Rerun inference with the current "
            "MLTrace version before evaluating it."
        )
    if evaluation_run.artifact_signature != other_run.artifact_signature:
        raise ValueError("Evaluation sources use different model artifacts.")
    comparable = (
        "preprocessing_pipeline_name",
        "method_type",
        "method_family",
        "artifact_kind",
    )
    if any(getattr(evaluation_run, field) != getattr(other_run, field) for field in comparable):
        raise ValueError("Evaluation sources have incompatible model or preprocessing semantics.")
    if _hash_json(evaluation_run.roi_geometry) != _hash_json(other_run.roi_geometry):
        raise ValueError("Evaluation sources use different ROI geometry.")
    if _hash_json(evaluation_run.inference_config) != _hash_json(other_run.inference_config):
        raise ValueError("Evaluation sources use different inference/score configuration.")


def _assert_range_available(
    db: Session, run_id: int, start: datetime, end: datetime, score_series: str
) -> None:
    column = SCORE_COLUMNS[score_series]
    minimum, maximum, count, finite_count = db.execute(
        select(
            func.min(models.TestingRunResult.timestamp),
            func.max(models.TestingRunResult.timestamp),
            func.count(models.TestingRunResult.id),
            func.count(column),
        ).where(models.TestingRunResult.testing_run_id == run_id)
    ).one()
    if not count:
        raise ValueError(f"Testing run #{run_id} has no stored score points.")
    if finite_count != count:
        raise ValueError(f"Score series '{score_series}' is missing values in testing run #{run_id}.")
    if start < minimum or end > maximum:
        raise ValueError(
            f"Requested range for testing run #{run_id} is outside its stored score timestamps."
        )


def _range_overlaps(left_start: datetime, left_end: datetime, right_start: datetime, right_end: datetime) -> bool:
    return left_start <= right_end and right_start <= left_end


def _assert_no_training_leakage(
    db: Session,
    run: models.TestingRun,
    start: datetime,
    end: datetime,
) -> None:
    training = db.get(models.TrainingRun, run.training_run_id)
    if training is None:
        return
    training_rules = db.execute(
        select(
            models.TrainingDatasetRule.folder_id,
            models.TrainingDatasetRule.start_timestamp,
            models.TrainingDatasetRule.end_timestamp,
        )
        .join(
            models.TrainingPipelineDataset,
            models.TrainingPipelineDataset.training_dataset_id
            == models.TrainingDatasetRule.training_dataset_id,
        )
        .where(models.TrainingPipelineDataset.training_pipeline_id == training.training_pipeline_id)
    ).all()
    role_rules = db.execute(
        select(
            models.TrainingDatasetRule.folder_id,
            models.TrainingDatasetRule.start_timestamp,
            models.TrainingDatasetRule.end_timestamp,
        ).where(models.TrainingDatasetRule.training_dataset_id == run.training_dataset_id)
    ).all()
    for role_folder, role_start, role_end in role_rules:
        selected_start = max(start, role_start)
        selected_end = min(end, role_end)
        if selected_end < selected_start:
            continue
        for train_folder, train_start, train_end in training_rules:
            if role_folder == train_folder and _range_overlaps(
                selected_start, selected_end, train_start, train_end
            ):
                raise ValueError(
                    f"Testing run #{run.id} range overlaps data used to train its model."
                )


def _load_points(
    db: Session,
    run_id: int,
    start: datetime,
    end: datetime,
    score_series: str,
) -> list[dict]:
    column = SCORE_COLUMNS.get(score_series)
    if column is None:
        raise ValueError(f"Unsupported score series '{score_series}'.")
    rows = db.execute(
        select(
            models.TestingRunResult.id,
            models.TestingRunResult.position,
            models.TestingRunResult.timestamp,
            models.TestingRunResult.image_path,
            column.label("value"),
        )
        .where(
            models.TestingRunResult.testing_run_id == run_id,
            models.TestingRunResult.timestamp >= start,
            models.TestingRunResult.timestamp <= end,
        )
        .order_by(models.TestingRunResult.timestamp, models.TestingRunResult.position)
    ).all()
    if not rows:
        raise ValueError("The selected score range contains no points.")
    for result in rows:
        if result.value is None or not math.isfinite(float(result.value)):
            raise ValueError(f"Score series '{score_series}' contains missing or non-finite values.")
    segments = continuity_segments(
        [result.timestamp for result in rows],
        [source_group(result.image_path) for result in rows],
    )
    return [
        {
            "result_id": result.id,
            "position": result.position,
            "timestamp": result.timestamp,
            "score": float(result.value),
            "continuity_segment": segment,
        }
        for result, segment in zip(rows, segments, strict=True)
    ]


def _selected_events(
    row: models.ModelEvaluation, *, require_targets: bool = True
) -> tuple[list[dict], list[dict]]:
    snapshot = row.label_snapshot
    if not snapshot:
        raise ValueError("A ground-truth label set is required for this stage.")
    selected = set(row.selected_categories or [])
    targets: list[dict] = []
    exclusions: list[dict] = []
    for event in snapshot.get("events") or []:
        interval = {
            "event_id": event["event_id"],
            "name": event.get("name") or "",
            "category": event.get("category") or "",
            "start_timestamp": event["start_timestamp"],
            "end_timestamp": event["end_timestamp"],
        }
        if event.get("type") == "exclusion" or (
            event.get("type") == "target"
            and selected
            and event.get("category") not in selected
        ):
            exclusions.append(
                {
                    "start_timestamp": event["start_timestamp"],
                    "end_timestamp": event["end_timestamp"],
                }
            )
        elif not selected or event.get("category") in selected:
            targets.append(interval)
    if require_targets and not targets:
        raise ValueError("The selected ground-truth categories contain no target events.")
    return targets, exclusions


def _assert_confirmed_normal_source(
    row: models.ModelEvaluation,
    run: models.TestingRun,
    role: str,
    start: datetime,
    end: datetime,
) -> None:
    snapshot = row.label_snapshot or {}
    if snapshot.get("training_dataset_id") != run.training_dataset_id:
        return
    conflicts: list[str] = []
    for event in snapshot.get("events") or []:
        event_start = datetime.fromisoformat(event["start_timestamp"])
        event_end = datetime.fromisoformat(event["end_timestamp"])
        # Label intervals and outer source ranges both use inclusive endpoints.
        if event_start <= end and start <= event_end:
            conflicts.append(event["event_id"])
    if conflicts:
        raise ValueError(
            f"{role.title()} range contains labeled target/exclusion intervals: "
            + ", ".join(conflicts)
            + ". Choose an explicitly normal range."
        )


def _prepare_stage(
    db: Session,
    row: models.ModelEvaluation,
    roles: tuple[str, ...],
    *,
    require_targets: bool = True,
) -> tuple[dict[str, list[dict]], list[dict], list[dict], dict]:
    if row.status == "finalized":
        raise ValueError("Finalized evaluations are immutable.")
    if _sync_source_revisions(db, row):
        db.flush()
    if not row.profile_snapshot:
        raise ValueError("An evaluation profile is required.")
    _assert_disjoint_same_run_ranges(row)
    role_points: dict[str, list[dict]] = {}
    evaluation_run: models.TestingRun | None = None
    for role in roles:
        run_id, start, end = _assert_complete_range(row, role)
        run = db.get(models.TestingRun, run_id)
        if run is None:
            raise ValueError(f"Testing run #{run_id} not found.")
        if run.status != "finished":
            raise ValueError(f"Testing run #{run_id} is not finished.")
        if evaluation_run is None:
            evaluation_run = db.get(models.TestingRun, row.evaluation_testing_run_id)
        assert evaluation_run is not None
        _assert_compatible_runs(db, evaluation_run, run)
        _assert_range_available(db, run_id, start, end, row.score_series)
        _assert_no_training_leakage(db, run, start, end)
        if role in {"reference", "calibration"}:
            _assert_confirmed_normal_source(row, run, role, start, end)
        role_points[role] = _load_points(db, run_id, start, end, row.score_series)
    events, exclusions = _selected_events(row, require_targets=require_targets)
    return role_points, events, exclusions, _effective_profile(row)


def _refresh_warnings(row: models.ModelEvaluation) -> None:
    warnings: list = []
    for stage in STAGES:
        result = getattr(row, f"{stage}_result") or {}
        for warning in result.get("warnings") or []:
            warnings.append({"stage": stage, "warning": warning})
    row.warnings = warnings


def _store_stage_error(db: Session, row: models.ModelEvaluation, stage: str, exc: Exception) -> None:
    setattr(row, f"{stage}_status", "error")
    setattr(row, f"{stage}_error", str(exc))
    code = getattr(exc, "code", "calculation_error")
    details = _json_value(getattr(exc, "details", {}) or {})
    setattr(
        row,
        f"{stage}_result",
        {
            "error": {"code": str(code), "message": str(exc), "details": details},
            # Keep stage-specific diagnostics (notably discarded drift windows)
            # available to the UI even though no scalar result can be produced.
            "diagnostics": details,
            "warnings": [],
        },
    )
    scalar_fields = {
        "separation": ("sep_median", "sep_min"),
        "drift": ("drift_mean", "drift_max"),
        "detection": (
            "event_recall",
            "median_delay_seconds",
            "frame_fpr",
            "false_alarm_rate_t0",
        ),
    }
    for field in scalar_fields[stage]:
        setattr(row, field, None)
    _refresh_warnings(row)
    db.commit()


def calculate_separation(db: Session, evaluation_id: int) -> ModelEvaluationRead | None:
    row = db.get(models.ModelEvaluation, evaluation_id)
    if row is None:
        return None
    try:
        from app.evaluation.metrics import calculate_separation as calculate

        points, events, exclusions, profile = _prepare_stage(db, row, ("evaluation",))
        result = calculate(
            points["evaluation"],
            events,
            evaluation_range={
                "start_timestamp": row.evaluation_start_timestamp,
                "end_timestamp": row.evaluation_end_timestamp,
            },
            normal_window_duration_seconds=profile["normal_window_duration_seconds"],
            normal_window_buffer_seconds=profile["normal_window_buffer_seconds"],
            exclusions=exclusions,
            normal_window_overrides=row.normal_window_overrides or {},
            epsilon=profile["epsilon"],
        )
        payload = result.to_dict()
        row.separation_result = payload
        row.sep_median = float(result.sep_median)
        row.sep_min = float(result.sep_min)
        row.separation_status = "current"
        row.separation_error = None
        row.separation_config_signature = _hash_json(_stage_config_payload(row, "separation"))
        _refresh_warnings(row)
        db.commit()
        db.refresh(row)
        return _serialize_evaluation(row)
    except (ValueError, KeyError) as exc:
        _store_stage_error(db, row, "separation", exc)
        raise ValueError(str(exc)) from exc


def calculate_drift(db: Session, evaluation_id: int) -> ModelEvaluationRead | None:
    row = db.get(models.ModelEvaluation, evaluation_id)
    if row is None:
        return None
    try:
        from app.evaluation.metrics import calculate_drift as calculate

        points, events, exclusions, profile = _prepare_stage(
            db, row, ("evaluation", "reference"), require_targets=False
        )
        result = calculate(
            points["evaluation"],
            points["reference"],
            evaluation_range={
                "start_timestamp": row.evaluation_start_timestamp,
                "end_timestamp": row.evaluation_end_timestamp,
            },
            reference_range={
                "start_timestamp": row.reference_start_timestamp,
                "end_timestamp": row.reference_end_timestamp,
            },
            window_duration_seconds=profile["drift_window_seconds"],
            events=events,
            exclusions=exclusions,
            epsilon=profile["epsilon"],
        )
        payload = result.to_dict()
        row.drift_result = payload
        row.drift_mean = float(result.d_mean)
        row.drift_max = float(result.d_max)
        row.drift_status = "current"
        row.drift_error = None
        row.drift_config_signature = _hash_json(_stage_config_payload(row, "drift"))
        _refresh_warnings(row)
        db.commit()
        db.refresh(row)
        return _serialize_evaluation(row)
    except (ValueError, KeyError) as exc:
        _store_stage_error(db, row, "drift", exc)
        raise ValueError(str(exc)) from exc


def _operating_point(payload: dict, quantile: float) -> dict:
    rows = payload.get("operating_points") or payload.get("quantiles") or []
    if isinstance(rows, dict):
        rows = [dict(value, quantile=float(key)) for key, value in rows.items()]
    for item in rows:
        q = item.get("quantile", item.get("q"))
        if q is not None and math.isclose(float(q), quantile, rel_tol=0.0, abs_tol=1e-10):
            return item
    raise ValueError(f"Detection result does not contain operating point q={quantile}.")


def calculate_detection(db: Session, evaluation_id: int) -> ModelEvaluationRead | None:
    row = db.get(models.ModelEvaluation, evaluation_id)
    if row is None:
        return None
    try:
        from app.evaluation.metrics import calculate_detection as calculate

        points, events, exclusions, profile = _prepare_stage(
            db, row, ("evaluation", "calibration")
        )
        result = calculate(
            points["evaluation"],
            points["calibration"],
            events,
            evaluation_range={
                "start_timestamp": row.evaluation_start_timestamp,
                "end_timestamp": row.evaluation_end_timestamp,
            },
            calibration_range={
                "start_timestamp": row.calibration_start_timestamp,
                "end_timestamp": row.calibration_end_timestamp,
            },
            standard_duration_seconds=profile["false_alarm_horizon_seconds"],
            exclusions=exclusions,
            anticipation_seconds=profile["anticipation_seconds"],
        )
        payload = result.to_dict()
        active = _operating_point(payload, row.active_quantile)
        row.detection_result = payload
        row.event_recall = float(active["event_recall"])
        delay = active.get("median_delay_seconds")
        row.median_delay_seconds = None if delay is None else float(delay)
        row.frame_fpr = float(active["frame_fpr"])
        row.false_alarm_rate_t0 = float(active.get("far_t0", active.get("false_alarm_rate_t0")))
        row.detection_status = "current"
        row.detection_error = None
        row.detection_config_signature = _hash_json(_stage_config_payload(row, "detection"))
        _refresh_warnings(row)
        db.commit()
        db.refresh(row)
        return _serialize_evaluation(row)
    except (ValueError, KeyError, TypeError) as exc:
        _store_stage_error(db, row, "detection", exc)
        raise ValueError(str(exc)) from exc


def set_active_quantile(
    db: Session, row: models.ModelEvaluation, quantile: float
) -> None:
    if not row.detection_result:
        row.active_quantile = quantile
        return
    active = _operating_point(row.detection_result, quantile)
    row.active_quantile = quantile
    row.event_recall = float(active["event_recall"])
    delay = active.get("median_delay_seconds")
    row.median_delay_seconds = None if delay is None else float(delay)
    row.frame_fpr = float(active["frame_fpr"])
    row.false_alarm_rate_t0 = float(active.get("far_t0", active.get("false_alarm_rate_t0")))


def finalize_evaluation(db: Session, evaluation_id: int) -> ModelEvaluationRead | None:
    row = db.get(models.ModelEvaluation, evaluation_id)
    if row is None:
        return None
    if row.status == "finalized":
        return _serialize_evaluation(row)
    if _sync_source_revisions(db, row):
        db.commit()
    if any(getattr(row, f"{stage}_status") != "current" for stage in STAGES):
        raise ValueError("All three evaluation stages must be successfully calculated and current.")
    for stage in STAGES:
        expected = _hash_json(_stage_config_payload(row, stage))
        if getattr(row, f"{stage}_config_signature") != expected:
            _mark_stale(row, (stage,))
            db.commit()
            raise ValueError(f"The {stage} result is stale and must be recalculated.")
    _update_source_snapshot(db, row)
    _refresh_config_signature(row)
    row.status = "finalized"
    row.finalized_at = _utcnow()
    db.commit()
    db.refresh(row)
    return _serialize_evaluation(row)


def _decimate_extrema(points: list[dict], max_points: int) -> list[dict]:
    if len(points) <= max_points:
        return points
    mandatory: set[int] = {0, len(points) - 1}
    for index in range(1, len(points)):
        if points[index]["continuity_segment"] != points[index - 1]["continuity_segment"]:
            mandatory.update((index - 1, index))
    budget = max(2, max_points - len(mandatory))
    bucket_count = max(1, budget // 2)
    for bucket in range(bucket_count):
        start = bucket * len(points) // bucket_count
        end = max(start + 1, (bucket + 1) * len(points) // bucket_count)
        indexes = range(start, min(end, len(points)))
        mandatory.add(min(indexes, key=lambda idx: points[idx]["score"]))
        mandatory.add(max(indexes, key=lambda idx: points[idx]["score"]))
    return [points[index] for index in sorted(mandatory)]


def score_preview(
    db: Session,
    run_id: int,
    *,
    score_series: str = "score",
    start_timestamp: datetime | None = None,
    end_timestamp: datetime | None = None,
    max_points: int = 8000,
) -> EvaluationScorePreviewRead | None:
    run = db.get(models.TestingRun, run_id)
    if run is None:
        return None
    if score_series not in SCORE_COLUMNS:
        raise ValueError(f"Unsupported score series '{score_series}'.")
    bounds = db.execute(
        select(
            func.min(models.TestingRunResult.timestamp),
            func.max(models.TestingRunResult.timestamp),
        ).where(models.TestingRunResult.testing_run_id == run_id)
    ).one()
    if bounds[0] is None or bounds[1] is None:
        return EvaluationScorePreviewRead(
            testing_run_id=run_id,
            score_series=score_series,
            start_timestamp=None,
            end_timestamp=None,
            total=0,
            decimated=False,
            points=[],
        )
    start = start_timestamp or bounds[0]
    end = end_timestamp or bounds[1]
    start = _dataset_local_timestamp(start)
    end = _dataset_local_timestamp(end)
    if end < start:
        raise ValueError("Preview end timestamp must not be before start timestamp.")
    points = _load_points(db, run_id, start, end, score_series)
    total = len(points)
    selected = _decimate_extrema(points, max(100, min(20000, max_points)))
    return EvaluationScorePreviewRead(
        testing_run_id=run_id,
        score_series=score_series,
        start_timestamp=start,
        end_timestamp=end,
        total=total,
        decimated=len(selected) < total,
        points=[
            EvaluationScorePreviewPoint(
                result_id=point["result_id"],
                position=point["position"],
                timestamp=point["timestamp"],
                value=point["score"],
                continuity_segment=point["continuity_segment"],
            )
            for point in selected
        ],
    )
