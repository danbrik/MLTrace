"""Data Manager service: generic listing/filtering/detail/cascade-delete over
the declarative :mod:`app.registry.specs` entity registry."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app import models
from app.registry.specs import DELETE_ORDER, Dependent, ENTITY_SPECS, EntitySpec


class RegistryConflict(Exception):
    """Deletion refused: dependents present (without cascade) or blocked jobs."""


def _spec(entity_type: str) -> EntitySpec:
    spec = ENTITY_SPECS.get(entity_type)
    if spec is None:
        raise ValueError(f"Unknown entity type: {entity_type}")
    return spec


# -- used-id sets (drive the used/unused filter cheaply, one query per edge) ----


def _distinct_ids(db: Session, column) -> set[int]:
    return {int(v) for v in db.scalars(select(column).where(column.is_not(None)).distinct()) if v is not None}


def _used_ids(db: Session, entity_type: str) -> set[int]:
    m = models
    if entity_type == "dataset":
        rows = db.scalars(
            select(m.DatasetFolder.dataset_id)
            .join(m.TrainingDatasetRule, m.TrainingDatasetRule.folder_id == m.DatasetFolder.id)
            .distinct()
        )
        return {int(v) for v in rows}
    columns = {
        "training_dataset": [
            m.TrainingPipelineDataset.training_dataset_id,
            m.TestingRun.training_dataset_id,
            m.InspectRun.training_dataset_id,
            m.OptimizationStudy.normal_train_dataset_id,
            m.OptimizationStudy.normal_validation_dataset_id,
            m.OptimizationStudy.anomaly_validation_dataset_id,
            m.OptimizationStudy.normal_holdout_dataset_id,
            m.OptimizationStudy.anomaly_holdout_dataset_id,
        ],
        "preprocessing_pipeline": [
            m.TrainingPipeline.preprocessing_pipeline_id,
            m.InspectRun.preprocessing_pipeline_id,
            m.OptimizationStudy.preprocessing_pipeline_id,
        ],
        "method_configuration": [m.TrainingPipeline.method_configuration_id],
        "training_pipeline": [m.TrainingRun.training_pipeline_id],
        "training_run": [m.TestingRun.training_run_id],
        "testing_run": [m.HeatmapRun.testing_run_id, m.HeatmapRangeRun.testing_run_id],
    }.get(entity_type, [])
    used: set[int] = set()
    for column in columns:
        used |= _distinct_ids(db, column)
    return used


# -- summary --------------------------------------------------------------------


def _filter_payload(db: Session, spec: EntitySpec) -> list[dict]:
    payload = []
    for f in spec.filters:
        options = f.options
        if f.kind == "select" and options is None and f.column is not None:
            column = getattr(spec.model, f.column)
            options = sorted(
                {str(v) for v in db.scalars(select(column).where(column.is_not(None)).distinct()) if v is not None}
            )
        payload.append({"key": f.key, "label": f.label, "kind": f.kind, "options": options})
    return payload


def registry_summary(db: Session) -> dict:
    types = []
    for key, spec in ENTITY_SPECS.items():
        count = db.scalar(select(func.count(spec.model.id))) or 0
        types.append({
            "key": key,
            "label": spec.label,
            "count": int(count),
            "filters": _filter_payload(db, spec),
        })
    return {"types": types}


# -- listing ----------------------------------------------------------------------


def _row_payload(spec: EntitySpec, row) -> dict:
    payload = {name: getattr(row, name) for name in spec.list_fields}
    payload["name"] = spec.name_of(row)
    if spec.size_of is not None:
        payload["disk_size_bytes"] = spec.size_of(row)
    return payload


def list_registry_rows(
    db: Session,
    entity_type: str,
    *,
    search: str | None = None,
    filters: dict[str, str] | None = None,
    sort: str | None = None,
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    spec = _spec(entity_type)
    filters = filters or {}
    query = select(spec.model)

    if search:
        needle = f"%{search.strip()}%"
        clauses = [getattr(spec.model, name).ilike(needle) for name in spec.search_fields]
        if clauses:
            query = query.where(or_(*clauses))

    for f in spec.filters:
        value = filters.get(f.key)
        if not value:
            continue
        if f.kind == "select" and f.column is not None:
            column = getattr(spec.model, f.column)
            # Numeric select columns (e.g. output_width) arrive as strings.
            try:
                python_type = column.type.python_type
            except NotImplementedError:
                python_type = str
            try:
                query = query.where(column == python_type(value))
            except (TypeError, ValueError):
                query = query.where(column == value)
        elif f.kind == "daterange" and f.column is not None:
            column = getattr(spec.model, f.column)
            parts = value.split("..", 1)
            if parts[0]:
                query = query.where(column >= datetime.fromisoformat(parts[0]))
            if len(parts) > 1 and parts[1]:
                query = query.where(column <= datetime.fromisoformat(parts[1]))
        elif f.kind == "usage":
            used = _used_ids(db, entity_type)
            if value == "used":
                query = query.where(spec.model.id.in_(used or {-1}))
            elif value == "unused":
                if used:
                    query = query.where(spec.model.id.not_in(used))

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    sort_column = getattr(spec.model, sort, None) if sort else None
    if sort_column is None:
        sort_column = getattr(spec.model, "created_at", spec.model.id)
    query = query.order_by(sort_column.asc() if order == "asc" else sort_column.desc())
    rows = db.scalars(query.limit(max(1, min(200, limit))).offset(max(0, offset))).all()

    payload_rows = []
    for row in rows:
        payload = _row_payload(spec, row)
        payload["usage_count"] = len(spec.dependents(db, row.id)) if spec.dependents else 0
        payload_rows.append(payload)
    return {"total": int(total), "rows": payload_rows}


# -- detail -----------------------------------------------------------------------


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _artifact_payload(db: Session, spec: EntitySpec, row) -> list[dict]:
    payload = []
    for path in (spec.artifacts(db, row) if spec.artifacts else []):
        payload.append({
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "size_bytes": _path_size(path),
        })
    return payload


def get_registry_detail(db: Session, entity_type: str, entity_id: int) -> dict | None:
    spec = _spec(entity_type)
    row = db.get(spec.model, entity_id)
    if row is None:
        return None
    fields = {
        column.name: getattr(row, column.name)
        for column in spec.model.__table__.columns
        if column.name not in spec.detail_exclude
    }
    dependents = spec.dependents(db, entity_id) if spec.dependents else []
    return {
        "entity_type": entity_type,
        "id": entity_id,
        "name": spec.name_of(row),
        "fields": fields,
        "artifacts": _artifact_payload(db, spec, row),
        "dependents": [d.__dict__ for d in dependents],
        "blockers": spec.blockers(row) if spec.blockers else [],
    }


# -- delete preview + cascade delete ----------------------------------------------


def _closure(db: Session, items: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """BFS over dependents: all transitively affected objects.

    Returns mapping (type, id) -> {name, selected, blockers, artifacts}.
    """
    result: dict[tuple[str, int], dict] = {}
    queue: list[tuple[str, int, bool]] = [(t, i, True) for t, i in items]
    while queue:
        entity_type, entity_id, selected = queue.pop(0)
        key = (entity_type, entity_id)
        if key in result:
            if selected:
                result[key]["selected"] = True
            continue
        spec = _spec(entity_type)
        row = db.get(spec.model, entity_id)
        if row is None:
            continue
        result[key] = {
            "name": spec.name_of(row),
            "selected": selected,
            "blockers": spec.blockers(row) if spec.blockers else [],
            "artifact_paths": [str(p) for p in (spec.artifacts(db, row) if spec.artifacts else [])],
        }
        for dep in (spec.dependents(db, entity_id) if spec.dependents else []):
            queue.append((dep.entity_type, dep.id, False))
    return result


def _roi_detach_notes(db: Session, items: list[tuple[str, int]]) -> list[str]:
    notes = []
    for entity_type, entity_id in items:
        if entity_type != "roi":
            continue
        testing = db.scalar(
            select(func.count(models.TestingRun.id)).where(models.TestingRun.roi_id == entity_id)
        ) or 0
        inspect = db.scalar(
            select(func.count(models.InspectRun.id)).where(models.InspectRun.roi_id == entity_id)
        ) or 0
        if testing or inspect:
            notes.append(
                f"ROI #{entity_id}: {testing} inference run(s) and {inspect} inspect run(s) keep their results "
                "but lose the ROI reference (set to null)."
            )
    return notes


def delete_preview(db: Session, items: list[tuple[str, int]]) -> dict:
    closure = _closure(db, items)
    groups: dict[str, list[dict]] = {}
    blockers: list[str] = []
    total_bytes = 0
    files: list[dict] = []
    seen_paths: set[str] = set()
    for (entity_type, entity_id), info in closure.items():
        groups.setdefault(entity_type, []).append({
            "id": entity_id,
            "name": info["name"],
            "selected": info["selected"],
        })
        for reason in info["blockers"]:
            blockers.append(f"{ENTITY_SPECS[entity_type].label} #{entity_id}: {reason}")
        for raw in info["artifact_paths"]:
            if raw in seen_paths:
                continue
            seen_paths.add(raw)
            path = Path(raw)
            size = _path_size(path)
            total_bytes += size
            files.append({"path": raw, "size_bytes": size, "exists": path.exists(), "is_dir": path.is_dir()})

    ordered_groups = [
        {"entity_type": key, "label": ENTITY_SPECS[key].label, "items": sorted(groups[key], key=lambda x: x["id"])}
        for key in DELETE_ORDER
        if key in groups
    ]
    return {
        "groups": ordered_groups,
        "total_objects": sum(len(g["items"]) for g in ordered_groups),
        "dependent_objects": sum(1 for info in closure.values() if not info["selected"]),
        "files": files,
        "total_bytes": total_bytes,
        "blockers": blockers,
        "notes": _roi_detach_notes(db, items),
    }


def delete_entities(db: Session, items: list[tuple[str, int]], *, cascade: bool) -> dict:
    preview = delete_preview(db, items)
    if preview["blockers"]:
        raise RegistryConflict("; ".join(preview["blockers"]))
    if not cascade and preview["dependent_objects"] > 0:
        raise RegistryConflict(
            "Objects are still in use by dependent objects. Enable cascade to delete them together."
        )

    freed = preview["total_bytes"]
    deleted: dict[str, int] = {}
    for entity_type in DELETE_ORDER:
        group = next((g for g in preview["groups"] if g["entity_type"] == entity_type), None)
        if group is None:
            continue
        spec = _spec(entity_type)
        for item in group["items"]:
            if spec.deleter is None:
                continue
            if spec.deleter(db, item["id"]):
                deleted[entity_type] = deleted.get(entity_type, 0) + 1
    return {"deleted": deleted, "freed_bytes": freed}
