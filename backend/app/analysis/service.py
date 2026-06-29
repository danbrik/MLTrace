from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import models
from app.schemas import AnalysisLayoutCreate


def _find_by_name_case_insensitive(
    db: Session,
    name: str,
    *,
    exclude_id: int | None = None,
) -> models.AnalysisLayout | None:
    query = select(models.AnalysisLayout).where(func.lower(models.AnalysisLayout.name) == name.lower())
    if exclude_id is not None:
        query = query.where(models.AnalysisLayout.id != exclude_id)
    return db.scalar(query)


def list_analysis_layouts(db: Session) -> list[models.AnalysisLayout]:
    return list(db.scalars(select(models.AnalysisLayout).order_by(models.AnalysisLayout.updated_at.desc())))


def get_analysis_layout(db: Session, layout_id: int) -> models.AnalysisLayout | None:
    return db.get(models.AnalysisLayout, layout_id)


def create_analysis_layout(db: Session, payload: AnalysisLayoutCreate) -> models.AnalysisLayout:
    name = payload.name.strip()
    if not name:
        raise ValueError("Analysis layout name is required.")
    if _find_by_name_case_insensitive(db, name):
        raise ValueError(f"Analysis layout name already exists: {name}")
    layout = models.AnalysisLayout(
        name=name,
        description=payload.description,
        layout=payload.layout,
    )
    db.add(layout)
    db.commit()
    db.refresh(layout)
    return layout


def update_analysis_layout(db: Session, layout_id: int, payload: AnalysisLayoutCreate) -> models.AnalysisLayout | None:
    layout = db.get(models.AnalysisLayout, layout_id)
    if layout is None:
        return None
    name = payload.name.strip()
    if not name:
        raise ValueError("Analysis layout name is required.")
    if _find_by_name_case_insensitive(db, name, exclude_id=layout_id):
        raise ValueError(f"Analysis layout name already exists: {name}")
    layout.name = name
    layout.description = payload.description
    layout.layout = payload.layout
    db.commit()
    db.refresh(layout)
    return layout


def delete_analysis_layout(db: Session, layout_id: int) -> bool:
    layout = db.get(models.AnalysisLayout, layout_id)
    if layout is None:
        return False
    db.delete(layout)
    db.commit()
    return True
