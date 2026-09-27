from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import TimeRangePreset
from app.time_range_presets.schemas import TimeRangePresetRead, TimeRangePresetWrite

router = APIRouter(prefix="/api/time-range-presets", tags=["time-range-presets"])


def required(db, preset_id):
    row = db.get(TimeRangePreset, preset_id)
    if row is None:
        raise HTTPException(404, "Zeitraumvorlage nicht gefunden.")
    return row


def save(db, row, payload):
    row.name = payload.name
    row.name_key = payload.name.casefold()
    row.start = payload.start
    row.end = payload.end
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "Eine Zeitraumvorlage mit diesem Namen existiert bereits.") from exc
    db.refresh(row)
    return row


@router.get("", response_model=list[TimeRangePresetRead])
def list_presets(db: Session = Depends(get_db)):
    return db.scalars(select(TimeRangePreset).order_by(TimeRangePreset.name_key, TimeRangePreset.id)).all()


@router.post("", response_model=TimeRangePresetRead, status_code=201)
def create_preset(payload: TimeRangePresetWrite, db: Session = Depends(get_db)):
    return save(db, TimeRangePreset(), payload)


@router.put("/{preset_id}", response_model=TimeRangePresetRead)
def update_preset(preset_id: int, payload: TimeRangePresetWrite, db: Session = Depends(get_db)):
    return save(db, required(db, preset_id), payload)


@router.delete("/{preset_id}", status_code=204)
def delete_preset(preset_id: int, db: Session = Depends(get_db)):
    db.delete(required(db, preset_id))
    db.commit()
