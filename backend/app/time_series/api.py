from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.database import get_db
from app.models import TimeSeriesDataset, TimeSeriesSplit
from app.time_series import service
from app.time_series.schemas import DatasetImport, DatasetUpdate, SplitInput

router = APIRouter(prefix="/api/time-series", tags=["Time series"])


def get_record(db, model, record_id):
    record = db.get(model, record_id)
    if record is None:
        raise HTTPException(404, "Eintrag nicht gefunden.")
    return record


def checked(operation):
    try:
        return operation()
    except ValidationError as exc:
        raise HTTPException(422, "; ".join(error["msg"] for error in exc.errors())) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def read_upload(file: UploadFile):
    content = file.file.read(service.MAX_CSV_BYTES + 1)
    if len(content) > service.MAX_CSV_BYTES:
        raise HTTPException(413, "Die CSV darf höchstens 50 MB groß sein.")
    return content


@router.post("/preview")
def preview(file: UploadFile = File(...), timestamp_column: str | None = Form(None), timestamp_format: str = Form("ISO8601")):
    content = read_upload(file)
    return checked(lambda: service.preview(content, timestamp_column, timestamp_format))


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.scalars(select(TimeSeriesDataset).options(selectinload(TimeSeriesDataset.splits)).order_by(TimeSeriesDataset.id.desc()))
    return [service.dataset_read(item) for item in datasets]


@router.post("/datasets", status_code=201)
def create_dataset(file: UploadFile = File(...), metadata: str = Form(...), db: Session = Depends(get_db)):
    payload = checked(lambda: DatasetImport.model_validate_json(metadata))
    content = read_upload(file)
    return checked(lambda: service.dataset_read(service.create_dataset(db, content, file.filename or "data.csv", payload), detail=True))


@router.get("/datasets/{record_id}")
def get_dataset(record_id: int, db: Session = Depends(get_db)):
    return service.dataset_read(get_record(db, TimeSeriesDataset, record_id), detail=True)


@router.put("/datasets/{record_id}")
def update_dataset(record_id: int, payload: DatasetUpdate, db: Session = Depends(get_db)):
    dataset = get_record(db, TimeSeriesDataset, record_id)
    return checked(lambda: service.dataset_read(service.update_dataset(db, dataset, payload), detail=True))


@router.delete("/datasets/{record_id}", status_code=204)
def delete_dataset(record_id: int, db: Session = Depends(get_db)):
    db.delete(get_record(db, TimeSeriesDataset, record_id))
    db.commit()


@router.get("/splits")
def list_splits(db: Session = Depends(get_db)):
    splits = db.scalars(select(TimeSeriesSplit).options(selectinload(TimeSeriesSplit.dataset)).order_by(TimeSeriesSplit.id.desc()))
    return [service.split_read(item) for item in splits]


@router.post("/splits", status_code=201)
def create_split(payload: SplitInput, db: Session = Depends(get_db)):
    dataset = get_record(db, TimeSeriesDataset, payload.dataset_id)
    return checked(lambda: service.split_read(service.save_split(db, dataset, payload)))


@router.get("/splits/{record_id}")
def get_split(record_id: int, db: Session = Depends(get_db)):
    return service.split_read(get_record(db, TimeSeriesSplit, record_id))


@router.put("/splits/{record_id}")
def update_split(record_id: int, payload: SplitInput, db: Session = Depends(get_db)):
    split = get_record(db, TimeSeriesSplit, record_id)
    dataset = get_record(db, TimeSeriesDataset, payload.dataset_id)
    return checked(lambda: service.split_read(service.save_split(db, dataset, payload, split)))


@router.delete("/splits/{record_id}", status_code=204)
def delete_split(record_id: int, db: Session = Depends(get_db)):
    db.delete(get_record(db, TimeSeriesSplit, record_id))
    db.commit()
