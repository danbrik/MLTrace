from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.reference_image import service
from app.reference_image.schemas import ReferenceImageConfig, ReferenceImageRunRead, ReferenceImageSelectionPreview, FrameLookup
from app.reference_image.engine import resolve_frame
from app.schemas import _dataset_local_naive

router = APIRouter(prefix="/api/reference-image-analysis", tags=["reference-image"])


def required(value):
    if value is None:
        raise HTTPException(404, "Analyse oder Artefakt nicht gefunden.")
    return value


@router.post("/preview", response_model=ReferenceImageSelectionPreview)
def preview(payload: ReferenceImageConfig, db: Session = Depends(get_db)):
    try:
        return service.preview(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/runs", response_model=ReferenceImageRunRead)
def enqueue(payload: ReferenceImageConfig, db: Session = Depends(get_db)):
    try:
        return service.enqueue(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs", response_model=list[ReferenceImageRunRead])
def list_runs(db: Session = Depends(get_db)):
    return service.list_runs(db)


@router.get("/runs/{run_id}", response_model=ReferenceImageRunRead)
def get_run(run_id: int, db: Session = Depends(get_db)):
    return required(service.get_run(db, run_id))


@router.get("/runs/{run_id}/results")
def results(run_id: int, db: Session = Depends(get_db)):
    return required(service.results(db, run_id))


@router.get("/runs/{run_id}/frame", response_model=FrameLookup)
def frame(run_id: int, timestamp: datetime, db: Session = Depends(get_db)):
    try:
        timestamp = _dataset_local_naive(timestamp)
        return resolve_frame(required(service.results(db, run_id))["frames"], timestamp)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs/{run_id}/log")
def log(run_id: int, db: Session = Depends(get_db)):
    return {"log": required(service.read_log(db, run_id))}


@router.post("/runs/{run_id}/abort", response_model=ReferenceImageRunRead)
def abort(run_id: int, db: Session = Depends(get_db)):
    try:
        return required(service.abort_run(db, run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/runs/{run_id}", status_code=204)
def delete(run_id: int, db: Session = Depends(get_db)):
    try:
        if not service.delete_run(db, run_id):
            raise HTTPException(404, "Analyse nicht gefunden.")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/runs/{run_id}/artifacts/{name}")
def artifact(run_id: int, name: str, download: bool = False, db: Session = Depends(get_db)):
    path = required(service.artifact_path(db, run_id, name))
    return FileResponse(path, filename=f"reference-image-{run_id}-{name}",
                        content_disposition_type="attachment" if download else "inline")
