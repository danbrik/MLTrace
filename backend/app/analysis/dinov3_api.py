import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.analysis import dinov3_service as service
from app.analysis.dinov3_schemas import RepresentationConfig, RepresentationResults, RepresentationRunRead, SelectionPreview

router = APIRouter(prefix="/api/dinov3-analysis", tags=["representation"])


def required(value):
    if value is None:
        raise HTTPException(404, "Analyse oder Artefakt nicht gefunden.")
    return value


@router.post("/preview", response_model=SelectionPreview)
def preview(payload: RepresentationConfig, db: Session = Depends(get_db)):
    try:
        return service.preview(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/runs", response_model=RepresentationRunRead)
def enqueue(payload: RepresentationConfig, db: Session = Depends(get_db)):
    try:
        return service.enqueue(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs", response_model=list[RepresentationRunRead])
def list_runs(db: Session = Depends(get_db)):
    return service.list_runs(db)


@router.get("/runs/{run_id}", response_model=RepresentationRunRead)
def get_run(run_id: int, db: Session = Depends(get_db)):
    return required(service.get_run(db, run_id))


@router.get("/runs/{run_id}/results", response_model=RepresentationResults)
def results(run_id: int, db: Session = Depends(get_db)):
    run = required(service.get_run(db, run_id))
    path = required(service.artifact_path(db, run_id, "points.json"))
    return {"metrics": run.result, "points": json.loads(path.read_text(encoding="utf-8"))}


@router.get("/runs/{run_id}/log")
def log(run_id: int, db: Session = Depends(get_db)):
    return {"log": required(service.read_log(db, run_id))}


@router.post("/runs/{run_id}/abort", response_model=RepresentationRunRead)
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
def artifact(run_id: int, name: str, db: Session = Depends(get_db)):
    path = required(service.artifact_path(db, run_id, name))
    return FileResponse(path, filename=f"representation-{run_id}-{name}")
