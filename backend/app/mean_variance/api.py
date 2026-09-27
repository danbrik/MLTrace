
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.mean_variance import service
from app.mean_variance.schemas import MeanVarianceConfig, MeanVarianceRunRead, SelectionPreview

router = APIRouter(prefix="/api/mean-variance-analysis", tags=["mean-variance"])


def required(value):
    if value is None:
        raise HTTPException(404, "Analyse oder Artefakt nicht gefunden.")
    return value


@router.post("/preview", response_model=SelectionPreview)
def preview(payload: MeanVarianceConfig, db: Session = Depends(get_db)):
    try:
        return service.preview(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/runs", response_model=MeanVarianceRunRead)
def enqueue(payload: MeanVarianceConfig, db: Session = Depends(get_db)):
    try:
        return service.enqueue(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs", response_model=list[MeanVarianceRunRead])
def list_runs(db: Session = Depends(get_db)):
    return service.list_runs(db)


@router.get("/runs/{run_id}", response_model=MeanVarianceRunRead)
def get_run(run_id: int, db: Session = Depends(get_db)):
    return required(service.get_run(db, run_id))


@router.get("/runs/{run_id}/results")
def results(run_id: int, db: Session = Depends(get_db)):
    return required(service.results(db, run_id))


@router.get("/runs/{run_id}/log")
def log(run_id: int, db: Session = Depends(get_db)):
    return {"log": required(service.read_log(db, run_id))}


@router.post("/runs/{run_id}/abort", response_model=MeanVarianceRunRead)
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
    if name not in {"mean_difference.png", "variance_difference.png"}:
        raise HTTPException(404, "Heatmap nicht gefunden.")
    path = required(service.artifact_path(db, run_id, name))
    return FileResponse(path, filename=f"mean-variance-{run_id}-{name}",
                        content_disposition_type="attachment" if download else "inline")
