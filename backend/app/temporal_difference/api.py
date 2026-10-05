from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.temporal_difference import service
from app.temporal_difference.schemas import TemporalDifferenceConfig, TemporalDifferenceRunRead, PlotSettings

router = APIRouter(prefix="/api/temporal-difference", tags=["temporal-difference"])


def required(value):
    if value is None:
        raise HTTPException(404, "Analyse oder vollständiges Ergebnis nicht gefunden.")
    return value


@router.post("/preview")
def preview(payload: TemporalDifferenceConfig, db: Session = Depends(get_db)):
    try:
        return service.preview(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/runs", response_model=TemporalDifferenceRunRead)
def enqueue(payload: TemporalDifferenceConfig, db: Session = Depends(get_db)):
    try:
        return service.enqueue(db, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs", response_model=list[TemporalDifferenceRunRead])
def list_runs(db: Session = Depends(get_db)):
    return service.list_runs(db)


@router.get("/runs/{run_id}", response_model=TemporalDifferenceRunRead)
def get_run(run_id: int, db: Session = Depends(get_db)):
    return required(service.get_run(db, run_id))


@router.get("/runs/{run_id}/summary")
def summary(run_id: int, db: Session = Depends(get_db)):
    return required(service.summaries(db, run_id))


@router.get("/runs/{run_id}/pairs")
def pairs(run_id: int, offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=500),
          role: Literal["reference", "comparison"] | None = None, delta: int | None = Query(default=None, gt=0),
          db: Session = Depends(get_db)):
    return required(service.values(db, run_id, offset, limit, role, delta))


@router.put("/runs/{run_id}/plot-settings", response_model=PlotSettings)
def save_plot(run_id: int, payload: PlotSettings, db: Session = Depends(get_db)):
    return required(service.save_plot(db, run_id, payload))


@router.get("/runs/{run_id}/csv/{kind}")
def download(run_id: int, kind: Literal["summary", "pairs"], db: Session = Depends(get_db)):
    return FileResponse(required(service.csv_path(db, run_id, kind)), media_type="text/csv",
                        filename=f"temporal-difference-{run_id}-{kind}.csv")


@router.get("/runs/{run_id}/log")
def log(run_id: int, db: Session = Depends(get_db)):
    return {"log": required(service.read_log(db, run_id))}


@router.post("/runs/{run_id}/abort", response_model=TemporalDifferenceRunRead)
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


from app.temporal_difference import matrix
from app.temporal_difference.schemas import MatrixConfig


@router.get('/runs/{run_id}/matrix')
def matrix_state(run_id: int, role: Literal['reference', 'comparison'], db: Session = Depends(get_db)):
    return required(matrix.state(db, run_id, role))


@router.get('/runs/{run_id}/matrix/start-points')
def matrix_candidates(run_id: int, role: Literal['reference', 'comparison'], deltas: list[int] = Query(), db: Session = Depends(get_db)):
    try:
        return required(matrix.candidates(db, run_id, role, deltas))
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post('/runs/{run_id}/matrix')
def create_matrix(run_id: int, payload: MatrixConfig, db: Session = Depends(get_db)):
    try:
        return required(matrix.generate(db, run_id, payload))
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get('/runs/{run_id}/matrix/png')
def matrix_png(run_id: int, role: Literal['reference', 'comparison'], artifact: str, download: bool = False, db: Session = Depends(get_db)):
    path = required(matrix.png_path(db, run_id, role, artifact))
    return FileResponse(path, media_type='image/png', filename=f'temporal-difference-{run_id}-{role}-matrix.png' if download else None)
