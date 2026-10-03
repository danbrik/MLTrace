from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app import models
from app.database import get_db
from app.mean_variance import roi_service as service
from app.mean_variance.roi_engine import RoiConfig

router = APIRouter(prefix="/api/mean-variance-analysis", tags=["variance-roi"])


def invoke(fn, *args):
    try:
        return fn(*args)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/roi-jobs")
def jobs(db: Session = Depends(get_db)):
    return service.list_jobs(db)


@router.get("/runs/{run_id}/roi")
def state(run_id: int, db: Session = Depends(get_db)):
    return invoke(service.state, db, run_id)


@router.post("/runs/{run_id}/roi/prepare")
def prepare(run_id: int, db: Session = Depends(get_db)):
    return invoke(service.enqueue, db, run_id, "prepare")


@router.post("/runs/{run_id}/roi/evaluate")
def evaluate(run_id: int, payload: RoiConfig, db: Session = Depends(get_db)):
    return invoke(service.enqueue, db, run_id, "evaluate", payload)


@router.get("/runs/{run_id}/roi/images/{pair}/{layer}")
def image(run_id: int, pair: int, layer: str, db: Session = Depends(get_db)):
    state = invoke(service.state, db, run_id)
    if not state["ready"] or pair < 0 or pair >= len(state["basis"]["pairs"]) or layer not in {"background", "heatmap"}:
        raise HTTPException(404, "ROI-Bild nicht gefunden.")
    return FileResponse(service.basis_dir(run_id, db) / f"{pair}_{layer}.png", media_type="image/png")


@router.post("/roi-jobs/{job_id}/abort")
def abort(job_id: int, db: Session = Depends(get_db)):
    return invoke(service.abort_job, db, job_id)


@router.get("/roi-jobs/{job_id}/log")
def log(job_id: int, db: Session = Depends(get_db)):
    return {"log": invoke(service.job_log, db, job_id)}


@router.delete("/roi-jobs/{job_id}", status_code=204)
def delete(job_id: int, db: Session = Depends(get_db)):
    if not invoke(service.delete_job, db, job_id):
        raise HTTPException(404, "ROI-Auftrag nicht gefunden.")


@router.get("/runs/{run_id}/roi/artifacts/{job_id}/{name}")
def artifact(run_id: int, job_id: int, name: str, download: bool = False, db: Session = Depends(get_db)):
    invoke(service.parent_required, db, run_id)
    job = db.get(models.VarianceRoiJob, job_id)
    if not job or job.parent_run_id != run_id or job.status != "finished" or job.operation != "evaluate" or name not in {"roi_comparison.png", "roi_table.png"}:
        raise HTTPException(404, "ROI-Ergebnis nicht gefunden.")
    path = service.artifact_dir(job_id) / name
    if not path.is_file():
        raise HTTPException(404, "ROI-Ergebnis nicht gefunden.")
    return FileResponse(path, media_type="image/png", filename=f"variance-{run_id}-roi-{job_id}-{name}",
                        content_disposition_type="attachment" if download else "inline")
