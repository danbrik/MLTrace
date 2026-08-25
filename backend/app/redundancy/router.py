from __future__ import annotations

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db, project_context
from app import models
from app.redundancy import service
from app.schemas import (
    RedundancyAnalysisCreate,
    RedundancyAnalysisRead,
    RedundancyAnalysisUpdate,
    RedundancyClusterCutRequest,
    RedundancyFinalizeRequest,
    RedundancySeriesRead,
    RedundancySourceRead,
)


def _http_error(exc: ValueError) -> HTTPException:
    text = str(exc)
    status = 404 if "not found" in text.lower() else 409 if any(word in text.lower() for word in ("immutable", "running", "used by")) else 400
    return HTTPException(status_code=status, detail=text)


def _background_calculate(database_url: str, artifact_dir: str, analysis_id: int) -> None:
    with project_context(database_url, artifact_dir):
        db = SessionLocal()
        try:
            service.calculate_analysis(db, analysis_id)
        finally:
            db.close()


def register_routes(app: FastAPI) -> None:
    @app.get("/api/redundancy/sources", response_model=list[RedundancySourceRead])
    def list_sources(db: Session = Depends(get_db)):
        return service.list_sources(db)

    @app.post("/api/redundancy/sources", response_model=RedundancySourceRead)
    def upload_source(
        file: UploadFile = File(...),
        name: str | None = Query(default=None, max_length=255),
        db: Session = Depends(get_db),
    ):
        try:
            return service.create_source(db, file.file, file.filename or "source.csv", name)
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/redundancy/sources/{source_id}", response_model=RedundancySourceRead)
    def get_source(source_id: int, db: Session = Depends(get_db)):
        row = db.get(models.RedundancyCsvSource, source_id)
        if row is None:
            raise HTTPException(status_code=404, detail="CSV source not found.")
        return row

    @app.delete("/api/redundancy/sources/{source_id}", status_code=204)
    def delete_source(source_id: int, db: Session = Depends(get_db)):
        try:
            if not service.delete_source(db, source_id):
                raise HTTPException(status_code=404, detail="CSV source not found.")
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/redundancy/analyses", response_model=list[RedundancyAnalysisRead])
    def list_analyses(db: Session = Depends(get_db)):
        return service.list_analyses(db)

    @app.post("/api/redundancy/analyses", response_model=RedundancyAnalysisRead)
    def create_analysis(payload: RedundancyAnalysisCreate, db: Session = Depends(get_db)):
        try:
            return service.create_analysis(db, payload)
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/redundancy/analyses/{analysis_id}", response_model=RedundancyAnalysisRead)
    def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
        row = db.get(models.RedundancyAnalysis, analysis_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Redundancy analysis not found.")
        return row

    @app.patch("/api/redundancy/analyses/{analysis_id}", response_model=RedundancyAnalysisRead)
    def update_analysis(analysis_id: int, payload: RedundancyAnalysisUpdate, db: Session = Depends(get_db)):
        try:
            row = service.update_analysis(db, analysis_id, payload)
            if row is None:
                raise HTTPException(status_code=404, detail="Redundancy analysis not found.")
            return row
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/redundancy/analyses/{analysis_id}/calculate", response_model=RedundancyAnalysisRead)
    def calculate(
        analysis_id: int,
        request: Request,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
    ):
        row = db.scalar(select(models.RedundancyAnalysis).where(models.RedundancyAnalysis.id == analysis_id).with_for_update())
        if row is None:
            raise HTTPException(status_code=404, detail="Redundancy analysis not found.")
        if row.status == "finalized":
            raise HTTPException(status_code=409, detail="Finalized redundancy snapshots are immutable.")
        if row.job_status in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="Redundancy analysis is already running.")
        row.job_status = "queued"
        row.progress = 0
        row.error_message = None
        row.cancel_requested = False
        db.commit()
        db.refresh(row)
        project = getattr(request.state, "project", None)
        if project is None:
            return service.calculate_analysis(db, analysis_id)
        background_tasks.add_task(_background_calculate, project.database_url, project.artifact_dir, analysis_id)
        return row

    @app.post("/api/redundancy/analyses/{analysis_id}/cancel", response_model=RedundancyAnalysisRead)
    def cancel(analysis_id: int, db: Session = Depends(get_db)):
        try:
            row = service.request_cancel(db, analysis_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Redundancy analysis not found.")
            return row
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/redundancy/analyses/{analysis_id}/retry", response_model=RedundancyAnalysisRead)
    def retry(
        analysis_id: int,
        request: Request,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
    ):
        row = db.scalar(select(models.RedundancyAnalysis).where(models.RedundancyAnalysis.id == analysis_id).with_for_update())
        if row is None:
            raise HTTPException(status_code=404, detail="Redundancy analysis not found.")
        if row.status == "finalized" or row.job_status not in {"failed", "cancelled", "stale", "not_calculated"}:
            raise HTTPException(status_code=409, detail="Only an unfinished inactive draft can be retried.")
        row.job_status = "queued"
        row.progress = 0
        row.error_message = None
        row.cancel_requested = False
        db.commit()
        db.refresh(row)
        project = getattr(request.state, "project", None)
        if project is None:
            return service.calculate_analysis(db, analysis_id)
        background_tasks.add_task(_background_calculate, project.database_url, project.artifact_dir, analysis_id)
        return row

    @app.post("/api/redundancy/analyses/{analysis_id}/cluster-cut")
    def cluster_cut(analysis_id: int, payload: RedundancyClusterCutRequest, db: Session = Depends(get_db)):
        try:
            return service.preview_cluster_cut(db, analysis_id, payload.cutoff)
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/redundancy/analyses/{analysis_id}/finalize", response_model=RedundancyAnalysisRead)
    def finalize(analysis_id: int, payload: RedundancyFinalizeRequest, db: Session = Depends(get_db)):
        try:
            return service.finalize_analysis(db, analysis_id, payload.cutoff)
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/redundancy/analyses/{analysis_id}/duplicate", response_model=RedundancyAnalysisRead)
    def duplicate(analysis_id: int, db: Session = Depends(get_db)):
        try:
            return service.duplicate_analysis(db, analysis_id)
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.delete("/api/redundancy/analyses/{analysis_id}", status_code=204)
    def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
        try:
            if not service.delete_analysis(db, analysis_id):
                raise HTTPException(status_code=404, detail="Redundancy analysis not found.")
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/redundancy/analyses/{analysis_id}/series", response_model=RedundancySeriesRead)
    def series(
        analysis_id: int,
        columns: list[str] = Query(...),
        max_points: int = Query(default=8000, ge=0, le=250000),
        offset: int = Query(default=0, ge=0),
        page_size: int | None = Query(default=None, ge=1, le=50000),
        db: Session = Depends(get_db),
    ):
        try:
            return service.series(db, analysis_id, columns, max_points, offset, page_size)
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/redundancy/analyses/{analysis_id}/exports/{kind}")
    def export(analysis_id: int, kind: str, db: Session = Depends(get_db)):
        try:
            if kind == "parameters":
                return Response(service.export_parameters(db, analysis_id), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="redundancy-{analysis_id}-parameters.json"'})
            content = service.export_csv(db, analysis_id, kind)
            return Response(content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="redundancy-{analysis_id}-{kind}.csv"'})
        except ValueError as exc:
            raise _http_error(exc) from exc
