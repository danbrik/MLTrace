from datetime import datetime

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.database import SessionLocal, get_db, project_context
from app.data_quality import service
from app.data_quality.engine import heatmap
from app.data_quality.schemas import AnalysisRead, Parameters


def _background(database_url, artifact_dir, analysis_id):
    with project_context(database_url, artifact_dir):
        with SessionLocal() as db:
            service.calculate(db, analysis_id)


def register_routes(app: FastAPI):
    def get_row(db, analysis_id):
        row = db.get(models.DataQualityAnalysis, analysis_id)
        if row is None:
            raise HTTPException(404, 'Data quality analysis not found.')
        return row

    def schedule(request, tasks, db, row):
        project = getattr(request.state, 'project', None)
        if project is not None:
            tasks.add_task(_background, project.database_url, project.artifact_dir, row.id)
        else:
            # Standalone/test database; preserve the injected session.
            tasks.add_task(service.calculate, db, row.id)

    @app.post('/api/data-quality/analyses/lookup', response_model=AnalysisRead | None)
    def lookup(payload: Parameters, db: Session = Depends(get_db)):
        try:
            return service.lookup(db, payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/data-quality/analyses', response_model=AnalysisRead)
    def create(payload: Parameters, request: Request, tasks: BackgroundTasks, db: Session = Depends(get_db)):
        try:
            row, created = service.create(db, payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if created:
            schedule(request, tasks, db, row)
        return row

    @app.get('/api/data-quality/analyses', response_model=list[AnalysisRead])
    def list_analyses(db: Session = Depends(get_db)):
        return list(db.scalars(select(models.DataQualityAnalysis).order_by(models.DataQualityAnalysis.created_at.desc())))

    @app.get('/api/data-quality/analyses/{analysis_id}', response_model=AnalysisRead)
    def read(analysis_id: int, db: Session = Depends(get_db)):
        return get_row(db, analysis_id)

    @app.post('/api/data-quality/analyses/{analysis_id}/cancel', response_model=AnalysisRead)
    def cancel(analysis_id: int, db: Session = Depends(get_db)):
        row = get_row(db, analysis_id)
        if row.job_status not in {'queued', 'running'}:
            raise HTTPException(409, 'Only queued or running analyses can be cancelled.')
        row.cancel_requested = True
        db.commit()
        db.refresh(row)
        return row

    @app.post('/api/data-quality/analyses/{analysis_id}/retry', response_model=AnalysisRead)
    def retry(analysis_id: int, request: Request, tasks: BackgroundTasks, db: Session = Depends(get_db)):
        row = get_row(db, analysis_id)
        if not service.retry(db, row):
            raise HTTPException(409, 'Only failed or cancelled analyses can be retried.')
        schedule(request, tasks, db, row)
        return row

    @app.delete('/api/data-quality/analyses/{analysis_id}', status_code=204)
    def delete(analysis_id: int, db: Session = Depends(get_db)):
        try:
            if not service.delete_analysis(db, analysis_id):
                raise HTTPException(404, 'Data quality analysis not found.')
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/data-quality/analyses/{analysis_id}/heatmap')
    def read_heatmap(analysis_id: int, start: datetime | None = None, end: datetime | None = None, db: Session = Depends(get_db)):
        row = get_row(db, analysis_id)
        if row.job_status != 'ready' or row.result is None or row.missing_runs is None:
            raise HTTPException(409, 'A completed result is required.')
        if (start and start.tzinfo) or (end and end.tzinfo) or (start and end and end < start):
            raise HTTPException(400, 'Use an ordered range of CSV-local timestamps.')
        return heatmap(row.result, row.missing_runs, start, end)
