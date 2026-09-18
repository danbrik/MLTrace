from datetime import datetime
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session
from app.database import SessionLocal, get_db, project_context
from app.data_quality import characterization as service


def _background(database_url, artifact_dir, run_id):
    with project_context(database_url, artifact_dir), SessionLocal() as db:
        service.calculate(db, run_id)


def register_routes(app: FastAPI):
    prefix = '/api/data-quality/analyses/{analysis_id}/characterization'

    def schedule(request, tasks, db, row):
        project = getattr(request.state, 'project', None)
        if project:
            tasks.add_task(_background, project.database_url, project.artifact_dir, row.id)
        else:
            tasks.add_task(service.calculate, db, row.id)

    def required(db, analysis_id):
        row = service.get(db, analysis_id)
        if row is None:
            raise HTTPException(404, 'Characterization not found.')
        return row

    @app.get(prefix)
    def read(analysis_id: int, db: Session = Depends(get_db)):
        return service.read(service.get(db, analysis_id))

    @app.post(prefix)
    def start(analysis_id: int, request: Request, tasks: BackgroundTasks, db: Session = Depends(get_db)):
        try:
            row, created = service.start(db, analysis_id)
            if created:
                schedule(request, tasks, db, row)
            return service.read(row)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post(prefix + '/retry')
    def retry(analysis_id: int, request: Request, tasks: BackgroundTasks, db: Session = Depends(get_db)):
        row = required(db, analysis_id)
        if not service.retry(db, row):
            raise HTTPException(409, 'Only failed or cancelled characterization can be retried.')
        schedule(request, tasks, db, row)
        return service.read(row)

    @app.post(prefix + '/cancel')
    def cancel(analysis_id: int, db: Session = Depends(get_db)):
        row = required(db, analysis_id)
        if row.job_status not in service.ACTIVE:
            raise HTTPException(409, 'Only active characterization can be cancelled.')
        row.cancel_requested = True
        db.commit()
        return service.read(row)

    @app.get(prefix + '/export')
    def export(analysis_id: int, search: str = '', data_type: str = 'all', sort: str = 'variable', descending: bool = False, db: Session = Depends(get_db)):
        try:
            return Response(service.export(required(db, analysis_id), search, data_type, sort, descending), media_type='text/csv; charset=utf-8',
                            headers={'Content-Disposition': f'attachment; filename="characterization-{analysis_id}.csv"'})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get(prefix + '/detail')
    def detail(analysis_id: int, sensor: str, db: Session = Depends(get_db)):
        try:
            return service.detail(required(db, analysis_id), sensor)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get(prefix + '/series')
    def series(analysis_id: int, sensor: str, start: datetime | None = None, end: datetime | None = None, db: Session = Depends(get_db)):
        if (start and start.tzinfo) or (end and end.tzinfo) or (start and end and end < start):
            raise HTTPException(400, 'Use an ordered range of CSV-local timestamps.')
        try:
            return service.series(required(db, analysis_id), sensor, start, end)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
