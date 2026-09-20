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
    except training.Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def read_upload(file: UploadFile):
    content = file.file.read(service.MAX_CSV_BYTES + 1)
    if len(content) > service.MAX_CSV_BYTES:
        raise HTTPException(413, "Die CSV darf höchstens 50 MB groß sein.")
    return content


@router.post("/preview")
def preview(file: UploadFile = File(...), timestamp_column: str | None = Form(None), timestamp_format: str = Form("ISO8601"), label_column: str | None = Form(None)):
    content = read_upload(file)
    return checked(lambda: service.preview(content, timestamp_column, timestamp_format, label_column))


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
    checked(lambda: training.guard_delete(db, 'dataset', record_id))
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
    checked(lambda: training.guard_delete(db, 'split', record_id))
    db.delete(get_record(db, TimeSeriesSplit, record_id))
    db.commit()

# Training resources use the same project-scoped database dependency as datasets.
from fastapi import Query
from fastapi.responses import Response, StreamingResponse
from app import models
from app.time_series import training_service as training, results
from app.time_series.definitions import DEFINITIONS, definition
from app.time_series.schemas import ModelInput, PipelineInput, PipelinePreview


@router.get('/model-definitions')
def model_definitions():
    return [definition(kind) for kind in DEFINITIONS]


@router.get('/models')
def list_models(db: Session = Depends(get_db)):
    return training.list_models(db)


@router.post('/models', status_code=201)
def create_model(payload: ModelInput, db: Session = Depends(get_db)):
    return checked(lambda: training.save_model(db, payload))


@router.put('/models/{record_id}')
def update_model(record_id: int, payload: ModelInput, db: Session = Depends(get_db)):
    return checked(lambda: training.save_model(db, payload, record_id))


@router.delete('/models/{record_id}', status_code=204)
def delete_model(record_id: int, db: Session = Depends(get_db)):
    checked(lambda: training.guard_delete(db, 'model', record_id))
    row = get_record(db, models.TimeSeriesModel, record_id)
    # Keep a template marker so deleting a preset does not recreate it on refresh.
    if row.template_key:
        row.name = ''
        db.commit()
    else:
        db.delete(row)
        db.commit()


@router.post('/pipelines/preview')
def pipeline_preview(payload: PipelinePreview, db: Session = Depends(get_db)):
    return checked(lambda: training.resolve_preview(db, payload)[3])


@router.get('/pipelines')
def list_pipelines(db: Session = Depends(get_db)):
    return training.list_pipelines(db)


@router.post('/pipelines', status_code=201)
def create_pipeline(payload: PipelineInput, db: Session = Depends(get_db)):
    return checked(lambda: training.save_pipeline(db, payload))


@router.put('/pipelines/{record_id}')
def update_pipeline(record_id: int, payload: PipelineInput, db: Session = Depends(get_db)):
    return checked(lambda: training.save_pipeline(db, payload, record_id))


@router.delete('/pipelines/{record_id}', status_code=204)
def delete_pipeline(record_id: int, db: Session = Depends(get_db)):
    checked(lambda: training.guard_delete(db, 'pipeline', record_id))
    db.delete(get_record(db, models.TimeSeriesPipeline, record_id))
    db.commit()


@router.post('/pipelines/{record_id}/runs', status_code=201)
def start_run(record_id: int, db: Session = Depends(get_db)):
    return checked(lambda: training.enqueue(db, record_id))


@router.get('/runs')
def list_runs(db: Session = Depends(get_db)):
    return training.list_runs(db)


@router.get('/runs/{record_id}')
def get_run(record_id: int, db: Session = Depends(get_db)):
    row = get_record(db, models.TimeSeriesRun, record_id)
    response = training.run_read(row)
    response['metrics'] = [training.record_dict(metric, 'epoch train_loss val_loss details') for metric in
                           db.scalars(select(models.TimeSeriesEpochMetric).where(models.TimeSeriesEpochMetric.run_id == record_id).order_by(models.TimeSeriesEpochMetric.epoch))]
    return response


@router.post('/runs/{record_id}/abort')
def abort_run(record_id: int, db: Session = Depends(get_db)):
    return checked(lambda: training.abort_run(db, record_id))


@router.delete('/runs/{record_id}', status_code=204)
def delete_run(record_id: int, db: Session = Depends(get_db)):
    checked(lambda: training.delete_run(db, record_id))


@router.get('/runs/{record_id}/logs')
def run_logs(record_id: int, db: Session = Depends(get_db)):
    row = get_record(db, models.TimeSeriesRun, record_id)
    from pathlib import Path
    path = Path(row.log_path) if row.log_path else training.artifact_dir(row.id) / 'worker.log'
    if not path.exists():
        return {'text': ''}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 100000))
        return {'text': stream.read().decode('utf-8', errors='replace')}


@router.get('/runs/{record_id}/series')
def result_series(record_id: int, subset: str = 'test', sensor: int = 0, scaled: bool = False,
                  offset: int = Query(0, ge=0), limit: int = Query(2000, ge=1, le=10000),
                  start: str | None = None, end: str | None = None, db: Session = Depends(get_db)):
    row = get_record(db, models.TimeSeriesRun, record_id)
    return checked(lambda: results.series(row, subset, sensor, scaled, offset, limit, start, end))


@router.get('/runs/{record_id}/representations')
def representations(record_id: int, subset: str = 'test', offset: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000), db: Session = Depends(get_db)):
    from itertools import islice
    row = get_record(db, models.TimeSeriesRun, record_id)
    return checked(lambda: list(islice(results.result_rows(row, subset, include_latent=True), offset, offset + limit)))


@router.get('/runs/{record_id}/windows/{endpoint}')
def full_window(record_id: int, endpoint: int, db: Session = Depends(get_db)):
    return checked(lambda: results.replay(get_record(db, models.TimeSeriesRun, record_id), endpoint))


@router.get('/runs/{record_id}/export.csv')
def export_scores(record_id: int, sensor: int = 0, scaled: bool = False, db: Session = Depends(get_db)):
    row = get_record(db, models.TimeSeriesRun, record_id)
    checked(lambda: results.load_manifest(row))
    if not 0 <= sensor < len(row.snapshot['preview']['columns']):
        raise HTTPException(422, 'Unbekannter Sensor.')
    return StreamingResponse(results.export_csv(row, sensor=sensor, scaled=scaled), media_type='text/csv',
                             headers={'Content-Disposition': f'attachment; filename="run-{record_id}-sensor-{sensor}.csv"'})


@router.get('/runs/{record_id}/representations.zip')
def export_representations(record_id: int, db: Session = Depends(get_db)):
    content = checked(lambda: results.latent_export(get_record(db, models.TimeSeriesRun, record_id)))
    return Response(content, media_type='application/zip', headers={'Content-Disposition': f'attachment; filename="run-{record_id}-latents.zip"'})


@router.get('/models/{record_id}')
def get_model(record_id: int, db: Session = Depends(get_db)):
    row = get_record(db, models.TimeSeriesModel, record_id)
    if not row.name:
        raise HTTPException(404, 'Modell wurde gelöscht.')
    return training.model_read(row)


@router.get('/pipelines/{record_id}')
def get_pipeline(record_id: int, db: Session = Depends(get_db)):
    return training.pipeline_read(get_record(db, models.TimeSeriesPipeline, record_id))
