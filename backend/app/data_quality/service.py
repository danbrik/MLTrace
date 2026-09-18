from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models
from app.data_quality.engine import VERSION, analyze
from app.data_quality.schemas import Parameters
from app.redundancy.engine import AnalysisCancelled


def identity(db: Session, payload: Parameters):
    source = db.get(models.RedundancyCsvSource, payload.source_id)
    if source is None:
        raise ValueError('CSV source not found.')
    if payload.time_column not in source.headers or any(c not in source.headers for c in payload.selected_columns):
        raise ValueError('Selected columns do not exist in the CSV.')
    params = payload.model_dump(mode='json')
    key_params = {k: v for k, v in params.items() if k != 'source_id'}
    key = hashlib.sha256(json.dumps(dict(source_sha256=source.sha256, version=VERSION, **key_params), sort_keys=True).encode()).hexdigest()
    return source, params, key


def lookup(db, payload):
    _, _, key = identity(db, payload)
    return db.scalar(select(models.DataQualityAnalysis).where(models.DataQualityAnalysis.cache_key == key))


def create(db, payload):
    source, params, key = identity(db, payload)
    existing = db.scalar(select(models.DataQualityAnalysis).where(models.DataQualityAnalysis.cache_key == key))
    if existing:
        return existing, False
    row = models.DataQualityAnalysis(source_id=source.id, name=f'{source.name} · Data quality'[:255], cache_key=key, parameters=params)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        row = db.scalar(select(models.DataQualityAnalysis).where(models.DataQualityAnalysis.cache_key == key))
        if row is None:
            raise
        return row, False
    db.refresh(row)
    return row, True


def calculate(db: Session, analysis_id: int):
    claimed = db.execute(update(models.DataQualityAnalysis).where(
        models.DataQualityAnalysis.id == analysis_id, models.DataQualityAnalysis.job_status == 'queued'
    ).values(job_status='running', started_at=datetime.now(timezone.utc).replace(tzinfo=None)))
    db.commit()
    if not claimed.rowcount:
        return
    row = db.get(models.DataQualityAnalysis, analysis_id)
    began = time.monotonic()
    last_report = 0.0

    def cancelled():
        db.refresh(row, attribute_names=['cancel_requested'])
        return row.cancel_requested

    def progress(value, stage):
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < .3 and stage == row.stage:
            return
        row.progress, row.stage = value, stage
        row.elapsed_seconds = now - began
        row.eta_seconds = row.elapsed_seconds * (1 - value) / value if value >= .05 and row.elapsed_seconds >= 2 else None
        db.commit()
        last_report = now

    try:
        source = db.get(models.RedundancyCsvSource, row.source_id)
        result, runs = analyze(Path(source.artifact_path), source.delimiter, row.parameters, source.row_count, progress, cancelled)
        if cancelled():
            raise AnalysisCancelled('Calculation cancelled.')
        row.result, row.missing_runs = result, runs
        row.job_status, row.stage, row.progress = 'ready', 'Complete', 1.0
        row.eta_seconds = 0
    except AnalysisCancelled:
        row.job_status, row.stage = 'cancelled', 'Cancelled'
        row.eta_seconds = None
    except Exception as exc:
        row.job_status, row.stage = 'failed', 'Failed'
        row.error_message = str(exc)
        row.eta_seconds = None
    row.elapsed_seconds = time.monotonic() - began
    db.commit()


def retry(db, row):
    result = db.execute(update(models.DataQualityAnalysis).where(
        models.DataQualityAnalysis.id == row.id,
        models.DataQualityAnalysis.job_status.in_(['failed', 'cancelled'])
    ).values(job_status='queued', stage='Queued', progress=0, elapsed_seconds=0, eta_seconds=None,
             started_at=None, cancel_requested=False, error_message=None, result=None, missing_runs=None))
    db.commit()
    db.refresh(row)
    return bool(result.rowcount)


def delete_analysis(db, analysis_id):
    db.execute(update(models.DataQualityAnalysis).where(models.DataQualityAnalysis.id == analysis_id)
               .values(updated_at=models.DataQualityAnalysis.updated_at))
    row = db.get(models.DataQualityAnalysis, analysis_id)
    if row is None:
        return False
    if row.job_status in {'queued', 'running'}:
        raise ValueError('Cancel the running analysis before deleting it.')
    from app.data_quality.characterization import delete_for_analysis
    delete_for_analysis(db, analysis_id)
    db.delete(row)
    db.commit()
    return True


def reconcile_interrupted(db):
    db.execute(update(models.DataQualityAnalysis).where(models.DataQualityAnalysis.job_status.in_(['queued', 'running'])).values(
        job_status='failed', stage='Interrupted', eta_seconds=None, error_message='The server restarted during calculation. Retry this analysis.'))
    db.commit()
