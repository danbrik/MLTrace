from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app import models
from app.database import data_dir
from app.data_quality.engine import normalized_csv
from app.data_quality import characterization_engine as engine
from app.redundancy.engine import AnalysisCancelled

Run = models.CharacterizationRun
ACTIVE = ('queued', 'running')


def root(run_id):
    return data_dir() / 'characterization' / str(run_id)


def get(db, analysis_id):
    return db.scalar(select(Run).where(Run.analysis_id == analysis_id, Run.version == engine.VERSION))


def read(row):
    if row is None:
        return None
    return {key: getattr(row, key) for key in ('id', 'analysis_id', 'version', 'job_status', 'progress', 'stage',
            'elapsed_seconds', 'eta_seconds', 'error_message', 'result')}


def start(db, analysis_id):
    # A write lock also serializes start/delete on SQLite, where FOR UPDATE is ignored.
    db.execute(update(models.DataQualityAnalysis).where(models.DataQualityAnalysis.id == analysis_id)
               .values(updated_at=models.DataQualityAnalysis.updated_at))
    parent = db.scalar(select(models.DataQualityAnalysis).where(models.DataQualityAnalysis.id == analysis_id).with_for_update())
    if parent is None or parent.job_status != 'ready':
        raise ValueError('Load the data quality summary first.')
    row = get(db, analysis_id)
    if row:
        return row, False
    row = Run(analysis_id=analysis_id, version=engine.VERSION)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        row = get(db, analysis_id)
        if row is None:
            raise
        return row, False
    db.refresh(row)
    return row, True


def retry(db, row):
    claimed = db.execute(update(Run).where(Run.id == row.id, Run.job_status.in_(('failed', 'cancelled'))).values(
        job_status='queued', progress=0, stage='Queued', cancel_requested=False, error_message=None,
        elapsed_seconds=0, eta_seconds=None, result=None, artifact_path=None))
    db.commit()
    db.refresh(row)
    return bool(claimed.rowcount)


def calculate(db, run_id):
    claimed = db.execute(update(Run).where(Run.id == run_id, Run.job_status == 'queued').values(job_status='running'))
    db.commit()
    if not claimed.rowcount:
        return
    row = db.get(Run, run_id)
    began, last_report = time.monotonic(), 0.0
    folder = root(row.id)
    staging = folder / ('.pending-' + uuid4().hex)
    final = folder / ('result-' + uuid4().hex)

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
        shutil.rmtree(folder, ignore_errors=True)
        staging.mkdir(parents=True)
        parent = db.get(models.DataQualityAnalysis, row.analysis_id)
        source = db.get(models.RedundancyCsvSource, parent.source_id)
        summaries, files = {}, {}
        with normalized_csv(Path(source.artifact_path), source.delimiter, parent.parameters, source.row_count,
                            progress, cancelled) as (metadata, sensors):
            for position, (name, sensor, _, _) in enumerate(sensors):
                dtype = parent.parameters['data_types'][name]
                if dtype == 'numeric':
                    indices = np.array(sorted(sensor), dtype=np.int64)
                    values = np.array([sensor[i] for i in indices], dtype=float)
                    filename = f'{position}.npz'
                    np.savez_compressed(staging / filename, indices=indices, values=values)
                    files[name] = filename
                else:
                    values = list(sensor.values())
                summaries[name] = engine.summary(name, dtype, values)
        if cancelled():
            raise AnalysisCancelled()
        progress(.97, 'Saving characterization')
        # Retain source order for the sensor selector independently of table sorting.
        result = dict(metadata=metadata, summary=[summaries[name] for name in source.headers if name in summaries], files=files)
        staging.rename(final)
        row.artifact_path = str(final)
        row.result = result
        row.job_status, row.stage, row.progress, row.eta_seconds = 'ready', 'Complete', 1.0, 0.0
    except AnalysisCancelled:
        row.job_status, row.stage, row.eta_seconds = 'cancelled', 'Cancelled', None
        shutil.rmtree(folder, ignore_errors=True)
    except Exception as exc:
        row.job_status, row.stage, row.eta_seconds = 'failed', 'Failed', None
        row.error_message = str(exc)
        shutil.rmtree(folder, ignore_errors=True)
    row.elapsed_seconds = time.monotonic() - began
    db.commit()


def sensor_data(row, sensor):
    if row is None or row.job_status != 'ready' or not row.result or not row.artifact_path:
        raise ValueError('Load characterization first.')
    filename = row.result['files'].get(sensor)
    if filename is None:
        raise ValueError('Select a numeric sensor included in this analysis.')
    path = Path(row.artifact_path) / filename
    if not path.exists():
        raise ValueError('Saved sensor data is missing. Delete this analysis and calculate it again.')
    return path


def detail(row, sensor):
    path = sensor_data(row, sensor)
    cache = path.with_suffix('.json')
    if cache.exists():
        return json.loads(cache.read_text())
    with np.load(path, allow_pickle=False) as data:
        result = engine.details(data['indices'], data['values'])
    temporary = cache.with_name(cache.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def series(row, sensor, start=None, end=None):
    with np.load(sensor_data(row, sensor), allow_pickle=False) as data:
        return engine.temporal(row.result['metadata'], data['indices'], data['values'], start, end)


def export(row, search='', data_type='all', sort='variable', descending=False):
    if row is None or row.job_status != 'ready':
        raise ValueError('Load characterization first.')
    keys = ('variable', 'data_type', 'valid_n', *engine.STAT_KEYS)
    if sort not in keys or data_type not in ('all', 'numeric', 'text'):
        raise ValueError('Invalid table filter or sort column.')
    rows = [r for r in row.result['summary'] if search.lower() in r['variable'].lower()
            and (data_type == 'all' or r['data_type'] == data_type)]
    present = [r for r in rows if r[sort] is not None]
    present.sort(key=lambda r: r[sort].lower() if isinstance(r[sort], str) else r[sort], reverse=descending)
    rows = present + [r for r in rows if r[sort] is None]
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['Variable', 'Type', 'N valid', 'Mean', 'Median', 'Std', 'IQR', 'Min', 'Q01', 'Q05', 'Q25', 'Q75', 'Q95', 'Q99', 'Max'])
    for item in rows:
        # Prevent spreadsheet formulas in user-controlled column names.
        name = item['variable']
        name = "'" + name if name.startswith(('=', '+', '-', '@', '\t', '\r')) else name
        writer.writerow([name, *[item[key] for key in keys[1:]]])
    return stream.getvalue()


def active_for_analysis(db, analysis_id):
    return db.scalar(select(Run.id).where(Run.analysis_id == analysis_id, Run.job_status.in_(ACTIVE))) is not None


def delete_for_analysis(db, analysis_id):
    rows = list(db.scalars(select(Run).where(Run.analysis_id == analysis_id)))
    if any(row.job_status in ACTIVE for row in rows):
        raise ValueError('Cancel the running characterization before deleting the analysis.')
    for row in rows:
        shutil.rmtree(root(row.id), ignore_errors=True)
        db.delete(row)
    db.flush()


def reconcile(db):
    for row in db.scalars(select(Run).where(Run.job_status.in_(ACTIVE))):
        row.job_status, row.stage, row.eta_seconds = 'failed', 'Interrupted', None
        row.error_message = 'The server restarted during characterization. Retry this calculation.'
        shutil.rmtree(root(row.id), ignore_errors=True)
    db.commit()
