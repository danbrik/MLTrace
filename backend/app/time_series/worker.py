import json
import os
import sys
import time
import traceback

from app import models
from app.database import SessionLocal
from app.time_series.engine import execute, Cancelled
from app.time_series.training_service import artifact_dir


def main(run_id):
    # Scheduler commits ownership after launching the subprocess.
    for _ in range(300):
        with SessionLocal() as db:
            row = db.get(models.TimeSeriesRun, run_id)
            if not row or row.status in ('aborted', 'failed'):
                return
            if row.status == 'running' and row.pid == os.getpid():
                break
        time.sleep(.1)
    else:
        raise RuntimeError('Scheduler dispatch timeout')
    started, last_check = time.monotonic(), 0.0
    def progress(**values):
        nonlocal last_check
        if not values and time.monotonic() - last_check < .5:
            return
        with SessionLocal() as db:
            row = db.get(models.TimeSeriesRun, run_id)
            if row.cancel_requested or row.status != 'running':
                raise Cancelled()
            for key, value in values.items():
                setattr(row, key, value)
            row.heartbeat_at = models.utc_now()
            row.duration_seconds = time.monotonic() - started
            if 'epoch' in values:
                db.add(models.TimeSeriesEpochMetric(run_id=run_id, epoch=values['epoch'],
                       train_loss=values['train_loss'], val_loss=values['val_loss'], details={}))
            db.commit()
        last_check = time.monotonic()
    try:
        directory = artifact_dir(run_id)
        manifest = execute(directory, json.loads((directory / 'snapshot.json').read_text()), progress)
        progress()
        with SessionLocal() as db:
            row = db.get(models.TimeSeriesRun, run_id)
            if row.cancel_requested:
                raise Cancelled()
            row.status = row.current_step = 'finished'
            row.checkpoint, row.result = manifest['checkpoint'], manifest
            row.selected_epoch, row.selected_metric = manifest['checkpoint']['epoch'], manifest['checkpoint']['metric_value']
            row.ended_at, row.duration_seconds = models.utc_now(), time.monotonic() - started
            db.commit()
    except Exception as exc:
        traceback.print_exc()
        with SessionLocal() as db:
            row = db.get(models.TimeSeriesRun, run_id)
            row.status = row.current_step = 'aborted' if isinstance(exc, Cancelled) else 'failed'
            row.error_message = None if isinstance(exc, Cancelled) else str(exc)
            row.ended_at, row.duration_seconds = models.utc_now(), time.monotonic() - started
            db.commit()


if __name__ == '__main__':
    from app.database import project_context
    from app.config import get_settings
    with project_context(get_settings().database_url, os.environ.get('MLTRACE_ARTIFACT_DIR', str(artifact_dir(0).parent.parent))):
        main(int(sys.argv[1]))
