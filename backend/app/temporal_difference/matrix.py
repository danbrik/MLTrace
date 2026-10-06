"""Matrix exports from immutable, already computed temporal pairs."""
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4
from threading import RLock

import numpy as np
from app import models
from app.preprocessing.pipeline import compile_pipeline
from app.schemas import PreprocessingGraph
from app.temporal_difference import service
from app.temporal_difference.engine import grayscale, LABELS

from app.temporal_difference.units import to_percent, NORMALIZATION

LOCK = RLock()


def finished(db, run_id):
    run = db.get(models.TemporalDifferenceRun, run_id)
    return run if run and run.status == 'finished' else None


def manifest(run):
    return json.loads((service.artifact_dir(run.id) / 'manifest.json').read_text())


def pair_index(data, role, deltas):
    grouped = {}
    for pair in data['pairs']:
        if pair['role'] == role and pair['delta_seconds'] in deltas:
            grouped.setdefault(pair['first'], {})[pair['delta_seconds']] = pair['second']
    return {first: others for first, others in grouped.items() if all(d in others for d in deltas)}


def candidates(db, run_id, role, deltas):
    run = finished(db, run_id)
    if not run:
        return None
    if not 1 <= len(deltas) <= 3 or len(set(deltas)) != len(deltas) or not set(deltas) <= set(run.config['deltas_seconds']):
        raise ValueError('Bitte ein bis drei unterschiedliche berechnete Zeitabstände auswählen.')
    data = manifest(run)
    starts = sorted(data['samples'][role][i]['timestamp'] for i in pair_index(data, role, deltas))
    indices = sorted({0, len(starts)//2, len(starts)-1}) if starts else []
    return dict(start_times=starts, suggested=[starts[i] for i in indices])


def state(db, run_id, role):
    run = finished(db, run_id)
    if not run:
        return None
    path = service.artifact_dir(run_id) / f'matrix-{role}.json'
    result = json.loads(path.read_text()) if path.exists() else {'config': None, 'artifact': None, 'warnings': []}
    result.setdefault('render_version', 1)
    result.setdefault('unit', 'raw')
    result['available_deltas'] = sorted({p['delta_seconds'] for p in manifest(run)['pairs'] if p['role'] == role})
    return result


def png_path(db, run_id, role, artifact):
    if not finished(db, run_id):
        return None
    # Published immutable files only, never accept arbitrary user paths.
    if not artifact.startswith(f'matrix-{role}-') or Path(artifact).name != artifact or not artifact.endswith('.png'):
        return None
    path = service.artifact_dir(run_id) / artifact
    return path if path.is_file() else None


def filtered_difference(diff, top_percent):
    if top_percent is None:
        return diff
    threshold = np.quantile(diff, 1-top_percent/100, method='linear')
    return np.ma.masked_where((diff < threshold) | (diff == 0), diff)


def generate(db, run_id, config):
    run = finished(db, run_id)
    if not run:
        return None
    choices = candidates(db, run_id, config.role, config.deltas_seconds)
    if not set(config.start_times) <= set(choices['start_times']):
        raise ValueError('Die Startpunkte müssen gespeicherte Paare für alle ausgewählten Abstände haben.')
    directory = service.artifact_dir(run_id)
    data = manifest(run)
    samples = data['samples'][config.role]
    index = pair_index(data, config.role, config.deltas_seconds)
    lookup = {sample['timestamp']: i for i, sample in enumerate(samples)}
    pipeline = compile_pipeline(PreprocessingGraph.model_validate(run.pipeline_snapshot['graph']))
    artifact = f'matrix-{config.role}-{uuid4().hex}.png'
    warnings, timestamps = [], []
    shape = None
    def read(i):
        nonlocal shape
        sample = samples[i]
        service.check_file(sample)
        array = grayscale(service.read_frozen_image(sample, pipeline), shape)
        shape = array.shape
        return array
    # Persist numerical working maps temporarily, keeping image-series memory bounded.
    with tempfile.TemporaryDirectory(prefix='matrix-', dir=directory) as temp:
        temp = Path(temp)
        low, high, limit = float('inf'), float('-inf'), 0.
        for row, timestamp in enumerate(config.start_times):
            first = lookup[timestamp]
            background = read(first)
            low, high = min(low, float(background.min())), max(high, float(background.max()))
            np.save(temp/f'{row}-base.npy', background)
            targets = []
            for col, delta in enumerate(config.deltas_seconds):
                other = index[first][delta]
                diff = to_percent(np.abs(read(other) - background))
                if not np.isfinite(diff).all():
                    raise ValueError('Die Pixeldifferenz enthält nicht endliche Werte.')
                limit = max(limit, float(diff.max()))
                if not np.any(diff):
                    warnings.append(f'{timestamp} · {delta} s: keine Änderung.')
                np.save(temp/f'{row}-{col}.npy', diff)
                targets.append(dict(delta_seconds=delta, timestamp=samples[other]['timestamp']))
            timestamps.append(dict(start=timestamp, partners=targets))
        metadata = dict(render_version=2, **NORMALIZATION, config=config.model_dump(), dataset=run.dataset_snapshot, pipeline=run.pipeline_snapshot,
            period=run.config[config.role], timestamps=timestamps, difference_limit=limit, background_range=[low, high])
        with LOCK:
            _render(temp, config, low, high, limit, metadata)
            # A deleted parent must not be recreated by an export finishing late.
            db.expire(run)
            if run.status != 'finished':
                raise ValueError('Der Lauf ist nicht mehr verfügbar.')
            os.replace(temp/'matrix.png', directory/artifact)
            result = dict(render_version=2, **NORMALIZATION, config=config.model_dump(), artifact=artifact, warnings=warnings,
                available_deltas=sorted({p['delta_seconds'] for p in data['pairs'] if p['role'] == config.role}))
            pending = directory/f'.{uuid4().hex}.json'
            try:
                pending.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
                os.replace(pending, directory/f'matrix-{config.role}.json')
            except Exception:
                (directory/artifact).unlink(missing_ok=True)
                raise
            finally:
                pending.unlink(missing_ok=True)
    return result


def _render(temp, config, low, high, limit, metadata):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib import colormaps
    rows, cols = len(config.start_times), 1+len(config.deltas_seconds)
    fig = Figure(figsize=(4*cols, 3.4*rows+1.4), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, cols, squeeze=False)
    cmap = colormaps['viridis'].with_extremes(bad='white')
    norm = Normalize(vmin=0, vmax=limit if limit > 0 else 1)
    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]
            if col == 0:
                array = np.load(temp/f'{row}-base.npy')
                ax.imshow(array if high > low else np.full(array.shape, .5), cmap='gray',
                    vmin=low if high>low else 0, vmax=high if high>low else 1, interpolation='nearest', origin='upper')
            else:
                array = np.load(temp/f'{row}-{col-1}.npy')
                shown = filtered_difference(array, config.top_percent)
                if not np.any(array):
                    shown = np.ma.masked_all(array.shape)
                ax.imshow(shown, cmap=cmap, norm=norm, interpolation='nearest', origin='upper')
            ax.tick_params(labelleft=col==0, labelbottom=row==rows-1, left=col==0, bottom=row==rows-1)
            if row == 0:
                ax.set_title('Bild bei t' if col==0 else f'|Bild(t + {config.deltas_seconds[col-1]} s) − Bild(t)|', fontsize=11)
        axes[row, 0].set_ylabel(config.start_times[row].replace('T','\n'), fontsize=9)
    fig.suptitle(LABELS[config.role] + (f' · Top {config.top_percent:g} % pro Differenzbild' if config.top_percent is not None else ''))
    fig.subplots_adjust(left=.14, right=.98, top=.9, bottom=.23 if rows==1 else .15, hspace=.18, wspace=.12)
    fig.supxlabel('x (Pixel)', y=.13 if rows==1 else .08)
    fig.supylabel('y (Pixel)', x=.015)
    cax = fig.add_axes([.32, .06 if rows==1 else .035, .5, .022])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=cax, orientation='horizontal', label='Absolute Pixeländerung (%)')
    fig.savefig(temp/'matrix.png', metadata={'Description': json.dumps(metadata, ensure_ascii=False, allow_nan=False)})
