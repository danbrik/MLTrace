import csv
import io
import json
import zipfile

import numpy as np
from app.time_series.data import timestamp, SUBSETS
from app.time_series.service import boundary
from app.time_series.training_service import artifact_dir, Conflict


def load_manifest(run):
    if run.status != 'finished':
        raise Conflict('Ergebnisse stehen nach erfolgreichem Abschluss zur Verfügung.')
    return json.loads((artifact_dir(run.id) / 'manifest.json').read_text())


def _load_inputs(directory, scaled):
    # NpzFile does not cache decompressed members. Read each array once, outside
    # every row loop; keep only the requested value space in memory.
    with np.load(directory / 'input.npz') as archive:
        inputs = {key: archive[key] for key in ('timestamps', 'groups', 'segment_ids', 'interval_indices', 'endpoints')}
        inputs['values'] = archive['scaled' if scaled else 'values']
    return inputs


def _selection(inputs, subset, start, end):
    if subset is not None and subset not in SUBSETS:
        raise ValueError('Unbekannte Gruppe.')
    start_ns = boundary(start).value if start else None
    end_ns = boundary(end).value if end else None
    endpoints = inputs['endpoints']
    mask = np.ones(len(endpoints), dtype=bool)
    if subset is not None:
        mask &= inputs['groups'][endpoints] == SUBSETS.index(subset)
    if start_ns is not None:
        mask &= inputs['timestamps'][endpoints] >= start_ns
    if end_ns is not None:
        mask &= inputs['timestamps'][endpoints] <= end_ns
    return np.flatnonzero(mask), start_ns, end_ns


def _result_rows(run, manifest, inputs, positions, sensor, scaled, include_latent=False):
    columns = manifest['sensor_order']
    if not 0 <= sensor < len(columns):
        raise ValueError('Unbekannter Sensor.')
    snapshot = run.snapshot
    directory = artifact_dir(run.id)
    values, times, groups = inputs['values'], inputs['timestamps'], inputs['groups']
    segments, interval_indices = inputs['segment_ids'], inputs['interval_indices']
    scaler = snapshot['preview']['scaler']
    denominator = 1 if scaled else scaler['denominator'][sensor]
    minimum = 0 if scaled else scaler['minimum'][sensor]
    length = snapshot['window_length']
    # Evaluation writes chunks in input.endpoints order. Counts form an index
    # also for existing runs, so unrelated blocks need not be opened at all.
    cursor = 0
    for chunk in manifest['chunks']:
        stop = cursor + chunk['count']
        left, right = np.searchsorted(positions, [cursor, stop])
        local = positions[left:right] - cursor
        cursor = stop
        if not len(local):
            continue
        with np.load(directory / chunk['file']) as archive:
            keys = ['end_indices', 'window_score', 'endpoint_score', 'reconstruction']
            keys += [key for key in ('cascade', 'second', 'variance') if key in archive.files]
            if include_latent:
                keys.append('latent')
            block = {key: archive[key] for key in keys}
        for i in local:
            endpoint = block['end_indices'][i]
            group = SUBSETS[groups[endpoint]]
            ts = timestamp(times[endpoint])
            interval = snapshot['split']['intervals'][int(interval_indices[endpoint])]
            row = dict(run_id=run.id, sensor=columns[sensor], value_space='scaled' if scaled else 'original', endpoint_index=int(endpoint), timestamp=ts,
                       window_start=timestamp(times[endpoint - length + 1]), window_end=ts,
                       segment_id=int(segments[endpoint]), subset=group, tags=interval.get('tags', []),
                       window_score=float(block['window_score'][i]), endpoint_score=float(block['endpoint_score'][i]),
                       original=float(values[endpoint, sensor]), reconstruction=float(block['reconstruction'][i, sensor]) * denominator + minimum,
                       z_sensor_ref=f'{chunk["file"]}:{i}')
            if 'cascade' in block:
                row['cascade'] = float(block['cascade'][i, sensor]) * denominator + minimum
                row['second'] = float(block['second'][i, sensor]) * denominator + minimum
            if 'variance' in block:
                row['std'] = float(np.sqrt(block['variance'][i, sensor])) * denominator
            if include_latent:
                row['z_sensor'] = block['latent'][i].tolist()
            yield row


def result_rows(run, subset=None, sensor=0, scaled=False, start=None, end=None, include_latent=False):
    manifest = load_manifest(run)
    inputs = _load_inputs(artifact_dir(run.id), scaled)
    positions, _, _ = _selection(inputs, subset, start, end)
    yield from _result_rows(run, manifest, inputs, positions, sensor, scaled, include_latent)


def series(run, subset='test', sensor=0, scaled=False, offset=0, limit=2000, start=None, end=None):
    manifest = load_manifest(run)
    inputs = _load_inputs(artifact_dir(run.id), scaled)
    positions, start_ns, end_ns = _selection(inputs, subset, start, end)
    total = len(positions)
    rows = list(_result_rows(run, manifest, inputs, positions[offset:offset + limit], sensor, scaled))
    indices = np.flatnonzero(inputs['groups'] == SUBSETS.index(subset))
    if rows:
        first = rows[0]['endpoint_index'] - run.snapshot['window_length'] + 1
        last = rows[-1]['endpoint_index']
        indices = indices[(indices >= first) & (indices <= last)]
    elif total or offset:
        indices = indices[:0]
    # A time-filtered empty response must not leak unrelated original samples.
    if start_ns is not None:
        indices = indices[inputs['timestamps'][indices] >= start_ns]
    if end_ns is not None:
        indices = indices[inputs['timestamps'][indices] <= end_ns]
    indices = indices[:20000]
    warmup = ~np.isin(indices, inputs['endpoints'], assume_unique=True)
    raw = [dict(timestamp=timestamp(inputs['timestamps'][index]),
                original=float(inputs['values'][index, sensor]),
                segment_id=int(inputs['segment_ids'][index]), warmup=bool(is_warmup))
           for index, is_warmup in zip(indices, warmup)]
    return dict(rows=rows, raw=raw, total=total, offset=offset, limit=limit, metadata=manifest)


def export_csv(run, subset=None, sensor=0, scaled=False):
    rows = result_rows(run, subset, sensor, scaled)
    stream = io.StringIO()
    writer = None
    for row in rows:
        row['tags'] = json.dumps(row['tags'], ensure_ascii=False)
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        yield stream.getvalue()
        stream.seek(0)
        stream.truncate(0)


def latent_export(run):
    manifest = load_manifest(run)
    directory = artifact_dir(run.id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('metadata.json', json.dumps(dict(manifest=manifest, snapshot=run.snapshot), ensure_ascii=False, indent=2))
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['run_id', 'timestamp', 'window_start', 'window_end', 'segment_id', 'subset', 'tags', 'window_score', 'endpoint_score', 'z_sensor_ref'])
        for row in result_rows(run):
            writer.writerow([row[key] if key != 'tags' else json.dumps(row[key]) for key in
                             ['run_id', 'timestamp', 'window_start', 'window_end', 'segment_id', 'subset', 'tags', 'window_score', 'endpoint_score', 'z_sensor_ref']])
        archive.writestr('endpoints.csv', output.getvalue())
        for chunk in manifest['chunks']:
            with np.load(directory / chunk['file']) as data:
                payload = io.BytesIO()
                np.savez_compressed(payload, end_indices=data['end_indices'], z_sensor=data['latent'])
                archive.writestr(chunk['file'], payload.getvalue())
    return buffer.getvalue()


def replay(run, endpoint):
    import torch
    from app.time_series.networks import build_model
    load_manifest(run)
    directory = artifact_dir(run.id)
    snapshot = run.snapshot
    with np.load(directory / 'input.npz') as data:
        if endpoint not in data['endpoints']:
            raise ValueError('Dieser Zeitpunkt ist kein vollständiger Fensterendpunkt.')
        x = data['scaled'][endpoint - snapshot['window_length'] + 1:endpoint + 1]
        times = data['timestamps'][endpoint - snapshot['window_length'] + 1:endpoint + 1]
    model = build_model(run.kind, len(x), x.shape[1], snapshot['model']['config'])
    model.load_state_dict(torch.load(directory / 'weights.pt', weights_only=True, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x[None]))
    return dict(timestamps=[timestamp(t) for t in times], scaled_input=x.tolist(),
                outputs={key: value[0].numpy().tolist() for key, value in output.items()}, checkpoint=run.checkpoint)
