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


def result_rows(run, subset=None, sensor=0, scaled=False, start=None, end=None, include_latent=False):
    manifest = load_manifest(run)
    directory = artifact_dir(run.id)
    snapshot = run.snapshot
    columns = manifest['sensor_order']
    if not 0 <= sensor < len(columns):
        raise ValueError('Unbekannter Sensor.')
    if subset is not None and subset not in SUBSETS:
        raise ValueError('Unbekannte Gruppe.')
    with np.load(directory / 'input.npz') as data:
        values, times, groups = data['scaled'] if scaled else data['values'], data['timestamps'], data['groups']
        segments, interval_indices = data['segment_ids'], data['interval_indices']
    scaler = snapshot['preview']['scaler']
    denominator = 1 if scaled else scaler['denominator'][sensor]
    minimum = 0 if scaled else scaler['minimum'][sensor]
    length = snapshot['window_length']
    start_ns = boundary(start).value if start else None
    end_ns = boundary(end).value if end else None
    for chunk in manifest['chunks']:
        with np.load(directory / chunk['file']) as block:
            for i, endpoint in enumerate(block['end_indices']):
                group = SUBSETS[groups[endpoint]]
                ts = timestamp(times[endpoint])
                if subset and group != subset or start_ns is not None and times[endpoint] < start_ns or end_ns is not None and times[endpoint] > end_ns:
                    continue
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


def series(run, subset='test', sensor=0, scaled=False, offset=0, limit=2000, start=None, end=None):
    rows, total = [], 0
    for row in result_rows(run, subset, sensor, scaled, start, end):
        if offset <= total < offset + limit:
            rows.append(row)
        total += 1
    raw = []
    with np.load(artifact_dir(run.id) / 'input.npz') as inputs:
        indices = np.flatnonzero(inputs['groups'] == SUBSETS.index(subset))
        if rows:
            first = rows[0]['endpoint_index'] - run.snapshot['window_length'] + 1
            last = rows[-1]['endpoint_index']
            indices = indices[(indices >= first) & (indices <= last)]
        complete = set(inputs['endpoints'].tolist())
        # Bound display payload independently of result pagination; exports retain all results.
        for index in indices[:20000]:
            raw.append(dict(timestamp=timestamp(inputs['timestamps'][index]),
                            original=float(inputs['scaled' if scaled else 'values'][index, sensor]),
                            segment_id=int(inputs['segment_ids'][index]),
                            warmup=int(index) not in complete))
    return dict(rows=rows, raw=raw, total=total, offset=offset, limit=limit, metadata=load_manifest(run))


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
