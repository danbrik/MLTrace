from __future__ import annotations

import csv
import math
import json
import struct
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.redundancy.engine import DEFAULT_MISSING_TOKENS, AnalysisCancelled, parse_number, parse_timestamp

VERSION = 1


def missing_runs(valid_indices: list[int], count: int) -> list[list[int]]:
    """Half-open runs of missing expected observations, never timestamp distances."""
    runs = []
    cursor = 0
    for index in sorted(valid_indices):
        if index > cursor:
            runs.append([cursor, index])
        cursor = index + 1
    if cursor < count:
        runs.append([cursor, count])
    return runs


def analyze(path: Path, delimiter: str, params: dict, row_count: int, progress=lambda *args: None, cancelled=lambda: False):
    with TemporaryDirectory(prefix='mltrace-quality-') as directory:
        with ExitStack() as stack:
            return _analyze(path, delimiter, params, row_count, progress, cancelled, Path(directory), stack)


def _analyze(path, delimiter, params, row_count, progress, cancelled, directory, stack):
    columns = params['selected_columns']
    start = datetime.fromisoformat(params['start_timestamp'])
    end = datetime.fromisoformat(params['end_timestamp'])
    step_us = params['interval_seconds'] * 1_000_000
    tokens = set(DEFAULT_MISSING_TOKENS)

    def report(value, stage):
        if cancelled():
            raise AnalysisCancelled('Calculation cancelled.')
        progress(value, stage)

    # First scan establishes the source-wide origin, independent of the selected range.
    origin = None
    invalid_time = 0
    report(0.01, 'Reading CSV')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for position, row in enumerate(csv.DictReader(handle, delimiter=delimiter)):
            timestamp = parse_timestamp(row.get(params['time_column']) or '')
            if timestamp is None:
                invalid_time += 1
            elif origin is None or timestamp < origin:
                origin = timestamp
            if position % 2000 == 0:
                report(0.01 + 0.19 * min(1, position / max(1, row_count)), 'Reading CSV')
    if origin is None:
        raise ValueError('The selected time column contains no valid timestamps.')

    def micros(delta):
        return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds

    first = max(0, -(-micros(start - origin) // step_us))
    last = micros(end - origin) // step_us
    count = last - first + 1
    if count <= 0:
        raise ValueError('The selected range contains no expected grid points.')
    grid_start = origin + timedelta(microseconds=first * step_us)
    # Spool valid observations to disk; only one sensor's values are held in memory.
    spools = {name: stack.enter_context((directory / str(i)).open('w+b')) for i, name in enumerate(columns)}
    invalid = dict.fromkeys(columns, 0)
    seen = set()
    present = set()
    duplicates = backwards = offgrid = 0
    previous = None
    report(0.22, 'Checking time axis and resolving duplicates')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for position, row in enumerate(csv.DictReader(handle, delimiter=delimiter)):
            if position % 2000 == 0:
                report(0.22 + 0.33 * min(1, position / max(1, row_count)), 'Checking time axis and resolving duplicates')
            timestamp = parse_timestamp(row.get(params['time_column']) or '')
            if timestamp is None or not start <= timestamp <= end:
                continue
            duplicates += int(timestamp in seen)
            seen.add(timestamp)
            backwards += int(previous is not None and timestamp < previous)
            previous = timestamp
            offset = micros(timestamp - origin)
            if offset % step_us:
                offgrid += 1
                continue
            index = offset // step_us - first
            present.add(index)
            for name in columns:
                raw = row.get(name)
                if params['data_types'][name] == 'numeric':
                    value, state = parse_number(raw, tokens)
                    invalid[name] += int(state == 'invalid')
                    if state != 'valid':
                        continue
                else:
                    if raw is None or raw.strip().lower() in tokens:
                        continue
                    value = raw.strip()
                if params['data_types'][name] == 'numeric':
                    spools[name].write(struct.pack('<qd', index, value))
                else:
                    spools[name].write((json.dumps([index, value], ensure_ascii=False) + '\n').encode('utf-8'))

    quality = []
    all_runs = {}
    for position, name in enumerate(columns):
        report(0.55 + 0.35 * position / len(columns), f'Calculating sensor statistics ({position + 1}/{len(columns)})')
        sensor = {}
        conflicts = set()
        spool = spools[name]
        spool.seek(0)
        numeric = params['data_types'][name] == 'numeric'
        def observations():
            if numeric:
                while chunk := spool.read(16 * 4096):
                    yield from struct.iter_unpack('<qd', chunk)
            else:
                for line in spool:
                    yield json.loads(line)
        for n, (index, value) in enumerate(observations()):
            if n % 10000 == 0 and cancelled():
                raise AnalysisCancelled('Calculation cancelled.')
            if index not in sensor:
                sensor[index] = value
            elif sensor[index] != value:
                conflicts.add(index)
        spool.close()
        runs = missing_runs(list(sensor), count)
        all_runs[name] = runs
        valid = len(sensor)
        unique = len(set(sensor.values()))
        numeric = params['data_types'][name] == 'numeric'
        item = dict(sensor=name, data_type=params['data_types'][name], valid_n=valid,
                    missing_percent=(count - valid) / count * 100,
                    longest_gap_minutes=max((b - a for a, b in runs), default=0) * params['interval_seconds'] / 60,
                    unique=unique, constant=valid >= 2 and unique == 1,
                    invalid_n=invalid[name], conflict_n=len(conflicts),
                    min=None, q01=None, median=None, q99=None, max=None, iqr=None, std=None)
        if numeric and valid:
            array = np.array(list(sensor.values()), dtype=float)
            q01, q25, median, q75, q99 = np.quantile(array, [.01, .25, .5, .75, .99], method='linear')
            with np.errstate(over='ignore', invalid='ignore'):
                item.update(min=float(array.min()), q01=float(q01), median=float(median), q99=float(q99),
                            max=float(array.max()), iqr=float(q75 - q25), std=float(np.std(array, ddof=1)) if valid > 1 else None)
            for key in ('min', 'q01', 'median', 'q99', 'max', 'iqr', 'std'):
                if item[key] is not None and not math.isfinite(item[key]):
                    item[key] = None
        quality.append(item)
    report(0.92, 'Preparing missingness heatmap')
    result = dict(summary=dict(start_timestamp=params['start_timestamp'], end_timestamp=params['end_timestamp'],
                  interval_seconds=params['interval_seconds'], expected_timepoints=count, present_timepoints=len(present),
                  missing_timepoints=count - len(present), coverage_percent=len(present) / count * 100,
                  duplicate_timestamps=duplicates, non_monotone_timestamps=backwards, invalid_timestamps=invalid_time,
                  off_grid_rows=offgrid, variable_count=len(columns),
                  variables_with_missing=sum(x['valid_n'] < count for x in quality),
                  constant_variables=sum(x['constant'] for x in quality),
                  duplicate_conflicts=sum(x['conflict_n'] for x in quality)), quality=quality,
                  grid_start=grid_start.isoformat(), grid_count=count, interval_seconds=params['interval_seconds'])
    report(0.97, 'Saving results')
    return result, all_runs


def heatmap(result: dict, runs: dict, start: datetime | None = None, end: datetime | None = None):
    origin = datetime.fromisoformat(result['grid_start'])
    seconds = result['interval_seconds']
    count = result['grid_count']
    lo = max(0, math.ceil((start - origin).total_seconds() / seconds)) if start else 0
    hi = min(count, math.floor((end - origin).total_seconds() / seconds) + 1) if end else count
    if hi <= lo:
        return dict(x=[], y=list(runs), z=[], aggregated=False, bin_end=[])
    width = max(1, math.ceil((hi - lo) / 1000))
    bins = [(a, min(hi, a + width)) for a in range(lo, hi, width)]
    z = []
    for sensor_runs in runs.values():
        row = []
        cursor = 0
        for a, b in bins:
            while cursor < len(sensor_runs) and sensor_runs[cursor][1] <= a:
                cursor += 1
            j, missing = cursor, 0
            while j < len(sensor_runs) and sensor_runs[j][0] < b:
                left, right = sensor_runs[j]
                missing += max(0, min(b, right) - max(a, left))
                j += 1
            row.append(missing / (b - a))
        z.append(row)
    return dict(x=[(origin + timedelta(seconds=a * seconds)).isoformat() for a, _ in bins],
                bin_end=[(origin + timedelta(seconds=(b - 1) * seconds)).isoformat() for _, b in bins],
                y=list(runs), z=z, aggregated=width > 1)
