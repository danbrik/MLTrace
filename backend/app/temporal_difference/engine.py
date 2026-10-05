from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from bisect import bisect_left
from math import ceil

ROLES = ("reference", "comparison")
LABELS = {"reference": "Referenz", "comparison": "Vergleich"}
BERLIN = ZoneInfo("Europe/Berlin")


def utc_instant(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(UTC)
    candidates = {value.replace(tzinfo=BERLIN, fold=fold).astimezone(UTC) for fold in (0, 1)
                  if value.replace(tzinfo=BERLIN, fold=fold).astimezone(UTC).astimezone(BERLIN).replace(tzinfo=None) == value}
    if len(candidates) != 1:
        reason = "Mehrdeutige" if candidates else "Nicht existente"
        raise ValueError(f"{reason} Ortszeit in Europe/Berlin: {value.isoformat(sep=' ')}.")
    return candidates.pop()


def select_pairs(records, config):
    if config.selection_version == 2:
        return stratified_pairs(records, config)
    return exact_pairs(records, config)


def exact_pairs(records, config, validate_only=False):
    ordered = sorted({row.file_path: row for row in records}.values(), key=lambda row: (row.timestamp_parsed, row.file_path))
    samples, pairs, periods, errors = {}, [], {}, []
    for role in ROLES:
        interval = getattr(config, role)
        utc_instant(interval.start)
        utc_instant(interval.end)
        group = [row for row in ordered if interval.start <= row.timestamp_parsed <= interval.end]
        by_time = {}
        samples[role] = []
        for index, row in enumerate(group):
            instant = utc_instant(row.timestamp_parsed)
            if instant in by_time:
                raise ValueError(f"{LABELS[role]}: Mehrere Dateien am Zeitpunkt {row.timestamp_parsed.isoformat(sep=' ')}.")
            by_time[instant] = index
            samples[role].append({"file_path": row.file_path, "timestamp": row.timestamp_parsed.isoformat(), "utc": instant.isoformat()})
        if validate_only:
            continue
        counts = []
        for delta in config.deltas_seconds:
            found = 0
            for instant, index in by_time.items():
                try:
                    target = instant + timedelta(seconds=delta)
                except OverflowError:
                    continue
                other = by_time.get(target)
                if other is not None:
                    pairs.append({"role": role, "delta_seconds": delta, "first": index, "second": other})
                    found += 1
            counts.append({"delta_seconds": delta, "pair_count": found, "missing_targets": len(group) - found})
        periods[role] = {"image_count": len(group), "deltas": counts}
        if not any(row["pair_count"] for row in counts):
            errors.append(f"{LABELS[role]}: Für keinen Zeitabstand sind vollständige Bildpaare vorhanden.")
    return samples, pairs, {"periods": periods, "errors": errors}


def absolute_change(first, second):
    # Inputs are pipeline outputs; no scaling, clipping or intensity transforms.
    first = grayscale(first)
    second = grayscale(second, first.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean(np.abs(second - first)))
    if not np.isfinite(value):
        raise ValueError("Die Pixeländerung enthält nicht endliche Werte.")
    return value


def grayscale(image, shape=None):
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2 or not array.size:
        raise ValueError("Die Preprocessing-Pipeline muss einkanalige Graustufenbilder liefern.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"Uneinheitliche Bildgrößen: {array.shape} statt {shape}.")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Das Bild enthält nicht endliche Pixelwerte.")
    return array


def statistics(values):
    if not len(values):
        return dict(pair_count=0, median=None, q1=None, q3=None, iqr=None)
    q1, median, q3 = np.quantile(values, [.25, .5, .75], method="linear")
    return dict(pair_count=len(values), median=float(median), q1=float(q1), q3=float(q3), iqr=float(q3 - q1))


def stratified_pairs(records, config):
    # Reuse legacy ordering, duplicate-timestamp checks and timezone validation.
    samples, _, _ = exact_pairs(records, config, validate_only=True)
    pairs, periods, errors = [], {}, []
    for role in ROLES:
        interval = getattr(config, role)
        begin, end = utc_instant(interval.start), utc_instant(interval.end)
        duration = (end - begin).total_seconds() - max(config.deltas_seconds)
        if duration < 0:
            errors.append(f"{LABELS[role]}: Zeitraum ist kürzer als der größte Abstand.")
        last = begin + timedelta(seconds=max(0, duration))
        blocks = max(1, ceil(duration / config.block_seconds)) if duration >= 0 else 0
        times = [datetime.fromisoformat(row['utc']) for row in samples[role]]
        buckets, missing = {}, dict.fromkeys(config.deltas_seconds, 0)
        candidates = 0
        for index, instant in enumerate(times):
            if duration < 0 or instant > last:
                continue
            candidates += 1
            partners = []
            for delta in config.deltas_seconds:
                target = instant + timedelta(seconds=delta)
                pos = bisect_left(times, target)
                choices = [i for i in (pos-1, pos) if 0 <= i < len(times) and i != index
                           and abs((times[i]-target).total_seconds()) <= .5]
                other = min(choices, key=lambda i: (abs((times[i]-target).total_seconds()), times[i])) if choices else None
                if other is None:
                    missing[delta] += 1
                partners.append(other)
            if all(other is not None for other in partners):
                block = min(blocks-1, int((instant-begin).total_seconds() // config.block_seconds))
                buckets.setdefault(block, []).append((index, partners))
        rng = np.random.default_rng(config.seed)
        selected = []
        for block in sorted(buckets):
            options = buckets[block]
            index, partners = options[int(rng.integers(len(options)))]
            selected.append((index, partners))
        # Freeze only files actually needed; pair indices address this compact list.
        used = sorted({i for first, others in selected for i in [first, *others]})
        remap = {old: new for new, old in enumerate(used)}
        for first, others in selected:
            for delta, other in zip(config.deltas_seconds, others):
                pairs.append(dict(role=role, delta_seconds=delta, first=remap[first], second=remap[other]))
        image_count = len(samples[role])
        samples[role] = [samples[role][i] for i in used]
        periods[role] = dict(image_count=image_count, start_range_start=interval.start.isoformat(),
            start_range_end=last.astimezone(BERLIN).replace(tzinfo=None).isoformat() if duration >= 0 else None,
            block_count=blocks, candidate_count=candidates, valid_candidates=sum(map(len, buckets.values())),
            selected_start_count=len(selected), empty_blocks=blocks-len(buckets),
            deltas=[dict(delta_seconds=d, pair_count=len(selected), missing_targets=missing[d]) for d in config.deltas_seconds])
        if not selected and duration >= 0:
            errors.append(f"{LABELS[role]}: Kein Startzeitpunkt hat Partnerbilder für alle Abstände innerhalb ±0,5 s.")
    return samples, pairs, dict(periods=periods, errors=errors)
