"""Deterministic image selection and signed grayscale rendering."""
from bisect import bisect_left
from datetime import datetime

import numpy as np

from app.reference_image.schemas import ReferenceImageConfig, SelectionCount, SelectionPreview


def select_records(records, config: ReferenceImageConfig):
    ordered = sorted({item.file_path: item for item in records}.values(),
                     key=lambda item: (item.timestamp_parsed, item.file_path))
    selected = {}
    counts = {}
    errors = []
    for role in ("reference", "anomaly"):
        interval = getattr(config, role)
        candidates = [item for item in ordered if interval.start <= item.timestamp_parsed <= interval.end]
        random = role == "reference" and interval.mode == "random"
        if random:
            if interval.count > len(candidates):
                errors.append(f"Referenz: {interval.count} Zufallsbilder angefordert, aber nur {len(candidates)} verfügbar.")
                chosen = []
            else:
                indices = sorted(np.random.default_rng(interval.seed).choice(len(candidates), interval.count, replace=False))
                chosen = [candidates[int(index)] for index in indices]
        else:
            chosen = candidates[interval.sampling_rate - 1::interval.sampling_rate]
        label = "Referenz" if role == "reference" else "Anomalie"
        if not chosen:
            errors.append(f"{label}: Die Auswahl enthält keine Bilder.")
        selected[role] = [{"file_path": item.file_path, "timestamp": item.timestamp_parsed.isoformat()} for item in chosen]
        counts[role] = SelectionCount(available=len(candidates), selected=len(chosen),
                                      remainder=0 if random else len(candidates) % interval.sampling_rate)
    return selected, SelectionPreview(**counts, errors=errors)


def grayscale(image, shape=None):
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2 or not array.size:
        raise ValueError("Die Preprocessing-Pipeline muss einkanalige Graustufenbilder liefern.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"Uneinheitliche Bildgrößen: {array.shape} statt {shape}.")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Das Bild enthält nicht endliche Pixelwerte.")
    return array


def render_difference(difference, limit: float):
    if limit == 0:
        return np.full(difference.shape, 128, dtype=np.uint8)
    relative = np.clip(difference / limit, -1, 1)
    # Zero is exactly 128; the negative half has 128 steps, the positive 127.
    return np.rint(128 + relative * np.where(relative < 0, 128, 127)).astype(np.uint8)


def resolve_frame(frames: list[dict], timestamp: datetime):
    if not frames:
        raise ValueError("Es sind keine fertigen Frames verfügbar.")
    times = [datetime.fromisoformat(frame["timestamp"]) for frame in frames]
    index = min(bisect_left(times, timestamp), len(frames) - 1)
    return {"requested_timestamp": timestamp, "exact": times[index] == timestamp, "frame": frames[index]}
