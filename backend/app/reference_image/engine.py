"""Deterministic image selection and signed grayscale rendering."""
from bisect import bisect_left
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from app.reference_image.schemas import ReferenceImageConfig, SelectionCount, SelectionPreview


_SOURCE_TIMEZONE = ZoneInfo("Europe/Berlin")


def selection_config(config: ReferenceImageConfig) -> ReferenceImageConfig:
    """Extend only the selection interval, preserving the user's stored configuration."""
    try:
        start = config.anomaly.start - timedelta(minutes=config.start_offset_minutes)
    except OverflowError as exc:
        raise ValueError("Offset Start liegt außerhalb des unterstützten Datumsbereichs.") from exc
    return config.model_copy(update={"anomaly": config.anomaly.model_copy(update={"start": start})})


def utc_timestamp_label(value: datetime) -> str:
    """Resolve dataset-local Berlin times without guessing at a DST fold or gap."""
    if value.tzinfo is not None:
        instant = value.astimezone(UTC)
    else:
        candidates = set()
        for fold in (0, 1):
            candidate = value.replace(tzinfo=_SOURCE_TIMEZONE, fold=fold).astimezone(UTC)
            if candidate.astimezone(_SOURCE_TIMEZONE).replace(tzinfo=None) == value:
                candidates.add(candidate)
        if len(candidates) != 1:
            reason = "Mehrdeutige" if candidates else "Nicht existente"
            raise ValueError(f"{reason} Ortszeit in Europe/Berlin: {value.isoformat(sep=' ')}. "
                             "UTC-Beschriftung ist bei dieser Zeitumstellung nicht eindeutig möglich.")
        instant = candidates.pop()
    return instant.strftime("%Y-%m-%d %H:%M:%S (UTC)")


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


def shift_clip_difference(difference, shift: float, clip_min: int, clip_max: int):
    """Clip in floating point before conversion to avoid unsigned wraparound.

    Fractional reference differences are rounded to nearest, ties to even.
    Stored intensities are not stretched to fill the uint16 range.
    """
    shifted = np.asarray(difference, dtype=np.float64) + shift
    return np.rint(np.clip(shifted, clip_min, clip_max)).astype(np.uint16)


def stamp_uint16(image, timestamp: datetime, clip_min: int, clip_max: int, *, label: str | None = None):
    from app.video import timestamp_overlay
    overlay = np.asarray(timestamp_overlay(image.shape[1], image.shape[0], timestamp, label=label))
    alpha = overlay[:, :, 3].astype(np.float64) / 255
    ink = clip_min + overlay[:, :, 0].astype(np.float64) / 255 * (clip_max - clip_min)
    return np.rint(image.astype(np.float64) * (1 - alpha) + ink * alpha).astype(np.uint16)


def preview_uint16(image, clip_min: int, clip_max: int):
    """A fixed display mapping, separate from the lossless 16-bit pixels."""
    values = (image.astype(np.float64) - clip_min) / (clip_max - clip_min)
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
