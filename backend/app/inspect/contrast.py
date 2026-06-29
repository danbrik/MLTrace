"""Contrast-enhancement helpers for Inspect videos.

The contrast-enhanced mode subtracts a rolling mean reference image (built from
the first N processed frames) from every frame, applies a fixed shift, clips the
result to ``[0, vmax]`` and rescales it to 8-bit. An optional centered moving
average over ``+/- ma_radius`` frames smooths the result. All math is performed
in a single-channel intensity space normalised to ``[0, 65535]`` so that the
shift/vmax semantics match the legacy 16-bit TIFF workflow.
"""

from __future__ import annotations

import numpy as np

# All contrast math happens in a 16-bit intensity space so that the shift/vmax
# values carry the same meaning as the legacy raw-TIFF workflow.
INTENSITY_SCALE = 65535.0


def to_intensity_16scale(image: np.ndarray) -> np.ndarray:
    """Reduce a preprocessing output to single-channel float intensity in [0, 65535].

    Uses the same absolute, dtype-based scaling as the rest of MLTrace so that a
    given source value always maps to the same intensity. Multi-channel images
    are reduced to grayscale by averaging the colour channels.
    """
    array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        elif array.shape[2] >= 3:
            array = array[..., :3].astype(np.float64).mean(axis=2)
        else:
            array = array.astype(np.float64).mean(axis=2)
    elif array.ndim != 2:
        raise ValueError(f"Contrast intensity expects a 2D or 3D image; got shape {array.shape}.")

    if array.dtype == np.uint8:
        values = array.astype(np.float64) / 255.0
    elif np.issubdtype(array.dtype, np.bool_):
        values = array.astype(np.float64)
    elif np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        values = array.astype(np.float64)
        if info.min < 0:
            values = (values - info.min) / float(info.max - info.min)
        else:
            values = values / float(info.max)
    elif np.issubdtype(array.dtype, np.floating):
        values = np.nan_to_num(array.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        values = np.clip(values, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype for contrast intensity: {array.dtype}.")

    return (values * INTENSITY_SCALE).astype(np.float32)


def enhance_to_uint8(
    intensity16: np.ndarray,
    reference16: np.ndarray,
    shift: float,
    vmax: float,
) -> np.ndarray:
    """Apply ``diff -> shift -> clip -> scale`` to a single frame, returning uint8 grayscale."""
    diff = intensity16.astype(np.float32) - reference16.astype(np.float32)
    shifted = diff + float(shift)
    if vmax <= 0:
        return np.zeros(shifted.shape, dtype=np.uint8)
    clipped = np.clip(shifted, 0.0, float(vmax))
    scale = 255.0 / float(vmax)
    return np.round(clipped * scale).astype(np.uint8)


def moving_average_uint8(frames: list[np.ndarray], index: int, radius: int) -> np.ndarray:
    """Centered moving average over uint8 frames within ``+/- radius`` of ``index``."""
    if radius <= 0:
        return frames[index]
    start = max(0, index - radius)
    end = min(len(frames), index + radius + 1)
    window = frames[start:end]
    if len(window) == 1:
        return window[0]
    avg = np.mean(np.stack(window, axis=0).astype(np.float32), axis=0)
    return np.round(avg).astype(np.uint8)
