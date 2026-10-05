"""Continuous statistical reference baseline, operating in pipeline intensity units."""
from __future__ import annotations

import numpy as np


def grayscale(image):
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("Statistical Reference Baseline requires non-empty grayscale images (H × W).")
    if not np.isfinite(values).all():
        raise ValueError("Statistical reference input contains non-finite pixel values.")
    return values


def reference_statistics(images):
    mean = m2 = None
    count = 0
    for image in images:
        values = grayscale(image)
        if mean is None:
            mean, m2 = np.zeros_like(values), np.zeros_like(values)
        if values.shape != mean.shape:
            raise ValueError("All statistical reference images must have the same size.")
        count += 1
        delta = values - mean
        mean += delta / count
        m2 += delta * (values - mean)
    if count == 0:
        raise ValueError("No images available for the statistical reference.")
    std = np.sqrt(np.maximum(m2 / count, 0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Statistical reference calculation exceeded the numeric range.")
    return mean, std, count


def anomaly_map(image, mean, std, epsilon):
    values = grayscale(image)
    if values.shape != mean.shape:
        raise ValueError(f"Reference shape {mean.shape} does not match image shape {values.shape}.")
    result = np.abs(values - mean) / (std + epsilon)
    if not np.isfinite(result).all():
        raise ValueError("Statistical anomaly map exceeded the numeric range; increase epsilon.")
    return result


def load_reference(path):
    with np.load(path, allow_pickle=False) as data:
        mean, std = grayscale(data["mean"]), grayscale(data["std"])
        epsilon, count = float(data["epsilon"]), int(data["count"])
        if mean.shape != std.shape or np.any(std < 0) or not np.isfinite(epsilon) or epsilon <= 0 or count < 1:
            raise ValueError("Invalid statistical reference artifact.")
        return mean, std, epsilon, count
