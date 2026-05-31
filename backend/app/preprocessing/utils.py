from __future__ import annotations

import cv2
import numpy as np


def interpolation_flag(value: str) -> int:
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
    }.get(value, cv2.INTER_LINEAR)


def default_quad(width: int, height: int, margin_ratio: float = 0.2) -> list[dict[str, int]]:
    margin_x = max(0, round(width * margin_ratio))
    margin_y = max(0, round(height * margin_ratio))
    return [
        {"x": margin_x, "y": margin_y},
        {"x": max(margin_x + 1, width - 1 - margin_x), "y": margin_y},
        {"x": max(margin_x + 1, width - 1 - margin_x), "y": max(margin_y + 1, height - 1 - margin_y)},
        {"x": margin_x, "y": max(margin_y + 1, height - 1 - margin_y)},
    ]


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    array = image.astype(np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - minimum) / (maximum - minimum) * 255).astype(np.uint8)

