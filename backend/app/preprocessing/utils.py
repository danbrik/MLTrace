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


def order_quad_points(points: list[dict] | np.ndarray) -> np.ndarray:
    """Return four points ordered as top-left, top-right, bottom-right, bottom-left."""
    array = np.asarray(
        [[float(point["x"]), float(point["y"])] for point in points]
        if not isinstance(points, np.ndarray)
        else points,
        dtype=np.float32,
    )
    if array.shape != (4, 2):
        raise ValueError("Perspective warp requires exactly four 2D source points.")

    if len({(float(x), float(y)) for x, y in array}) != 4:
        raise ValueError("Perspective warp source points must be four distinct corners.")
    center = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
    ordered = array[np.argsort(angles)]
    # Rotate the clockwise image-coordinate order so the visually top-left
    # corner starts the TL, TR, BR, BL sequence. The y tie-break handles diamonds.
    start = min(range(4), key=lambda index: (float(ordered[index].sum()), float(ordered[index, 1])))
    ordered = np.roll(ordered, -start, axis=0)
    x = ordered[:, 0]
    y = ordered[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 1e-6:
        raise ValueError("Perspective warp source points must form a non-degenerate quadrilateral.")
    return ordered


def rectified_quad_size(points: list[dict] | np.ndarray) -> tuple[int, int]:
    """Calculate an aspect-preserving rectified size from an arbitrary quadrilateral."""
    top_left, top_right, bottom_right, bottom_left = order_quad_points(points)
    width = max(
        int(round(float(np.linalg.norm(top_right - top_left)))),
        int(round(float(np.linalg.norm(bottom_right - bottom_left)))),
        1,
    )
    height = max(
        int(round(float(np.linalg.norm(bottom_left - top_left)))),
        int(round(float(np.linalg.norm(bottom_right - top_right)))),
        1,
    )
    return width, height


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    array = image.astype(np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - minimum) / (maximum - minimum) * 255).astype(np.uint8)
