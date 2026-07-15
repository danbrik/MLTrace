"""Shared MP4 and timestamp-overlay helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def timestamp_label(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def add_timestamp_watermark(image_rgb: np.ndarray, value: datetime) -> np.ndarray:
    """Return an RGB frame with a readable timestamp baked into its top-right."""
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(11, min(image.width, image.height) // 32))
    label = timestamp_label(value)
    box = draw.textbbox((0, 0), label, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    margin = max(6, min(image.width, image.height) // 80)
    padding = max(4, margin // 2)
    left = max(0, image.width - margin - text_width - 2 * padding)
    top = margin
    draw.rounded_rectangle(
        (left, top, image.width - margin, top + text_height + 2 * padding),
        radius=max(2, padding),
        fill=(0, 0, 0, 155),
    )
    draw.text((left + padding, top + padding - box[1]), label, font=font, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot create an MP4 without frames.")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(max(1, fps)), (width, height)
    )
    if not writer.isOpened():
        raise ValueError("Could not open MP4 video writer.")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All MP4 frames must have the same dimensions.")
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
