"""Shared MP4 and timestamp-overlay helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_TRANSCODE_LOCK = threading.Lock()


def _ready_marker(path: Path) -> Path:
    return path.with_name(f"{path.name}.browser-ready")


def _ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def finalize_browser_mp4(path: Path) -> None:
    """Transcode an OpenCV MP4 to browser-safe H.264/YUV420p with faststart."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise ValueError("ffmpeg is required to create browser-compatible H.264 MP4 videos.")
    marker = _ready_marker(path)
    with _TRANSCODE_LOCK:
        if marker.exists() and marker.stat().st_mtime >= path.stat().st_mtime:
            return
        output = path.with_name(f"{path.stem}.browser-tmp.mp4")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            output.unlink(missing_ok=True)
            raise ValueError(f"Could not encode browser-compatible MP4: {result.stderr.strip()}")
        output.replace(path)
        marker.write_text("h264/yuv420p/faststart\n", encoding="utf-8")


def ensure_browser_mp4(path: Path) -> None:
    """One-time compatibility upgrade for MP4s created before H.264 support."""
    marker = _ready_marker(path)
    if marker.exists() and marker.stat().st_mtime >= path.stat().st_mtime:
        return
    finalize_browser_mp4(path)


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
    finalize_browser_mp4(path)
