from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

from PIL import Image


TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class TimestampPattern:
    regex: str
    timestamp_format: str
    example: str
    matches: int


@dataclass(frozen=True)
class ScannedImage:
    file_path: Path
    relative_path: str
    folder_relative_path: str
    file_name: str
    extension: str
    width: int | None
    height: int | None
    timestamp_raw: str
    timestamp_parsed: datetime
    file_size_bytes: int | None
    modified_time: datetime | None


COMMON_TIMESTAMP_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?P<timestamp>\d{8}_\d{6})", "%Y%m%d_%H%M%S"),
    (r"(?P<timestamp>\d{8}-\d{6})", "%Y%m%d-%H%M%S"),
    (r"(?P<timestamp>\d{14})", "%Y%m%d%H%M%S"),
    (r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", "%Y-%m-%dT%H:%M:%S"),
    (r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", "%Y-%m-%d_%H-%M-%S"),
    (r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})", "%Y-%m-%d_%H:%M:%S"),
    (r"(?P<timestamp>\d{4}\.\d{2}\.\d{2}_\d{2}-\d{2}-\d{2})", "%Y.%m.%d_%H-%M-%S"),
    (r"(?P<timestamp>\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", "%y-%m-%d_%H-%M-%S"),
    (r"(?P<timestamp>\d{6}_\d{6})", "%d%m%y_%H%M%S"),
    (r"(?P<timestamp>\d{6}-\d{6})", "%d%m%y-%H%M%S"),
    (r"(?P<timestamp>\d{12})", "%d%m%y%H%M%S"),
)


def iter_tiff_files(root_path: str | Path) -> list[Path]:
    root = Path(root_path).expanduser()
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS
    )


def extract_timestamp(file_name: str, regex: str, timestamp_format: str) -> tuple[str, datetime]:
    match = re.search(regex, file_name)
    if not match:
        raise ValueError(f"Timestamp regex did not match {file_name}")

    raw_timestamp = match.groupdict().get("timestamp")
    if raw_timestamp is None:
        raw_timestamp = match.group(1) if match.groups() else match.group(0)

    return raw_timestamp, datetime.strptime(raw_timestamp, timestamp_format)


def detect_timestamp_pattern(root_path: str | Path) -> TimestampPattern | None:
    files = iter_tiff_files(root_path)
    if not files:
        return None

    file_name = files[0].name

    for regex, timestamp_format in COMMON_TIMESTAMP_PATTERNS:
        try:
            raw, _ = extract_timestamp(file_name, regex, timestamp_format)
        except ValueError:
            continue
        return TimestampPattern(regex=regex, timestamp_format=timestamp_format, example=raw, matches=1)

    return None


def read_tiff_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def scan_dataset_files(root_path: str | Path, timestamp_regex: str, timestamp_format: str) -> tuple[list[ScannedImage], dict]:
    root = Path(root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {root}")

    scanned: list[ScannedImage] = []
    skipped_unparseable = 0
    skipped_unreadable = 0

    for path in iter_tiff_files(root):
        try:
            timestamp_raw, timestamp_parsed = extract_timestamp(path.name, timestamp_regex, timestamp_format)
        except ValueError:
            skipped_unparseable += 1
            continue

        width, height = read_tiff_size(path)
        if width is None or height is None:
            skipped_unreadable += 1
            continue

        stat = path.stat()
        relative_path = path.relative_to(root).as_posix()
        parent = path.parent.relative_to(root).as_posix()
        folder_relative_path = "." if parent == "." else parent

        scanned.append(
            ScannedImage(
                file_path=path,
                relative_path=relative_path,
                folder_relative_path=folder_relative_path,
                file_name=path.name,
                extension=path.suffix.lower(),
                width=width,
                height=height,
                timestamp_raw=timestamp_raw,
                timestamp_parsed=timestamp_parsed,
                file_size_bytes=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
            )
        )

    summary = {
        "total_tiff_files": len(iter_tiff_files(root)),
        "indexed_images": len(scanned),
        "skipped_unparseable_timestamp": skipped_unparseable,
        "skipped_unreadable_tiff": skipped_unreadable,
    }
    return scanned, summary


def summarize_folders(images: list[ScannedImage]) -> dict[str, dict]:
    by_folder: dict[str, list[ScannedImage]] = defaultdict(list)
    for image in images:
        by_folder[image.folder_relative_path].append(image)

    summaries: dict[str, dict] = {}
    for folder, folder_images in by_folder.items():
        sorted_images = sorted(folder_images, key=lambda image: image.timestamp_parsed)
        deltas = [
            (right.timestamp_parsed - left.timestamp_parsed).total_seconds()
            for left, right in zip(sorted_images, sorted_images[1:])
        ]
        resolutions = Counter(f"{image.width}x{image.height}" for image in folder_images)
        extensions = Counter(image.extension for image in folder_images)

        summaries[folder] = {
            "image_count": len(folder_images),
            "first_timestamp": sorted_images[0].timestamp_parsed,
            "last_timestamp": sorted_images[-1].timestamp_parsed,
            "extension_summary": dict(extensions),
            "resolution_summary": dict(resolutions),
            "cadence_summary": {
                "min_seconds": min(deltas) if deltas else None,
                "median_seconds": median(deltas) if deltas else None,
                "max_seconds": max(deltas) if deltas else None,
            },
        }

    return summaries
