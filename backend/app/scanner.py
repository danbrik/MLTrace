from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
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


@dataclass(frozen=True)
class TiffMetadata:
    width: int
    height: int
    image_format: str | None
    mode: str
    dtype: str
    channels: int


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


def direct_tiff_files(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path).expanduser()
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS
    )


def iter_tiff_files(root_path: str | Path) -> list[Path]:
    return list(chain.from_iterable(files for _, files in iter_tiff_file_groups(root_path)))


def iter_tiff_file_groups(root_path: str | Path) -> list[tuple[str, list[Path]]]:
    root = Path(root_path).expanduser()
    groups: list[tuple[str, list[Path]]] = []

    root_files = direct_tiff_files(root)
    if root_files:
        groups.append((".", root_files))

    child_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for child in child_dirs:
        files = direct_tiff_files(child)
        if files:
            groups.append((child.relative_to(root).as_posix(), files))

    return groups


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


def infer_dtype(mode: str) -> str:
    if mode in {"L", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
        return "uint8"
    if mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
        return "uint16"
    if mode == "I":
        return "int32"
    if mode == "F":
        return "float32"
    return "unknown"


def read_tiff_metadata(path: Path) -> TiffMetadata | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            return TiffMetadata(
                width=width,
                height=height,
                image_format=image.format,
                mode=image.mode,
                dtype=infer_dtype(image.mode),
                channels=len(image.getbands()),
            )
    except Exception:
        return None


def read_tiff_size(path: Path) -> tuple[int | None, int | None]:
    metadata = read_tiff_metadata(path)
    if metadata is None:
        return None, None
    return metadata.width, metadata.height


def scan_dataset_files(
    root_path: str | Path, timestamp_regex: str, timestamp_format: str
) -> tuple[list[ScannedImage], dict[str, dict], dict]:
    root = Path(root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {root}")

    scanned: list[ScannedImage] = []
    folder_summaries: dict[str, dict] = {}
    skipped_unparseable = 0
    skipped_unreadable = 0
    total_tiff_files = 0

    for folder_relative_path, files in iter_tiff_file_groups(root):
        total_tiff_files += len(files)
        if not files:
            continue

        first_path = files[0]
        last_path = files[-1]
        try:
            first_timestamp_raw, first_timestamp = extract_timestamp(first_path.name, timestamp_regex, timestamp_format)
            last_timestamp_raw, last_timestamp = extract_timestamp(last_path.name, timestamp_regex, timestamp_format)
        except ValueError:
            skipped_unparseable += len(files)
            continue

        cadence_seconds: int | None = None
        if len(files) >= 3:
            try:
                _, second_timestamp = extract_timestamp(files[1].name, timestamp_regex, timestamp_format)
                _, third_timestamp = extract_timestamp(files[2].name, timestamp_regex, timestamp_format)
                cadence_seconds = round(abs((third_timestamp - second_timestamp).total_seconds()))
            except ValueError:
                skipped_unparseable += len(files)
                continue
        elif len(files) == 2:
            cadence_seconds = round(abs((last_timestamp - first_timestamp).total_seconds()))

        metadata = read_tiff_metadata(first_path)
        if metadata is None:
            skipped_unreadable += len(files)
            continue

        start_timestamp = min(first_timestamp, last_timestamp)
        end_timestamp = max(first_timestamp, last_timestamp)

        folder_summaries[folder_relative_path] = {
            "image_count": len(files),
            "first_timestamp": start_timestamp,
            "last_timestamp": end_timestamp,
            "extension_summary": {first_path.suffix.lower(): len(files)},
            "resolution_summary": {f"{metadata.width}x{metadata.height}": len(files)},
            "image_metadata": {
                "format": metadata.image_format,
                "mode": metadata.mode,
                "dtype": metadata.dtype,
                "channels": metadata.channels,
                "sample_file_path": str(first_path),
                "scan_strategy": "direct_folder_sample",
            },
            "cadence_summary": {
                "min_seconds": cadence_seconds,
                "median_seconds": cadence_seconds,
                "max_seconds": cadence_seconds,
            },
        }

        representative_paths = [first_path]
        for path in (files[1] if len(files) >= 2 else None, files[2] if len(files) >= 3 else None, last_path):
            if path is not None and path not in representative_paths:
                representative_paths.append(path)

        for path in representative_paths:
            timestamp_raw, timestamp_parsed = extract_timestamp(path.name, timestamp_regex, timestamp_format)
            stat = path.stat()
            scanned.append(
                ScannedImage(
                    file_path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    folder_relative_path=folder_relative_path,
                    file_name=path.name,
                    extension=path.suffix.lower(),
                    width=metadata.width,
                    height=metadata.height,
                    timestamp_raw=timestamp_raw,
                    timestamp_parsed=timestamp_parsed,
                    file_size_bytes=stat.st_size,
                    modified_time=datetime.fromtimestamp(stat.st_mtime),
                )
            )

    summary = {
        "total_tiff_files": total_tiff_files,
        "indexed_images": sum(summary["image_count"] for summary in folder_summaries.values()),
        "indexed_representative_images": len(scanned),
        "skipped_unparseable_timestamp": skipped_unparseable,
        "skipped_unreadable_tiff": skipped_unreadable,
        "scan_strategy": "direct_folder_sample",
    }
    return scanned, folder_summaries, summary


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
