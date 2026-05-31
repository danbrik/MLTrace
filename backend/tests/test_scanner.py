from pathlib import Path

from PIL import Image

from app.scanner import detect_timestamp_pattern, scan_dataset_files, summarize_folders


def write_tiff(path: Path, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=127).save(path)


def test_detect_timestamp_pattern_for_yyyymmdd_hhmmss(tmp_path: Path) -> None:
    write_tiff(tmp_path / "0226" / "frame_20260204_153000.tif")
    write_tiff(tmp_path / "0226" / "frame_20260204_153010.tiff")

    pattern = detect_timestamp_pattern(tmp_path)

    assert pattern is not None
    assert pattern.regex == r"(?P<timestamp>\d{8}_\d{6})"
    assert pattern.timestamp_format == "%Y%m%d_%H%M%S"
    assert pattern.matches == 2


def test_scan_dataset_files_and_folder_summary(tmp_path: Path) -> None:
    write_tiff(tmp_path / "0226" / "frame_20260204_153000.tif", (20, 10))
    write_tiff(tmp_path / "0226" / "frame_20260204_153010.tif", (20, 10))
    write_tiff(tmp_path / "bad_timestamp.tif", (20, 10))

    images, scan_summary = scan_dataset_files(
        tmp_path,
        r"(?P<timestamp>\d{8}_\d{6})",
        "%Y%m%d_%H%M%S",
    )
    folder_summary = summarize_folders(images)

    assert scan_summary["total_tiff_files"] == 3
    assert scan_summary["indexed_images"] == 2
    assert scan_summary["skipped_unparseable_timestamp"] == 1
    assert folder_summary["0226"]["image_count"] == 2
    assert folder_summary["0226"]["resolution_summary"] == {"20x10": 2}
    assert folder_summary["0226"]["cadence_summary"]["median_seconds"] == 10

