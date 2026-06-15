from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1] / "sample_datasets"


def write_tiff(path: Path, timestamp: datetime, size: tuple[int, int], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size, color=34 + (seed * 7) % 80)
    draw = ImageDraw.Draw(image)
    width, height = size
    offset = (seed * 11) % max(1, width // 2)
    draw.rectangle(
        [8 + offset // 4, 8, min(width - 8, 58 + offset), min(height - 8, 46 + offset // 3)],
        outline=150 + (seed * 5) % 80,
        width=2,
    )
    draw.line(
        [(0, (seed * 13) % height), (width, (seed * 17) % height)],
        fill=180,
        width=1,
    )
    image.save(path, format="TIFF", description=f"MLTrace sample {timestamp.isoformat()}")


def generate_range(
    root: Path,
    start: datetime,
    count: int,
    step: timedelta,
    filename_format: str,
    size: tuple[int, int],
    seed_offset: int,
    extension: str = ".tiff",
) -> None:
    for index in range(count):
        timestamp = start + index * step
        filename = filename_format.format(ts=timestamp)
        write_tiff(root / f"{filename}{extension}", timestamp, size, seed_offset + index)


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    dataset_a = ROOT / "sample_line_a_96x96"
    generate_range(
        dataset_a,
        datetime(2026, 4, 1, 8, 0, 0),
        18,
        timedelta(minutes=1),
        "lineA_{ts:%Y%m%d_%H%M%S}",
        (96, 96),
        0,
    )

    dataset_b = ROOT / "sample_line_b_96x96"
    generate_range(
        dataset_b,
        datetime(2026, 4, 1, 9, 0, 0),
        18,
        timedelta(minutes=1),
        "lineB_{ts:%Y%m%d_%H%M%S}",
        (96, 96),
        200,
    )

    dataset_c = ROOT / "sample_line_c_128x96"
    generate_range(
        dataset_c,
        datetime(2026, 4, 1, 10, 0, 0),
        18,
        timedelta(minutes=1),
        "lineC_{ts:%Y%m%d_%H%M%S}",
        (128, 96),
        400,
    )

    print(f"Generated sample datasets in {ROOT}")


if __name__ == "__main__":
    main()
