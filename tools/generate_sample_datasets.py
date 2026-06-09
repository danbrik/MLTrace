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
) -> None:
    for index in range(count):
        timestamp = start + index * step
        filename = filename_format.format(ts=timestamp)
        extension = ".tiff" if index % 5 == 0 else ".tif"
        write_tiff(root / f"{filename}{extension}", timestamp, size, seed_offset + index)


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    dataset_a = ROOT / "dataset_alpha_yyyymmdd_underscore"
    generate_range(
        dataset_a,
        datetime(2026, 2, 3, 16, 0, 0),
        28,
        timedelta(minutes=30),
        "camA_{ts:%Y%m%d_%H%M%S}",
        (128, 96),
        0,
    )

    dataset_b = ROOT / "dataset_beta_compact"
    generate_range(
        dataset_b,
        datetime(2026, 3, 10, 8, 0, 0),
        36,
        timedelta(minutes=10),
        "line01_{ts:%Y%m%d%H%M%S}",
        (160, 120),
        200,
    )

    dataset_c = ROOT / "dataset_gamma_two_digit_year"
    generate_range(
        dataset_c,
        datetime(2026, 1, 21, 16, 46, 25),
        24,
        timedelta(seconds=20),
        "W14_HF_{ts:%y-%m-%d_%H-%M-%S}",
        (96, 96),
        400,
    )

    print(f"Generated sample datasets in {ROOT}")


if __name__ == "__main__":
    main()
