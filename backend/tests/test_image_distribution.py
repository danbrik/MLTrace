from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
import pytest

from app import models
from app.analysis import image_distribution
from tests.test_testing_service import make_db


def _write_image(path: Path, pixels: list[int]) -> None:
    image = Image.new("L", (2, 2))
    image.putdata(pixels)
    image.save(path)


def test_image_distribution_aggregates_hourly_and_reuses_csv(tmp_path, monkeypatch) -> None:
    db = make_db()
    try:
        dataset = models.Dataset(name="camera", root_path=str(tmp_path), status="ready")
        pipeline = models.PreprocessingPipeline(
            name="raw",
            graph={
                "nodes": [{"id": "load", "type": "load_image", "config": {"mode": "unchanged", "dtype": "source"}}],
                "edges": [],
            },
        )
        db.add_all([dataset, pipeline])
        db.flush()
        folder = models.DatasetFolder(dataset_id=dataset.id, relative_path=".", image_count=3)
        db.add(folder)
        db.flush()
        start = datetime(2026, 1, 1, 10, 10)
        specs = [
            ("a.png", start, [0, 0, 0, 4]),
            ("b.png", start + timedelta(minutes=20), [2, 2, 2, 2]),
            ("c.png", start + timedelta(hours=3), [10, 10, 10, 10]),
        ]
        for name, timestamp, pixels in specs:
            path = tmp_path / name
            _write_image(path, pixels)
            db.add(models.DatasetImage(
                dataset_id=dataset.id,
                folder_id=folder.id,
                file_path=str(path),
                relative_path=name,
                file_name=name,
                extension=".png",
                timestamp_raw=name,
                timestamp_parsed=timestamp,
                file_size_bytes=path.stat().st_size,
            ))
        training = models.TrainingDataset(dataset_id=dataset.id, name="train period", usage_label="train")
        db.add(training)
        db.flush()
        db.add(models.TrainingDatasetRule(
            training_dataset_id=training.id,
            folder_id=folder.id,
            start_timestamp=start,
            end_timestamp=start + timedelta(minutes=30),
            stride=1,
        ))
        db.commit()
        monkeypatch.setattr(image_distribution, "cache_path", lambda key: tmp_path / "cache" / f"{key}.csv")

        first = image_distribution.calculate(db, dataset.id, pipeline.id)
        second = image_distribution.calculate(db, dataset.id, pipeline.id)

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert first.total_images == first.successful_images == 3
        assert [point.hour.hour for point in first.hourly] == [10, 13]
        assert first.hourly[0].mean_intensity.median == pytest.approx(1.5)
        assert first.hourly[0].q95_intensity.median == pytest.approx(2.7)
        assert first.periods[0].name == "train period"
        assert (tmp_path / "cache" / f"{first.cache_key}.csv").read_text().startswith("image_id,timestamp,relative_path")
    finally:
        db.close()
