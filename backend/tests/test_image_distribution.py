from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
import pytest
from sqlalchemy.orm import sessionmaker

from app import models
from app import database
from app.analysis import image_distribution
from app.analysis import image_distribution_runtime
from app.schemas import ImageDistributionRunCreate
from tests.test_testing_service import make_db


def _write_image(path: Path, pixels: list[int]) -> None:
    image = Image.new("L", (2, 2))
    image.putdata(pixels)
    image.save(path)


def test_image_distribution_aggregates_hourly_and_reuses_csv(tmp_path, monkeypatch) -> None:
    db = make_db()
    try:
        image_root = tmp_path / "images"
        image_root.mkdir()
        dataset = models.Dataset(
            name="camera",
            root_path=str(image_root),
            status="ready",
            timestamp_regex=r"(\d{8}_\d{6})",
            timestamp_format="%Y%m%d_%H%M%S",
        )
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
            ("20260101_101000.tif", start, [0, 0, 0, 4]),
            ("20260101_103000.tif", start + timedelta(minutes=20), [2, 2, 2, 2]),
            ("20260101_131000.tif", start + timedelta(hours=3), [10, 10, 10, 10]),
        ]
        for name, timestamp, pixels in specs:
            path = image_root / name
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
            end_timestamp=start + timedelta(hours=3),
            stride=1,
        ))
        db.commit()
        monkeypatch.setattr(image_distribution, "cache_path", lambda key: tmp_path / "cache" / f"{key}.csv")
        monkeypatch.setattr(
            image_distribution_runtime,
            "folder_index_path",
            lambda folder_id: tmp_path / "folder_indexes" / f"{folder_id}.sqlite",
        )
        monkeypatch.setattr(
            image_distribution_runtime,
            "run_manifest_path",
            lambda run_id: tmp_path / "runs" / str(run_id) / "manifest.sqlite",
        )
        monkeypatch.setattr(
            image_distribution_runtime,
            "cache_csv_path",
            lambda key: tmp_path / "stream_cache" / f"{key}.csv",
        )
        monkeypatch.setattr(
            image_distribution_runtime,
            "cache_result_path",
            lambda key: tmp_path / "stream_cache" / f"{key}.json",
        )

        progress = []
        first = image_distribution.calculate(
            db,
            training.id,
            pipeline.id,
            progress=lambda step, processed, total, successful, failed: progress.append(
                (step, processed, total, successful, failed)
            ),
        )
        second = image_distribution.calculate(db, training.id, pipeline.id)

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert first.total_images == first.successful_images == 3
        assert [point.hour.hour for point in first.hourly] == [10, 13]
        assert first.hourly[0].mean_intensity.median == pytest.approx(1.5)
        assert first.hourly[0].q95_intensity.median == pytest.approx(2.7)
        assert first.periods[0].name == "train period"
        assert progress[0][0] == "resolving_images"
        assert any(step[0] == "processing_images" and step[1] == 3 for step in progress)
        assert progress[-1][0] == "aggregating_hourly"
        assert first.training_dataset_name == "train period"
        assert (tmp_path / "cache" / f"{first.cache_key}.csv").read_text().startswith("image_index,timestamp,relative_path")

        run = image_distribution.enqueue(
            db,
            ImageDistributionRunCreate(
                training_dataset_id=training.id,
                preprocessing_pipeline_id=pipeline.id,
            ),
            wake_scheduler=False,
        )
        assert run.status == "queued"
        assert run.current_step == "queued"
        assert run.training_dataset_name == "train period"

        worker_sessions = sessionmaker(autocommit=False, autoflush=False, bind=db.get_bind())
        monkeypatch.setattr(database, "SessionLocal", worker_sessions)
        image_distribution.run_scheduled(run.id)
        db.expire_all()
        finished = db.get(models.ImageDistributionRun, run.id)
        assert finished is not None
        assert finished.status == "finished"
        assert finished.current_step == "finished"
        assert finished.processed_images == finished.total_images == 3
        assert finished.cache_hit is False
        assert finished.result["hourly"][0]["mean_intensity"]["median"] == pytest.approx(1.5)

        cached_run = image_distribution.enqueue(
            db,
            ImageDistributionRunCreate(
                training_dataset_id=training.id,
                preprocessing_pipeline_id=pipeline.id,
            ),
            wake_scheduler=False,
        )
        image_distribution.run_scheduled(cached_run.id)
        db.expire_all()
        cached = db.get(models.ImageDistributionRun, cached_run.id)
        assert cached is not None
        assert cached.status == "finished"
        assert cached.cache_hit is True
    finally:
        db.close()
