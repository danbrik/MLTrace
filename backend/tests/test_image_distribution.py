from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
import pytest
from sqlalchemy.orm import sessionmaker

from app import models
from app import database
from app.analysis import image_distribution
from app.analysis import image_distribution_runtime
from app.schemas import (
    ImageDistributionIntervalInput,
    ImageDistributionIntervalRequest,
    ImageDistributionRunCreate,
)
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

        interval_result = image_distribution.calculate_interval_summaries(
            db,
            run.id,
            ImageDistributionIntervalRequest(intervals=[
                ImageDistributionIntervalInput(
                    id="morning",
                    name="Morning block",
                    start=start,
                    end=start + timedelta(minutes=50),
                ),
                ImageDistributionIntervalInput(
                    id="empty",
                    name="Empty block",
                    start=start + timedelta(hours=1),
                    end=start + timedelta(hours=2),
                ),
            ]),
        )
        morning, empty = interval_result.intervals
        assert morning.image_count == 2
        assert morning.mean_intensity.median == pytest.approx(1.5)
        assert morning.mean_intensity.q25 == pytest.approx(1.25)
        assert morning.mean_intensity.q75 == pytest.approx(1.75)
        assert morning.mean_intensity.iqr == pytest.approx(0.5)
        assert morning.q95_intensity.median == pytest.approx(2.7)
        assert empty.image_count == 0
        assert empty.mean_intensity is None

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


def test_image_distribution_interval_statistics_fall_back_to_csv(tmp_path, monkeypatch) -> None:
    db = make_db()
    try:
        cache_key = "a" * 24
        csv_path = tmp_path / f"{cache_key}.csv"
        csv_path.write_text(
            "image_index,timestamp,relative_path,mean_intensity,spatial_std_intensity,q95_intensity,error\n"
            "0,2026-01-01T10:00:00,a.tif,1,2,3,\n"
            "1,2026-01-01T10:30:00,b.tif,3,4,7,\n"
            "2,2026-01-01T11:00:00,broken.tif,,,,broken\n",
            encoding="utf-8",
        )
        run = models.ImageDistributionRun(
            training_dataset_id=1,
            preprocessing_pipeline_id=1,
            training_dataset_name="Train",
            usage_label="train",
            preprocessing_pipeline_name="Raw",
            status="finished",
            current_step="finished",
            cache_key=cache_key,
        )
        db.add(run)
        db.commit()
        monkeypatch.setattr(image_distribution, "cache_path", lambda _key: csv_path)

        response = image_distribution.calculate_interval_summaries(
            db,
            run.id,
            ImageDistributionIntervalRequest(intervals=[ImageDistributionIntervalInput(
                id="block",
                name="Block",
                start=datetime(2026, 1, 1, 10, 0),
                end=datetime(2026, 1, 1, 11, 0),
            )]),
        )

        summary = response.intervals[0]
        assert summary.image_count == 2
        assert summary.mean_intensity.median == pytest.approx(2.0)
        assert summary.spatial_std_intensity.iqr == pytest.approx(1.0)
        assert summary.q95_intensity.q75 == pytest.approx(6.0)
    finally:
        db.close()
