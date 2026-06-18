from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app import services
from app.training.data import enumerate_training_dataset_image_records, enumerate_training_pipeline_images


def write_tiff(path: Path, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=127).save(path)


def test_runtime_resolver_enumerates_real_files_not_representative_rows(tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for index in range(6):
            timestamp = datetime(2026, 2, 4, 15, 30, 0) + timedelta(seconds=index * 10)
            write_tiff(tmp_path / f"frame_{timestamp:%Y%m%d_%H%M%S}.tif")

        dataset = models.Dataset(
            name="A",
            root_path=str(tmp_path),
            status="ready",
            timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})",
            timestamp_format="%Y%m%d_%H%M%S",
        )
        db.add(dataset)
        db.flush()
        folder = models.DatasetFolder(
            dataset_id=dataset.id,
            relative_path=".",
            image_count=6,
            first_timestamp=datetime(2026, 2, 4, 15, 30, 0),
            last_timestamp=datetime(2026, 2, 4, 15, 30, 50),
            extension_summary={".tif": 6},
            resolution_summary={"12x8": 6},
            image_metadata={"format": "TIFF", "mode": "L", "dtype": "uint8", "channels": 1},
            cadence_summary={"median_seconds": 10},
        )
        db.add(folder)
        db.flush()
        # Only one representative row exists, mirroring the fast scanner; the
        # runtime resolver still returns all matching files from disk.
        db.add(
            models.DatasetImage(
                dataset_id=dataset.id,
                folder_id=folder.id,
                file_path=str(tmp_path / "frame_20260204_153000.tif"),
                relative_path="frame_20260204_153000.tif",
                file_name="frame_20260204_153000.tif",
                extension=".tif",
                width=12,
                height=8,
                timestamp_raw="20260204_153000",
                timestamp_parsed=datetime(2026, 2, 4, 15, 30, 0),
            )
        )
        train_set = models.TrainingDataset(name="Train", usage_label="train")
        db.add(train_set)
        db.flush()
        db.add(
            models.TrainingDatasetRule(
                training_dataset_id=train_set.id,
                folder_id=folder.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 50),
                stride=2,
            )
        )
        pipeline = models.TrainingPipeline(
            name="P",
            preprocessing_pipeline_id=1,
            method_configuration_id=1,
            training_parameters={},
        )
        db.add(pipeline)
        db.flush()
        db.add(
            models.TrainingPipelineDataset(
                training_pipeline_id=pipeline.id,
                training_dataset_id=train_set.id,
                position=0,
            )
        )
        db.commit()
        db.refresh(train_set)
        db.refresh(pipeline)

        records = enumerate_training_dataset_image_records(train_set)
        assert [Path(record.file_path).name for record in records] == [
            "frame_20260204_153000.tif",
            "frame_20260204_153020.tif",
            "frame_20260204_153040.tif",
        ]
        assert enumerate_training_pipeline_images(db, pipeline) == [record.file_path for record in records]
    finally:
        db.close()


def test_missing_training_dataset_counts_can_be_refreshed(tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for index in range(5):
            timestamp = datetime(2026, 2, 4, 15, 30, 0) + timedelta(seconds=index * 10)
            write_tiff(tmp_path / f"frame_{timestamp:%Y%m%d_%H%M%S}.tif")

        dataset = models.Dataset(
            name="A",
            root_path=str(tmp_path),
            status="ready",
            timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})",
            timestamp_format="%Y%m%d_%H%M%S",
        )
        db.add(dataset)
        db.flush()
        folder = models.DatasetFolder(
            dataset_id=dataset.id,
            relative_path=".",
            image_count=5,
            first_timestamp=datetime(2026, 2, 4, 15, 30, 0),
            last_timestamp=datetime(2026, 2, 4, 15, 30, 40),
            extension_summary={".tif": 5},
            resolution_summary={"12x8": 5},
            image_metadata={"format": "TIFF", "mode": "L", "dtype": "uint8", "channels": 1},
            cadence_summary={"mean_seconds": 10},
        )
        db.add(folder)
        db.flush()
        train_set = models.TrainingDataset(name="Legacy", usage_label="train")
        db.add(train_set)
        db.flush()
        db.add(
            models.TrainingDatasetRule(
                training_dataset_id=train_set.id,
                folder_id=folder.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 40),
                stride=2,
            )
        )
        db.commit()
        db.refresh(train_set)

        serialized = services.serialize_training_dataset(db, train_set)
        assert serialized.counts_missing is True
        assert serialized.rules[0].matching_images is None
        assert serialized.integrity_warnings

        refreshed = services.refresh_training_dataset_counts(db, train_set.id)
        assert refreshed is not None
        assert refreshed.counts_missing is False
        assert refreshed.total_matching_images == 5
        assert refreshed.total_selected_images == 3
        assert refreshed.rules[0].matching_images == 5
        assert refreshed.rules[0].selected_images == 3
    finally:
        db.close()
