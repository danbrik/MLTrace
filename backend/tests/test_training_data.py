from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app import services
from app.training.data import (
    enumerate_training_dataset_clip_samples,
    enumerate_training_dataset_image_records,
    enumerate_training_pipeline_images,
)


def write_tiff(path: Path, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=127).save(path)


def make_memory_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


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


def test_ordered_index_sequence_clips_match_aebad_v_train_counts(tmp_path: Path) -> None:
    db = make_memory_session()
    try:
        train_set = models.TrainingDataset(name="AeBAD-V Train normal all videos", usage_label="train")
        db.add(train_set)
        db.flush()

        video_counts = [362, 173, 64, 108]
        starts = [
            datetime(2026, 5, 1, 8, 0, 0),
            datetime(2026, 5, 1, 9, 0, 0),
            datetime(2026, 5, 1, 10, 0, 0),
            datetime(2026, 5, 1, 11, 0, 0),
        ]
        for video_index, (count, start) in enumerate(zip(video_counts, starts), start=1):
            folder_path = tmp_path / f"video{video_index}"
            for index in range(count):
                timestamp = start + timedelta(seconds=index)
                write_tiff(folder_path / f"frame_{timestamp:%Y%m%d_%H%M%S}.tiff")

            dataset = models.Dataset(
                name=f"AeBAD-V Train video{video_index} normal",
                root_path=str(folder_path),
                status="ready",
                timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})",
                timestamp_format="%Y%m%d_%H%M%S",
            )
            db.add(dataset)
            db.flush()
            folder = models.DatasetFolder(
                dataset_id=dataset.id,
                relative_path=".",
                image_count=count,
                first_timestamp=start,
                last_timestamp=start + timedelta(seconds=count - 1),
                extension_summary={".tiff": count},
                resolution_summary={"12x8": count},
                image_metadata={"format": "TIFF", "mode": "L", "dtype": "uint8", "channels": 1},
                cadence_summary={"mean_seconds": 1},
            )
            db.add(folder)
            db.flush()
            db.add(
                models.TrainingDatasetRule(
                    training_dataset_id=train_set.id,
                    folder_id=folder.id,
                    start_timestamp=folder.first_timestamp,
                    end_timestamp=folder.last_timestamp,
                    stride=1,
                )
            )
        db.commit()
        db.refresh(train_set)

        summary = enumerate_training_dataset_clip_samples(
            train_set,
            clip_length=8,
            future_length=1,
            sequence_contiguity_mode="ordered_index",
        )

        assert summary.sequence_contiguity_mode == "ordered_index"
        assert summary.selected_frame_count == 707
        assert summary.possible_clip_count == 675
        assert len(summary.clips) == 675
        assert summary.skipped_missing == 0
    finally:
        db.close()


def test_ordered_index_keeps_timestamp_gap_clips_that_cadence_mode_skips(tmp_path: Path) -> None:
    db = make_memory_session()
    try:
        timestamps = [
            datetime(2026, 5, 1, 8, 0, 0),
            datetime(2026, 5, 1, 8, 0, 1),
            datetime(2026, 5, 1, 8, 0, 2),
            datetime(2026, 5, 1, 8, 0, 10),
            datetime(2026, 5, 1, 8, 0, 11),
            datetime(2026, 5, 1, 8, 0, 12),
        ]
        for timestamp in timestamps:
            write_tiff(tmp_path / f"frame_{timestamp:%Y%m%d_%H%M%S}.tiff")

        dataset = models.Dataset(
            name="Gappy video",
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
            image_count=len(timestamps),
            first_timestamp=timestamps[0],
            last_timestamp=timestamps[-1],
            extension_summary={".tiff": len(timestamps)},
            resolution_summary={"12x8": len(timestamps)},
            image_metadata={"format": "TIFF", "mode": "L", "dtype": "uint8", "channels": 1},
            cadence_summary={"mean_seconds": 1},
        )
        db.add(folder)
        db.flush()
        train_set = models.TrainingDataset(name="Gappy train", usage_label="train")
        db.add(train_set)
        db.flush()
        db.add(
            models.TrainingDatasetRule(
                training_dataset_id=train_set.id,
                folder_id=folder.id,
                start_timestamp=timestamps[0],
                end_timestamp=timestamps[-1],
                stride=1,
            )
        )
        db.commit()
        db.refresh(train_set)

        ordered_summary = enumerate_training_dataset_clip_samples(
            train_set,
            clip_length=3,
            future_length=1,
            sequence_contiguity_mode="ordered_index",
        )
        cadence_summary = enumerate_training_dataset_clip_samples(
            train_set,
            clip_length=3,
            future_length=1,
            sequence_contiguity_mode="timestamp_cadence",
        )

        assert ordered_summary.possible_clip_count == 3
        assert len(ordered_summary.clips) == 3
        assert ordered_summary.skipped_missing == 0
        assert cadence_summary.possible_clip_count == 3
        assert len(cadence_summary.clips) == 0
        assert cadence_summary.skipped_missing == 3
    finally:
        db.close()
