from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app.inspect.contrast import StreamingMovingAverage, moving_average_uint8
from app.inspect import engine as inspect_engine
from app.inspect import service as inspect_service
from app.schemas import InspectPreviewRequest, InspectRunCreate
from app.training import data as inspect_data


def write_tiff(path: Path, value: int = 127, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def make_db():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal


def seed_inspect_fixture(db, root: Path, image_count: int = 6):
    for index in range(image_count):
        timestamp = datetime(2026, 2, 4, 15, 30, 0) + timedelta(seconds=index * 10)
        write_tiff(root / f"frame_{timestamp:%Y%m%d_%H%M%S}.tif", value=40 + index)

    dataset = models.Dataset(
        name="A",
        root_path=str(root),
        status="ready",
        timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})",
        timestamp_format="%Y%m%d_%H%M%S",
    )
    db.add(dataset)
    db.flush()
    folder = models.DatasetFolder(
        dataset_id=dataset.id,
        relative_path=".",
        image_count=image_count,
        first_timestamp=datetime(2026, 2, 4, 15, 30, 0),
        last_timestamp=datetime(2026, 2, 4, 15, 30, 0) + timedelta(seconds=(image_count - 1) * 10),
        extension_summary={".tif": image_count},
        resolution_summary={"12x8": image_count},
        image_metadata={"format": "TIFF", "mode": "L", "dtype": "uint8", "channels": 1},
        cadence_summary={"mean_seconds": 10},
    )
    db.add(folder)
    db.flush()
    training_dataset = models.TrainingDataset(name="Trainset", usage_label="train")
    db.add(training_dataset)
    db.flush()
    db.add(
        models.TrainingDatasetRule(
            training_dataset_id=training_dataset.id,
            folder_id=folder.id,
            start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
            end_timestamp=datetime(2026, 2, 4, 15, 30, 50),
            stride=2,
            matching_images=6,
            selected_images=3,
        )
    )
    preprocessing = models.PreprocessingPipeline(
        name="Load only",
        graph={
            "nodes": [{"id": "load", "type": "load_image", "config": {}, "position": {"x": 0, "y": 0}}],
            "edges": [],
        },
        input_width=12,
        input_height=8,
        output_width=12,
        output_height=8,
    )
    db.add(preprocessing)
    db.commit()
    db.refresh(training_dataset)
    db.refresh(preprocessing)
    return training_dataset, preprocessing


def test_streaming_moving_average_matches_reference_list_implementation() -> None:
    frames = [
        np.full((3, 4), value, dtype=np.uint8)
        for value in (0, 10, 20, 40, 80, 160)
    ]
    smoother = StreamingMovingAverage(radius=2)
    streamed = []
    for frame in frames:
        output = smoother.push(frame)
        if output is not None:
            streamed.append(output)
    streamed.extend(smoother.flush())

    expected = [moving_average_uint8(frames, index, 2) for index in range(len(frames))]
    assert len(streamed) == len(expected)
    for actual, reference in zip(streamed, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_inspect_preview_uses_clipped_rules_and_extra_stride(tmp_path: Path) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images")
        preview = inspect_service.preview_inspect(
            db,
            InspectPreviewRequest(
                training_dataset_id=training_dataset.id,
                preprocessing_pipeline_id=preprocessing.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 50),
                stride=2,
            ),
        )
        # Saved rule stride selects timestamps 00,20,40; Inspect stride 2 keeps 00,40.
        assert preview.selected_images == 2
        assert preview.first_timestamp == datetime(2026, 2, 4, 15, 30, 0)
        assert preview.width == 12
        assert preview.height == 8
        assert preview.dtype == "uint8"
        assert preview.image_data_url.startswith("data:image/png;base64,")
        assert preview.preview_frame_count == 2
        assert len(preview.preview_frames) == 2
        assert preview.preview_frames[0]["image_data_url"].startswith("data:image/png;base64,")
    finally:
        db.close()


def test_inspect_preview_compiles_pipeline_once(tmp_path: Path, monkeypatch) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    calls = 0
    original_compile = inspect_service.compile_pipeline

    def counted_compile(graph):
        nonlocal calls
        calls += 1
        return original_compile(graph)

    monkeypatch.setattr(inspect_service, "compile_pipeline", counted_compile)
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images")
        inspect_service.preview_inspect(
            db,
            InspectPreviewRequest(
                training_dataset_id=training_dataset.id,
                preprocessing_pipeline_id=preprocessing.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 50),
                stride=1,
            ),
        )
        assert calls == 1
    finally:
        db.close()


def test_inspect_run_writes_png_frames_and_mp4(tmp_path: Path, monkeypatch) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images", image_count=4)
        run = inspect_service.create_inspect_run(
            db,
            InspectRunCreate(
                training_dataset_id=training_dataset.id,
                preprocessing_pipeline_id=preprocessing.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 30),
                stride=1,
                fps=4,
            ),
        )
        monkeypatch.setattr(inspect_engine, "SessionLocal", SessionLocal)
        monkeypatch.setattr(inspect_engine, "data_dir", lambda: tmp_path / "artifacts")

        inspect_engine.run_inspect(run.id)

        db.expire_all()
        stored = db.get(models.InspectRun, run.id)
        assert stored is not None
        assert stored.status == "finished"
        # Saved rule stride=2 over four images yields two frames.
        assert stored.frame_count == 2
        assert stored.done_count == 2
        assert stored.frames_dir is not None
        assert stored.video_path is not None
        assert (Path(stored.frames_dir) / "frame_00000.png").exists()
        assert (Path(stored.frames_dir) / "frame_00001.png").exists()
        assert Path(stored.video_path).exists()
        assert Path(stored.video_path).stat().st_size > 0
    finally:
        db.close()


def test_inspect_contrast_preview_reports_diff_range(tmp_path: Path) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images")
        preview = inspect_service.preview_inspect(
            db,
            InspectPreviewRequest(
                training_dataset_id=training_dataset.id,
                preprocessing_pipeline_id=preprocessing.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 50),
                stride=2,
                contrast_enabled=True,
                contrast_reference_frames=1,
                contrast_shift=10000,
                contrast_vmax=12000,
                contrast_ma_radius=0,
            ),
        )
        assert preview.contrast_enabled is True
        assert preview.contrast_reference_frames_used == 1
        assert preview.dtype == "uint8"
        assert preview.channels == 1
        assert preview.contrast_diff_min is not None
        assert preview.contrast_diff_max is not None
        assert preview.contrast_diff_max >= preview.contrast_diff_min
        assert preview.image_data_url.startswith("data:image/png;base64,")
    finally:
        db.close()


def test_create_inspect_run_enqueues_without_full_enumeration(tmp_path: Path, monkeypatch) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images", image_count=6)

        def boom(*_args, **_kwargs):
            raise AssertionError("create_inspect_run must not enumerate the full range")

        monkeypatch.setattr(inspect_data, "enumerate_training_dataset_image_records_for_range", boom)

        run = inspect_service.create_inspect_run(
            db,
            InspectRunCreate(
                training_dataset_id=training_dataset.id,
                preprocessing_pipeline_id=preprocessing.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 50),
                stride=1,
                fps=4,
            ),
        )
        assert run.status == "queued"
        assert run.frame_count and run.frame_count > 0  # cheap estimate, no enumeration
    finally:
        db.close()


def test_create_inspect_run_empty_range_raises(tmp_path: Path) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images", image_count=6)
        try:
            inspect_service.create_inspect_run(
                db,
                InspectRunCreate(
                    training_dataset_id=training_dataset.id,
                    preprocessing_pipeline_id=preprocessing.id,
                    start_timestamp=datetime(2000, 1, 1, 0, 0, 0),
                    end_timestamp=datetime(2000, 1, 1, 1, 0, 0),
                    stride=1,
                    fps=4,
                ),
            )
            raise AssertionError("Expected ValueError for an empty range")
        except ValueError as exc:
            assert "No images in selected range" in str(exc)
    finally:
        db.close()


def test_inspect_contrast_run_writes_video(tmp_path: Path, monkeypatch) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images", image_count=4)
        run = inspect_service.create_inspect_run(
            db,
            InspectRunCreate(
                training_dataset_id=training_dataset.id,
                preprocessing_pipeline_id=preprocessing.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 30),
                stride=1,
                fps=4,
                contrast_enabled=True,
                contrast_reference_frames=1,
                contrast_shift=10000,
                contrast_vmax=12000,
                contrast_ma_radius=1,
            ),
        )
        assert run.contrast_enabled is True
        assert run.contrast_ma_radius == 1

        monkeypatch.setattr(inspect_engine, "SessionLocal", SessionLocal)
        monkeypatch.setattr(inspect_engine, "data_dir", lambda: tmp_path / "artifacts")

        inspect_engine.run_inspect(run.id)

        db.expire_all()
        stored = db.get(models.InspectRun, run.id)
        assert stored is not None
        assert stored.status == "finished"
        assert stored.frame_count == 2
        assert stored.done_count == 2
        assert (Path(stored.frames_dir) / "frame_00000.png").exists()
        assert Path(stored.video_path).exists()
        assert Path(stored.video_path).stat().st_size > 0
    finally:
        db.close()


def test_preview_uses_head_enumeration_not_full_range(tmp_path: Path, monkeypatch) -> None:
    SessionLocal = make_db()
    db = SessionLocal()
    try:
        training_dataset, preprocessing = seed_inspect_fixture(db, tmp_path / "images", image_count=6)

        def boom(*_args, **_kwargs):
            raise AssertionError("preview_inspect must not enumerate the full range")

        monkeypatch.setattr(inspect_data, "enumerate_training_dataset_image_records_for_range", boom)

        preview = inspect_service.preview_inspect(
            db,
            InspectPreviewRequest(
                training_dataset_id=training_dataset.id,
                preprocessing_pipeline_id=preprocessing.id,
                start_timestamp=datetime(2026, 2, 4, 15, 30, 0),
                end_timestamp=datetime(2026, 2, 4, 15, 30, 50),
                stride=2,
                contrast_enabled=True,
                contrast_reference_frames=2,
                contrast_shift=10000,
                contrast_vmax=12000,
                contrast_ma_radius=0,
            ),
        )
        # Saved rule stride selects 00,20,40; extra stride 2 keeps 00,40 → 2 frames.
        assert preview.selected_images == 2
        assert preview.matching_images == 2
        assert preview.contrast_enabled is True
        assert preview.contrast_reference_frames_used == 2
        assert preview.preview_frame_count == 2
        assert preview.image_data_url.startswith("data:image/png;base64,")
        assert preview.contrast_diff_min is not None and preview.contrast_diff_max is not None
    finally:
        db.close()
