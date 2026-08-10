from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app.heatmap import engine as heatmap_engine
from app.heatmap import service as heatmap_service
from app.schemas import (
    HeatmapRangeRunCreate,
    HeatmapVisualizationConfig,
    TestingRunCreate as TestingRunCreatePayload,
)
from app.testing.service import create_testing_run
from app.testing.service import CURRENT_HEATMAP_RENDER_VERSION

from tests.test_testing_service import (
    FakeStaeHeatmapEvaluator,
    seed_finished_mean_image_run,
    seed_finished_stae_heatmap_run,
)


def _make_engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


def test_heatmap_range_renders_frames_and_dedups(tmp_path: Path, monkeypatch) -> None:
    engine = _make_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        testing_run = create_testing_run(
            db,
            TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id),
        )
        assert testing_run.status == "finished"
        assert testing_run.image_count == 3

        # Queue a range job covering all three frames (no scheduler).
        queued = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=1,
                scale_mode="shared",
            ),
            wake_scheduler=False,
        )
        assert queued.status == "queued"
        assert queued.render_version == CURRENT_HEATMAP_RENDER_VERSION

        # The engine opens its own session + writes frames under data_dir().
        data_root = tmp_path / "data"
        monkeypatch.setattr(heatmap_engine, "data_dir", lambda: data_root)
        monkeypatch.setattr(heatmap_engine, "SessionLocal", lambda: Session())

        heatmap_engine.run_heatmap_range(queued.id)

        run = db.get(models.HeatmapRangeRun, queued.id)
        db.refresh(run)
        assert run.status == "finished"
        assert run.frame_count == 3
        assert run.done_count == 3  # counter reached N
        assert run.global_vmax is not None  # shared scale recorded a ceiling
        assert run.frame_max_errors is not None
        assert len(run.frame_max_errors) == 3
        assert run.fps == 8
        assert run.video_path is not None
        assert Path(run.video_path).exists()
        assert Path(run.video_path).stat().st_size > 0

        frames_dir = data_root / "heatmap_ranges" / str(run.id)
        pngs = sorted(frames_dir.glob("frame_*.png"))
        assert len(pngs) == 3
        assert all(p.stat().st_size > 0 for p in pngs)
        # Shared-mode temp cache is cleaned up.
        assert not (frames_dir / "_err").exists()

        # Same config → dedup returns the existing finished job (no new row).
        again = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=1,
                scale_mode="shared",
            ),
            wake_scheduler=False,
        )
        assert again.id == run.id
        assert db.scalar(select(models.HeatmapRangeRun).where(models.HeatmapRangeRun.id != run.id)) is None

        irrelevant_stae_options = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=1,
                scale_mode="shared",
                stae_view="prediction",
                prediction_horizon=9,
            ),
            wake_scheduler=False,
        )
        assert irrelevant_stae_options.id == run.id
        assert irrelevant_stae_options.stae_view == "reconstruction"
        assert irrelevant_stae_options.prediction_horizon == 1

        forced = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=1,
                scale_mode="shared",
                force_recompute=True,
            ),
            wake_scheduler=False,
        )
        assert forced.id != run.id
        assert forced.config_signature == run.config_signature

        different_fps = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=1,
                fps=12,
                scale_mode="shared",
            ),
            wake_scheduler=False,
        )
        assert different_fps.id != run.id

        different_config = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=1,
                scale_mode="shared",
                visualization_config=HeatmapVisualizationConfig(
                    error_mode="absolute",
                    fixed_ceiling_enabled=True,
                    fixed_ceiling=20.0,
                ),
            ),
            wake_scheduler=False,
        )
        assert different_config.id != run.id
        heatmap_engine.run_heatmap_range(different_config.id)
        fixed_run = db.get(models.HeatmapRangeRun, different_config.id)
        db.refresh(fixed_run)
        assert fixed_run.status == "finished"
        assert not (data_root / "heatmap_ranges" / str(fixed_run.id) / "_err").exists()
    finally:
        db.close()


def test_heatmap_range_per_frame_scale(tmp_path: Path, monkeypatch) -> None:
    engine = _make_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        testing_run = create_testing_run(
            db,
            TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id),
        )
        queued = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=2,  # frames 0 and 2 → 2 frames
                scale_mode="per_frame",
            ),
            wake_scheduler=False,
        )
        data_root = tmp_path / "data"
        monkeypatch.setattr(heatmap_engine, "data_dir", lambda: data_root)
        monkeypatch.setattr(heatmap_engine, "SessionLocal", lambda: Session())

        heatmap_engine.run_heatmap_range(queued.id)

        run = db.get(models.HeatmapRangeRun, queued.id)
        db.refresh(run)
        assert run.status == "finished"
        assert run.frame_count == 2
        assert run.frame_max_errors is not None
        assert len(run.frame_max_errors) == 2
        frames_dir = data_root / "heatmap_ranges" / str(run.id)
        assert len(sorted(frames_dir.glob("frame_*.png"))) == 2
    finally:
        db.close()


def test_stae_heatmap_range_uses_5d_clip_batches_and_prediction_horizon(
    tmp_path: Path, monkeypatch
) -> None:
    engine = _make_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        _, _, testing_run_id, _ = seed_finished_stae_heatmap_run(db, tmp_path)
        prediction = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run_id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                stae_view="prediction",
                prediction_horizon=2,
            ),
            wake_scheduler=False,
        )
        other_horizon = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run_id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                stae_view="prediction",
                prediction_horizon=1,
            ),
            wake_scheduler=False,
        )
        reconstruction = heatmap_service.enqueue_heatmap_range(
            db,
            HeatmapRangeRunCreate(
                testing_run_id=testing_run_id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 0),
            ),
            wake_scheduler=False,
        )

        assert prediction.config_signature != other_horizon.config_signature
        assert prediction.config_signature != reconstruction.config_signature
        assert prediction.stae_view == "prediction"
        assert prediction.prediction_horizon == 2

        FakeStaeHeatmapEvaluator.clip_shapes = []
        captured_timestamps: list[datetime] = []
        monkeypatch.setattr(heatmap_engine, "ArtifactEvaluator", FakeStaeHeatmapEvaluator)
        monkeypatch.setattr(heatmap_engine, "data_dir", lambda: tmp_path / "data")
        monkeypatch.setattr(heatmap_engine, "SessionLocal", lambda: Session())
        monkeypatch.setattr(
            heatmap_engine,
            "add_timestamp_watermark",
            lambda image, timestamp: captured_timestamps.append(timestamp) or image,
        )

        heatmap_engine.run_heatmap_range(prediction.id)

        row = db.get(models.HeatmapRangeRun, prediction.id)
        db.refresh(row)
        assert row.status == "finished"
        assert FakeStaeHeatmapEvaluator.clip_shapes == [(1, 1, 6, 8)]
        assert captured_timestamps == [datetime(2026, 4, 1, 12, 0, 20)]
    finally:
        db.close()
