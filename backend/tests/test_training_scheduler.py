from pathlib import Path
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app.training.scheduler import _worker_database_url, move_queued_job, normalize_queue_ranks


def test_worker_database_url_resolves_relative_sqlite_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = _worker_database_url("sqlite:///./.mltrace/mltrace.db")
    url = make_url(resolved)

    assert url.drivername == "sqlite"
    assert Path(url.database).is_absolute()
    assert Path(url.database) == tmp_path / ".mltrace" / "mltrace.db"


def test_worker_database_url_leaves_non_sqlite_urls_unchanged() -> None:
    url = "postgresql+psycopg://user:password@localhost:5432/mltrace"

    assert _worker_database_url(url) == url


def make_db():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def test_move_queued_job_swaps_global_queue_order_across_kinds() -> None:
    db = make_db()
    try:
        train = models.TrainingRun(
            training_pipeline_id=1,
            status="queued",
            enqueued_at=datetime(2026, 1, 1, 12, 0, 0),
            queue_rank=1,
            training_pipeline_name="Train",
            method_type="mean_image",
            method_family="statistical_baseline",
            training_mode="fit",
            builder_kind="form",
            preprocessing_pipeline_name="Prep",
            dataset_names=[],
            dataset_names_text="",
            shuffle=False,
            training_parameters={},
        )
        test = models.TestingRun(
            name="Inference",
            training_run_id=1,
            training_dataset_id=1,
            status="queued",
            enqueued_at=datetime(2026, 1, 1, 12, 1, 0),
            queue_rank=2,
            training_run_name="Train",
            training_pipeline_name="Train",
            training_dataset_name="Dataset",
            preprocessing_pipeline_name="Prep",
            method_type="mean_image",
            method_family="statistical_baseline",
            training_mode="fit",
            artifact_kind="mean_image",
            artifact_path="/tmp/artifact.npy",
        )
        heatmap = models.HeatmapRangeRun(
            testing_run_id=1,
            status="queued",
            enqueued_at=datetime(2026, 1, 1, 12, 2, 0),
            queue_rank=3,
            start_timestamp=datetime(2026, 1, 1, 12, 0, 0),
            end_timestamp=datetime(2026, 1, 1, 12, 3, 0),
            stride=1,
            scale_mode="per_frame",
            visualization_config={},
            render_version=1,
            done_count=0,
            config_signature="abc",
            testing_run_name="Inference",
        )
        db.add_all([train, test, heatmap])
        db.commit()

        moved = move_queued_job(db, "heatmap", heatmap.id, "up")
        assert moved.id == heatmap.id
        normalize_queue_ranks(db)

        assert db.get(models.TrainingRun, train.id).queue_rank == 1
        assert db.get(models.HeatmapRangeRun, heatmap.id).queue_rank == 2
        assert db.get(models.TestingRun, test.id).queue_rank == 3

        db.get(models.TestingRun, test.id).status = "running"
        db.commit()
        with pytest.raises(ValueError, match="Only queued"):
            move_queued_job(db, "test", test.id, "up")
    finally:
        db.close()
