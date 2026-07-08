import csv
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app.schemas import TestingRunCreate
from app.testing import engine as testing_engine
from app.testing.service import enqueue_testing_run

from tests.test_testing_service import seed_finished_mean_image_run


def _make_engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


def test_run_testing_batched_matches_expected_scores(tmp_path: Path, monkeypatch) -> None:
    engine = _make_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        queued = enqueue_testing_run(
            db,
            TestingRunCreate(training_run_id=training_run_id, training_dataset_id=test_set_id),
            wake_scheduler=False,
        )

        # The engine opens its own session; route it to the test DB.
        monkeypatch.setattr(testing_engine, "SessionLocal", lambda: Session())
        testing_engine.run_testing(queued.id)

        run = db.get(models.TestingRun, queued.id)
        db.refresh(run)
        assert run.status == "finished"
        assert run.image_count == 3
        # Same as the synchronous create_testing_run path: (v-100)^2 for v in {100,110,120}.
        assert run.score_min == 0.0
        assert run.score_max == 400.0
        assert abs(run.score_mean - (0.0 + 100.0 + 400.0) / 3) < 1e-6

        rows = db.scalars(
            select(models.TestingRunResult)
            .where(models.TestingRunResult.testing_run_id == run.id)
            .order_by(models.TestingRunResult.position)
        ).all()
        assert [r.score for r in rows] == [0.0, 100.0, 400.0]

        # CSV written incrementally with a header + one row per image.
        assert run.results_path and Path(run.results_path).exists()
        with open(run.results_path, encoding="utf-8") as handle:
            csv_rows = list(csv.reader(handle))
        assert csv_rows[0][0] == "position"
        assert len(csv_rows) == 1 + 3
    finally:
        db.close()


def test_run_testing_skips_corrupt_image(tmp_path: Path, monkeypatch) -> None:
    engine = _make_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        # Corrupt the middle image (value 110) after indexing, like a truncated
        # or damaged file discovered at decode time.
        corrupt_path = tmp_path / "test_images" / "frame_20260401_120010.tiff"
        assert corrupt_path.exists()
        corrupt_path.write_bytes(b"this is not a tiff")

        queued = enqueue_testing_run(
            db,
            TestingRunCreate(training_run_id=training_run_id, training_dataset_id=test_set_id),
            wake_scheduler=False,
        )
        monkeypatch.setattr(testing_engine, "SessionLocal", lambda: Session())
        testing_engine.run_testing(queued.id)

        run = db.get(models.TestingRun, queued.id)
        db.refresh(run)
        assert run.status == "finished"
        assert run.expected_image_count == 3
        assert run.image_count == 2
        assert run.skipped_image_count == 1
        assert run.skipped_images == [str(corrupt_path)]

        # Scores computed over the two good images only: (v-100)^2 for v in {100,120}.
        assert run.score_min == 0.0
        assert run.score_max == 400.0
        assert abs(run.score_mean - (0.0 + 400.0) / 2) < 1e-6

        rows = db.scalars(
            select(models.TestingRunResult)
            .where(models.TestingRunResult.testing_run_id == run.id)
            .order_by(models.TestingRunResult.position)
        ).all()
        assert [r.position for r in rows] == [0, 1]
        assert [r.score for r in rows] == [0.0, 400.0]

        with open(run.results_path, encoding="utf-8") as handle:
            csv_rows = list(csv.reader(handle))
        assert len(csv_rows) == 1 + 2
    finally:
        db.close()
