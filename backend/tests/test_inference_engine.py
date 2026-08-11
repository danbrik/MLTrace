import csv
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app.schemas import TestingRunCreate
from app.testing import engine as testing_engine
from app.testing import service as testing_service
from app.testing.service import enqueue_testing_run

from tests.test_testing_service import seed_finished_mean_image_run, write_tiff


def _make_engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


def _append_test_frame(db, test_set_id: int, tmp_path: Path, *, second: int, value: int) -> None:
    rule = db.scalar(
        select(models.TrainingDatasetRule).where(
            models.TrainingDatasetRule.training_dataset_id == test_set_id
        )
    )
    folder = db.get(models.DatasetFolder, rule.folder_id)
    timestamp = datetime(2026, 4, 1, 12, 0, 0) + timedelta(seconds=second)
    write_tiff(tmp_path / "test_images" / f"frame_{timestamp:%Y%m%d_%H%M%S}.tiff", value)
    rule.end_timestamp = timestamp
    folder.last_timestamp = timestamp
    folder.image_count += 1
    db.commit()


def test_gradient_model_moves_to_cuda_before_materializing_forward(tmp_path: Path, monkeypatch) -> None:
    events = []

    class FakeTensor:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            events.append(("tensor.to", device))
            return self

    class FakeModel:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            events.append(("model.to", device))
            return self

        def eval(self):
            return self

        def __call__(self, tensor):
            events.append(("forward", self.device, tensor.device))
            assert self.device == "cuda"
            assert tensor.device == "cuda"

        def load_state_dict(self, state):
            events.append(("load_state_dict", state))

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda value: value,
        from_numpy=lambda _array: FakeTensor(),
        load=lambda _path, map_location: {"weight": 1},
        no_grad=nullcontext,
    )
    model = FakeModel()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(testing_service, "_build_model", lambda *_args, **_kwargs: (model, False))

    artifact_path = tmp_path / "weights.pt"
    artifact_path.touch()
    evaluator = testing_service.ArtifactEvaluator.__new__(testing_service.ArtifactEvaluator)
    evaluator.training_run = SimpleNamespace(artifact_kind="weights")
    evaluator.configuration = SimpleNamespace(
        builder_kind="sequential_autoencoder",
        method_config={"input_channels": 1, "input_height": 6, "input_width": 8},
    )
    evaluator.artifact_path = artifact_path
    evaluator.model = None
    evaluator.fast_anogan_modules = None
    evaluator.torch = None

    evaluator._ensure_torch_model(np.zeros((6, 8), dtype=np.float32))

    assert events[:3] == [
        ("model.to", "cuda"),
        ("tensor.to", "cuda"),
        ("forward", "cuda", "cuda"),
    ]
    assert evaluator._batch_device() == "cuda"
    assert events.count(("model.to", "cuda")) == 1


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


def test_failed_testing_run_resumes_from_last_checkpoint(tmp_path: Path, monkeypatch) -> None:
    engine = _make_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        _append_test_frame(db, test_set_id, tmp_path, second=30, value=130)
        queued = enqueue_testing_run(
            db,
            TestingRunCreate(training_run_id=training_run_id, training_dataset_id=test_set_id),
            wake_scheduler=False,
        )
        monkeypatch.setattr(testing_engine, "SessionLocal", lambda: Session())
        monkeypatch.setattr(testing_engine, "CHECKPOINT_INTERVAL", 2)
        monkeypatch.setattr(testing_engine, "_INFER_BATCH", 1)
        monkeypatch.setattr(testing_service.scheduler, "wake", lambda: None)

        original_score_batch = testing_service.ArtifactEvaluator.score_batch
        calls = {"count": 0, "fail": True}

        def flaky_score_batch(self, images, roi):
            calls["count"] += 1
            if calls["fail"] and calls["count"] == 4:
                raise RuntimeError("synthetic inference failure")
            return original_score_batch(self, images, roi)

        monkeypatch.setattr(testing_service.ArtifactEvaluator, "score_batch", flaky_score_batch)
        testing_engine.run_testing(queued.id)

        run = db.get(models.TestingRun, queued.id)
        db.refresh(run)
        assert run.status == "failed"
        assert run.image_count == 3
        assert run.checkpoint_input_count == 2
        assert run.checkpoint_result_count == 2
        assert run.checkpoint_at is not None
        assert db.scalar(
            select(models.TestingRunResult)
            .where(models.TestingRunResult.testing_run_id == run.id)
            .order_by(models.TestingRunResult.position.desc())
        ).position == 2

        restarted = testing_service.restart_testing_run_from_checkpoint(db, run.id)
        assert restarted is not None
        assert restarted.status == "queued"
        assert restarted.restart_mode == "checkpoint"
        assert restarted.image_count == 2

        calls.update(count=0, fail=False)
        testing_engine.run_testing(run.id)
        db.refresh(run)
        assert run.status == "finished"
        assert calls["count"] == 2
        assert run.image_count == 4
        assert run.score_mean == pytest.approx((0.0 + 100.0 + 400.0 + 900.0) / 4)
        assert run.checkpoint_at is None
        assert run.checkpoint_state is None
        assert run.restart_mode is None

        rows = db.scalars(
            select(models.TestingRunResult)
            .where(models.TestingRunResult.testing_run_id == run.id)
            .order_by(models.TestingRunResult.position)
        ).all()
        assert [row.position for row in rows] == [0, 1, 2, 3]
        assert [row.score for row in rows] == [0.0, 100.0, 400.0, 900.0]
        with open(run.results_path, encoding="utf-8") as handle:
            assert len(list(csv.reader(handle))) == 5
    finally:
        db.close()


def test_checkpoint_resume_rejects_changed_source_and_full_restart_clears_it(
    tmp_path: Path, monkeypatch
) -> None:
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
        monkeypatch.setattr(testing_engine, "SessionLocal", lambda: Session())
        monkeypatch.setattr(testing_engine, "CHECKPOINT_INTERVAL", 2)
        monkeypatch.setattr(testing_engine, "_INFER_BATCH", 1)
        monkeypatch.setattr(testing_service.scheduler, "wake", lambda: None)

        original_score_batch = testing_service.ArtifactEvaluator.score_batch
        calls = {"count": 0}

        def fail_after_checkpoint(self, images, roi):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("stop after checkpoint")
            return original_score_batch(self, images, roi)

        monkeypatch.setattr(testing_service.ArtifactEvaluator, "score_batch", fail_after_checkpoint)
        testing_engine.run_testing(queued.id)
        run = db.get(models.TestingRun, queued.id)
        db.refresh(run)
        assert run.status == "failed"
        assert run.checkpoint_input_count == 2

        _append_test_frame(db, test_set_id, tmp_path, second=30, value=130)
        with pytest.raises(ValueError, match="changed since the checkpoint"):
            testing_service.restart_testing_run_from_checkpoint(db, run.id)
        db.refresh(run)
        assert run.status == "failed"
        assert run.checkpoint_input_count == 2

        restarted = testing_service.restart_testing_run(db, run.id)
        assert restarted is not None
        assert restarted.status == "queued"
        assert restarted.checkpoint_at is None
        assert restarted.checkpoint_input_count is None
        assert db.scalar(
            select(models.TestingRunResult).where(models.TestingRunResult.testing_run_id == run.id)
        ) is None
    finally:
        db.close()


def test_stae_testing_run_resumes_by_clip_without_duplicate_results(tmp_path: Path, monkeypatch) -> None:
    engine = _make_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        _append_test_frame(db, test_set_id, tmp_path, second=30, value=130)
        training_run = db.get(models.TrainingRun, training_run_id)
        method = training_run.training_pipeline.method_configuration
        method.builder_kind = "spatiotemporal_autoencoder"
        method.method_type = "spatiotemporal_autoencoder"
        method.method_config = {
            "input_channels": 1,
            "input_height": 6,
            "input_width": 8,
            "clip_length": 1,
            "future_length": 0,
            "prediction_branch": False,
        }
        db.commit()
        queued = enqueue_testing_run(
            db,
            TestingRunCreate(training_run_id=training_run_id, training_dataset_id=test_set_id),
            wake_scheduler=False,
        )
        monkeypatch.setattr(testing_engine, "SessionLocal", lambda: Session())
        monkeypatch.setattr(testing_engine, "CHECKPOINT_INTERVAL", 2)
        monkeypatch.setattr(testing_engine, "_INFER_BATCH", 1)
        monkeypatch.setattr(testing_service.scheduler, "wake", lambda: None)

        calls = {"count": 0, "fail": True}

        class FakeStaeEvaluator:
            def __init__(self, training_run, _config):
                self.artifact_path = Path(training_run.artifact_path)
                self.model = None

            def reconstruct_clip_batch(self, clips):
                calls["count"] += 1
                if calls["fail"] and calls["count"] == 4:
                    raise RuntimeError("synthetic STAE failure")
                self.model = object()
                return [{"reconstruction": clip.copy(), "prediction": None} for clip in clips]

        monkeypatch.setattr(testing_engine, "ArtifactEvaluator", FakeStaeEvaluator)
        testing_engine.run_testing(queued.id)
        run = db.get(models.TestingRun, queued.id)
        db.refresh(run)
        assert run.status == "failed"
        assert run.checkpoint_input_count == 2
        assert run.checkpoint_result_count == 2

        testing_service.restart_testing_run_from_checkpoint(db, run.id)
        calls.update(count=0, fail=False)
        testing_engine.run_testing(run.id)
        db.refresh(run)
        assert run.status == "finished"
        assert calls["count"] == 2
        rows = db.scalars(
            select(models.TestingRunResult)
            .where(models.TestingRunResult.testing_run_id == run.id)
            .order_by(models.TestingRunResult.position)
        ).all()
        assert [row.position for row in rows] == [0, 1, 2, 3]
        assert all(row.result_metadata["sample_kind"] == "clip" for row in rows)
    finally:
        db.close()


def test_aborted_testing_run_keeps_checkpoint_for_resume(tmp_path: Path, monkeypatch) -> None:
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
        monkeypatch.setattr(testing_engine, "SessionLocal", lambda: Session())
        monkeypatch.setattr(testing_engine, "CHECKPOINT_INTERVAL", 2)
        monkeypatch.setattr(testing_engine, "_INFER_BATCH", 1)
        monkeypatch.setattr(testing_service.scheduler, "wake", lambda: None)

        class AbortAfterCheckpoint:
            checks = 0

            def is_set(self):
                self.checks += 1
                return self.checks >= 3

        testing_engine.run_testing(queued.id, AbortAfterCheckpoint())
        run = db.get(models.TestingRun, queued.id)
        db.refresh(run)
        assert run.status == "aborted"
        assert run.checkpoint_input_count == 2
        assert run.checkpoint_result_count == 2

        restarted = testing_service.restart_testing_run_from_checkpoint(db, run.id)
        assert restarted is not None
        assert restarted.status == "queued"
        assert restarted.restart_mode == "checkpoint"
    finally:
        db.close()
