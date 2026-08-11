import csv
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app.schemas import TestingRunCreate
from app.testing import engine as testing_engine
from app.testing import service as testing_service
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
