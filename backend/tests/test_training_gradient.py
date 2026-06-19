import threading
from pathlib import Path

import numpy as np

from app import models
from app.schemas import PreprocessingGraph
from app.training.engine import _guard_large_cpu_gradient_training, _to_nchw, train_gradient

from tests.test_modeling import autoencoder_payload
from tests.test_testing_service import make_db, write_tiff

LOAD_ONLY_GRAPH = {"nodes": [{"id": "load", "type": "load_image", "config": {}}], "edges": []}


def test_training_hot_path_scales_uint16_to_unit_range() -> None:
    image = np.array([[0, 32768, 65535]], dtype=np.uint16)

    normalized = _to_nchw(image)

    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 1, 3)
    assert normalized[0, 0, 0] == 0.0
    assert 0.49 < normalized[0, 0, 1] < 0.51
    assert normalized[0, 0, 2] == 1.0


def _seed_ae(db, tmp_path: Path, image_count: int = 6):
    payload = autoencoder_payload()
    root = tmp_path / "images"
    paths = []
    for index in range(image_count):
        path = root / f"img_{index:03d}.tiff"
        # 160x120 grayscale to match the AE's declared input (W=160, H=120).
        write_tiff(path, 80 + index * 10, size=(160, 120))
        paths.append(str(path))

    preprocessing = models.PreprocessingPipeline(
        name="Load only",
        graph=LOAD_ONLY_GRAPH,
        input_width=160,
        input_height=120,
        output_width=160,
        output_height=120,
    )
    method = models.MethodConfiguration(
        name="CNN-AE",
        method_type=payload["method_type"],
        method_family="autoencoder",
        method_version="1",
        training_mode="gradient",
        requires_training=True,
        supports_training_pipeline=True,
        artifact_kind="weights",
        builder_kind=payload["method_graph"]["builder_kind"],
        method_graph=payload["method_graph"],
        method_config=payload["method_config"],
        training_config=payload["training_config"],
        inference_config=payload["inference_config"],
        diagram={},
    )
    db.add_all([preprocessing, method])
    db.flush()
    pipeline = models.TrainingPipeline(
        name="AE pipeline",
        preprocessing_pipeline_id=preprocessing.id,
        method_configuration_id=method.id,
        training_parameters={},
    )
    db.add(pipeline)
    db.flush()
    run = models.TrainingRun(
        training_pipeline_id=pipeline.id,
        status="running",
        training_pipeline_name="AE pipeline",
        method_type=payload["method_type"],
        method_family="autoencoder",
        training_mode="gradient",
        builder_kind=payload["method_graph"]["builder_kind"],
        preprocessing_pipeline_name="Load only",
        dataset_names=["AE Set"],
        dataset_names_text="AE Set",
        shuffle=False,
        training_parameters={},
    )
    db.add(run)
    db.commit()
    return run, method, paths


def test_train_gradient_streams_and_persists(tmp_path: Path) -> None:
    db = make_db()
    try:
        run, method, paths = _seed_ae(db, tmp_path, image_count=6)
        graph = PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH)
        artifact_path = tmp_path / "artifact.pt"

        sample_count = train_gradient(
            db,
            run,
            method,
            paths,
            graph,
            {"epochs": 2, "batch_size": 2, "learning_rate": 0.001, "loss": "mse", "num_workers": 0},
            artifact_path,
            threading.Event(),
        )

        assert sample_count == 6
        assert run.epochs_total == 2
        assert run.epochs_completed == 2
        assert run.train_loss is not None
        assert artifact_path.exists() and artifact_path.stat().st_size > 0

        metrics = db.scalars(
            models.TrainingRunMetric.__table__.select().where(
                models.TrainingRunMetric.training_run_id == run.id
            )
        ).all()
        assert len(metrics) == 2
    finally:
        db.close()


def test_train_gradient_aborts_promptly(tmp_path: Path) -> None:
    db = make_db()
    try:
        run, method, paths = _seed_ae(db, tmp_path, image_count=4)
        graph = PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH)
        event = threading.Event()
        event.set()  # already aborted

        from app.training.engine import AbortedError

        try:
            train_gradient(
                db, run, method, paths, graph,
                {"epochs": 5, "batch_size": 2, "num_workers": 0},
                tmp_path / "artifact.pt", event,
            )
            raise AssertionError("Expected AbortedError")
        except AbortedError:
            pass
    finally:
        db.close()


def test_large_gradient_training_refuses_cpu(tmp_path: Path) -> None:
    db = make_db()
    try:
        _, method, _ = _seed_ae(db, tmp_path, image_count=1)
        method.method_config = {**method.method_config, "input_width": 960, "input_height": 960, "input_channels": 1}

        try:
            _guard_large_cpu_gradient_training(method, sample_count=60_177, device_type="cpu")
            raise AssertionError("Expected large CPU gradient training to be rejected")
        except ValueError as exc:
            assert "no CUDA GPU is available" in str(exc)
            assert "60177 images" in str(exc)
    finally:
        db.close()
