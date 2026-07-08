import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app import models
from app.schemas import PreprocessingGraph
from app.training.engine import (
    _PreprocessedClipDataset,
    _SkippedSample,
    _collate_clips_skip_bad,
    _collate_images_skip_bad,
    _guard_large_cpu_gradient_training,
    _to_nchw,
    fit_mean_image,
    train_gradient,
)

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


def test_train_gradient_skips_corrupt_image(tmp_path: Path) -> None:
    db = make_db()
    try:
        run, method, paths = _seed_ae(db, tmp_path, image_count=6)
        corrupt_path = Path(paths[2])
        corrupt_path.write_bytes(b"this is not a tiff")
        graph = PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH)
        artifact_path = tmp_path / "artifact.pt"

        sample_count = train_gradient(
            db,
            run,
            method,
            paths,
            graph,
            {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "loss": "mse", "num_workers": 0},
            artifact_path,
            threading.Event(),
        )

        assert sample_count == 6
        assert run.train_loss is not None
        assert artifact_path.exists()
        assert run.skipped_image_count == 1
        assert run.skipped_images == [str(corrupt_path)]
    finally:
        db.close()


def test_collate_images_skip_bad_partitions_batches() -> None:
    import torch

    good = [torch.zeros((1, 2, 2)), torch.ones((1, 2, 2))]

    batch, skipped = _collate_images_skip_bad(good)
    assert batch.shape == (2, 1, 2, 2)
    assert skipped == []

    batch, skipped = _collate_images_skip_bad([good[0], _SkippedSample("/bad.tiff"), good[1]])
    assert batch.shape == (2, 1, 2, 2)
    assert skipped == ["/bad.tiff"]

    batch, skipped = _collate_images_skip_bad([_SkippedSample("/a.tiff"), _SkippedSample("/b.tiff")])
    assert batch is None
    assert skipped == ["/a.tiff", "/b.tiff"]


def test_collate_clips_skip_bad_drops_bad_clips() -> None:
    import torch

    clip = (torch.zeros((1, 3, 2, 2)), torch.zeros((1, 0, 2, 2)))

    x_batch, y_batch, skipped = _collate_clips_skip_bad([clip, _SkippedSample("/bad.tiff")])
    assert x_batch.shape == (1, 1, 3, 2, 2)
    assert y_batch.shape == (1, 1, 0, 2, 2)
    assert skipped == ["/bad.tiff"]

    x_batch, y_batch, skipped = _collate_clips_skip_bad([_SkippedSample("/bad.tiff")])
    assert x_batch is None and y_batch is None
    assert skipped == ["/bad.tiff"]


def test_clip_dataset_returns_sentinel_for_corrupt_frame(tmp_path: Path) -> None:
    good_path = tmp_path / "good.tiff"
    write_tiff(good_path, 100, size=(8, 6))
    bad_path = tmp_path / "bad.tiff"
    bad_path.write_bytes(b"this is not a tiff")

    graph = PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH)
    clips = [
        SimpleNamespace(
            input_frames=[SimpleNamespace(file_path=str(good_path)), SimpleNamespace(file_path=str(bad_path))],
            future_frames=[],
        )
    ]
    dataset = _PreprocessedClipDataset(clips, graph)

    sample = dataset[0]
    assert isinstance(sample, _SkippedSample)
    assert sample.path == str(bad_path)


def test_fit_mean_image_skips_corrupt_image(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate([100, 110, 120]):
        path = tmp_path / f"img_{index}.tiff"
        write_tiff(path, value, size=(8, 6))
        paths.append(str(path))
    Path(paths[1]).write_bytes(b"this is not a tiff")

    graph = PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH)
    artifact_path = tmp_path / "mean.npy"
    skipped: list[str] = []

    count = fit_mean_image(paths, graph, {}, artifact_path, threading.Event(), skipped_paths=skipped)

    assert count == 2
    assert skipped == [paths[1]]
    mean = np.load(artifact_path)
    assert np.allclose(mean, (100 + 120) / 2)


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
