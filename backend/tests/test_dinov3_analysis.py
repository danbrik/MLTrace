from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from app import models
from app.analysis import dinov3_engine as engine, dinov3_service as service
from app.analysis.dinov3_schemas import RepresentationConfig
from app.training.data import ResolvedDatasetImage
from tests.test_training_scheduler import make_db

START = datetime(2026, 1, 1)


def interval(id, label, start, end, rate=1, random=False):
    return {"id": id, "name": id, "label": label, "start": (START + timedelta(seconds=start)).isoformat(),
            "end": (START + timedelta(seconds=end)).isoformat(), "sampling_rate": rate, "random": random}


def config(**changes):
    return RepresentationConfig.model_validate({"training_dataset_id": 1, "preprocessing_pipeline_id": 1,
        "intervals": [interval("n", "normal", 0, 4), interval("a", "anomaly", 4, 8), interval("b", "buffer", 8, 12)],
        **changes})


def records(count):
    return [ResolvedDatasetImage(str(i), START + timedelta(seconds=i), "images", "/images", 1, ".", str(i)) for i in range(count)]


def test_sampling_blocks_and_exclusive_end():
    cfg = config(intervals=[interval("n", "normal", 0, 65, 30), interval("a", "anomaly", 65, 125, 30)])
    rows, preview = engine.sample_records(records(126) + records(126), cfg)
    assert [row["file_path"] for row in rows] == ["29", "59", "94", "124"]
    assert preview.total == 4
    assert preview.intervals[0].remainder == 5
    assert preview.label_counts == {"normal": 2, "anomaly": 2, "buffer": 0}
    assert not preview.errors


def test_random_sampling_stable_on_reordering_and_label_changes():
    cfg = config(intervals=[interval("n", "normal", 0, 65, 30, True), interval("a", "anomaly", 65, 125, 30, True)])
    first, _ = engine.sample_records(records(125), cfg)
    swapped = config(intervals=list(reversed(cfg.model_dump()["intervals"])))
    second, _ = engine.sample_records(list(reversed(records(125))), swapped)
    assert first == second
    for sample, low in zip(first, [0, 30, 65, 95]):
        assert low <= int(sample["file_path"]) < low + 30
    cfg.intervals[0].label = "buffer"
    third, _ = engine.sample_records(records(125), cfg)
    assert [row["file_path"] for row in first] == [row["file_path"] for row in third]


def test_interval_validation_and_stable_events():
    with pytest.raises(ValidationError, match="überlappen"):
        config(intervals=[interval("n", "normal", 0, 10), interval("a", "anomaly", 9, 20)])
    with pytest.raises(ValidationError, match="eindeutig"):
        config(intervals=[interval("n", "normal", 0, 4), interval("n", "anomaly", 4, 8)])
    cfg = config(intervals=[interval("a", "anomaly", 4, 8), interval("n", "normal", 0, 4), interval("c", "anomaly", 8, 12)])
    assert [item.event_id for item in cfg.intervals] == ["A1", None, "A2"]
    restored = config(intervals=list(reversed(cfg.model_dump()["intervals"])))
    assert {item.id: item.event_id for item in restored.intervals} == {item.id: item.event_id for item in cfg.intervals}
    _, preview = engine.sample_records(records(2), cfg)
    assert preview.errors


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_dtype_intensity_preserves_absolute_values(dtype):
    info = np.iinfo(dtype)
    image = np.array([[info.min, info.max]], dtype=dtype)
    rgb = engine.unit_rgb(image)
    np.testing.assert_allclose(rgb[0, 0], 0)
    np.testing.assert_allclose(rgb[0, 1], 1)
    assert engine.model_rgb(image).shape == (224, 224, 3)
    # A constant image is not stretched to zero.
    np.testing.assert_allclose(engine.unit_rgb(np.full((2, 2), info.max, dtype=dtype)), 1)


@pytest.mark.parametrize("image", [np.array([[2.0]]), np.array([[np.nan]]), np.zeros((2, 2, 4)), np.array([1, 2])])
def test_invalid_pipeline_output_rejected(image):
    with pytest.raises(ValueError):
        engine.model_rgb(image)


@pytest.mark.parametrize("cached", [False, True])
def test_frozen_encoder_cls_and_normalization(monkeypatch, tmp_path, cached):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    hub = pytest.importorskip("huggingface_hub")
    safetensors = pytest.importorskip("safetensors.torch")
    from huggingface_hub.errors import LocalEntryNotFoundError
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.rope = SimpleNamespace(periods=torch.tensor([1.23456]))
        def forward_features(self, pixel_values):
            assert not self.training and not torch.is_grad_enabled()
            assert not self.weight.requires_grad
            np.testing.assert_allclose(pixel_values.numpy(), 1, atol=1e-6)
            return torch.arange(12).reshape(1, 3, 4).float()
    model = Model()
    (tmp_path / "config.json").write_text(json.dumps({"architecture": engine.MODEL_ARCHITECTURE,
        "pretrained_cfg": {"mean": [.5]*3, "std": [.5]*3}}))
    calls = []
    def download(**kwargs):
        assert kwargs["token"] is False
        assert kwargs["repo_id"] == engine.MODEL_ID
        assert kwargs["revision"] == engine.MODEL_REVISION
        calls.append(kwargs)
        if not cached and kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return str(tmp_path / kwargs["filename"])
    def create(name, **kwargs):
        assert name == engine.MODEL_ARCHITECTURE
        assert kwargs == dict(pretrained=False, img_size=224, num_classes=0, global_pool="token")
        return model
    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(timm, "create_model", create)
    monkeypatch.setattr(safetensors, "load_file", lambda *args, **kwargs: {"weight": torch.ones(1)})
    encoder = engine.DinoEncoder("cpu")
    result = encoder.encode([np.full((5, 8), 255, dtype=np.uint8)])
    np.testing.assert_array_equal(result, [[0, 1, 2, 3]])
    assert result.dtype == np.float32
    assert encoder.snapshot["revision"] == engine.MODEL_REVISION
    assert encoder.snapshot["download_authentication"] == "anonymous"
    assert len(calls) == (3 if cached else 6)
    if cached:
        assert all(call.get("local_files_only") for call in calls)


def test_public_model_download_failure(monkeypatch):
    pytest.importorskip("timm")
    hub = pytest.importorskip("huggingface_hub")
    def fail(**kwargs):
        raise OSError("unreachable")
    monkeypatch.setattr(hub, "hf_hub_download", fail)
    with pytest.raises(ValueError, match="ein Konto oder Token ist nicht erforderlich"):
        engine.DinoEncoder("cpu")


def test_geometry_label_independence_and_three_class_metrics():
    pytest.importorskip("sklearn")
    pytest.importorskip("umap")
    rng = np.random.default_rng(10)
    features = np.concatenate([rng.normal(center, .03, (8, 6)) for center in [-10, 0, 10]]).astype(np.float32)
    cfg = config()
    phases = []
    result = engine.analyze_features(features, cfg, lambda step, *args: phases.append(step))
    changed = cfg.model_copy(deep=True)
    for item in changed.intervals:
        item.label = "buffer"
    again = engine.analyze_features(features, changed, lambda *args: None)
    for a, b in zip(result[:3], again[:3]):
        np.testing.assert_allclose(a, b)
    metrics = engine.evaluate_clusters([label for label in engine.LABELS for _ in range(8)], result[2], result[3])
    assert metrics["ari"] == pytest.approx(1)
    assert metrics["nmi"] == pytest.approx(1)
    assert sum(row["buffer"] for row in metrics["contingency"]) == 8
    assert metrics["pca_retained_variance"] >= .95
    assert phases == ["pca", "umap", "kmeans"]
    for values, count in [(np.ones((4, 3)), 2), (np.array([[0, 0], [0, 0], [1, 1], [1, 1]]), 3)]:
        with pytest.raises(ValueError):
            engine.analyze_features(values, config(cluster_count=count), lambda *args: None)


@pytest.fixture
def source(tmp_path, monkeypatch):
    db = make_db()
    root = tmp_path / "images"
    root.mkdir()
    for i in range(24):
        Image.fromarray(np.full((8, 8), i * 10, dtype=np.uint8)).save(root / f"{(START + timedelta(seconds=i)):%Y%m%d_%H%M%S}.tif")
    dataset = models.Dataset(name="Images", root_path=str(root), status="ready", timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})", timestamp_format="%Y%m%d_%H%M%S")
    db.add(dataset); db.flush()
    folder = models.DatasetFolder(dataset_id=dataset.id, relative_path=".", image_count=24)
    db.add(folder); db.flush()
    selected = models.TrainingDataset(dataset_id=dataset.id, name="Selection", usage_label="test")
    db.add(selected); db.flush()
    db.add(models.TrainingDatasetRule(training_dataset_id=selected.id, folder_id=folder.id, start_timestamp=START, end_timestamp=START + timedelta(seconds=23), stride=2))
    pipeline = models.PreprocessingPipeline(name="Original", graph={"nodes": [{"id": "load", "type": "load_image", "config": {}}], "edges": []})
    db.add(pipeline); db.commit()
    monkeypatch.setattr(service, "artifact_dir", lambda id: tmp_path / "runs" / str(id))
    monkeypatch.setattr("app.training.data.data_dir", lambda: tmp_path)
    cfg = config(training_dataset_id=selected.id, preprocessing_pipeline_id=pipeline.id, intervals=[
        interval("n", "normal", 0, 8), interval("a", "anomaly", 8, 16), interval("b", "buffer", 16, 24)])
    yield db, cfg, pipeline, tmp_path
    db.close()


class FakeEncoder:
    snapshot = {"model_id": "test", "revision": "fixed"}
    def __init__(self, device):
        self.device = device
    def encode(self, images):
        return np.array([[image.mean(), image.mean() ** 2, image.mean() / 3] for image in images], dtype=np.float32)


def test_stride_snapshot_and_end_to_end_exports(source):
    db, cfg, pipeline, _ = source
    preview = service.preview(db, cfg)
    assert preview.total == 12
    assert preview.image.startswith("data:image/png;base64,")
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    run = db.get(models.RepresentationRun, queued.id)
    pipeline.graph = {"nodes": [], "edges": []}
    db.commit()
    manifest = json.loads((service.artifact_dir(run.id) / "manifest.json").read_text())
    assert [int(row["timestamp"][-2:]) for row in manifest["samples"]] == list(range(0, 24, 2))
    assert run.dataset_snapshot["rules"][0]["stride"] == 2
    assert run.pipeline_snapshot["graph"]["nodes"]
    phases = []
    metrics = service.calculate(run, lambda phase, *a: phases.append(phase), lambda *a: None, FakeEncoder)
    assert metrics["sample_count"] == 12
    assert metrics["label_counts"] == {"normal": 4, "anomaly": 4, "buffer": 4}
    assert phases[-1] == "saving"
    saved = np.load(service.artifact_dir(run.id) / "features.npz")
    assert saved["features"].shape == (12, 3)
    assert saved["features"][1, 0] == 20
    points = json.loads((service.artifact_dir(run.id) / "points.json").read_text())
    assert points[4]["label"] == "anomaly" and points[4]["event_id"] == "A1"
    assert service.artifact_path(db, run.id, "samples.csv") is None
    run.status = "finished"; run.result = metrics; db.commit()
    assert service.artifact_path(db, run.id, "samples.csv").is_file()
    assert service.artifact_path(db, run.id, "../manifest.json") is None
    assert service.get_run(db, run.id).result.ari == metrics["ari"]


def test_worker_lifecycle_failure_abort_and_queue(source, monkeypatch):
    db, cfg, _, _ = source
    factory = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr("app.database.SessionLocal", factory)
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    from app.training.scheduler import _KINDS, normalize_queue_ranks
    assert _KINDS["dinov3_analysis"]["model"] == models.RepresentationRun
    assert not _KINDS["dinov3_analysis"].get("force_cpu")
    normalize_queue_ranks(db)
    assert service.get_run(db, queued.id).queue_rank == 1
    assert service.abort_run(db, queued.id).status == "aborted"
    service.run_scheduled(queued.id)
    db.expire_all()
    assert service.get_run(db, queued.id).status == "aborted"
    assert service.delete_run(db, queued.id)
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    def fail(*args):
        raise ValueError("test failure")
    monkeypatch.setattr(service, "calculate", fail)
    service.run_scheduled(queued.id)
    db.expire_all()
    assert service.get_run(db, queued.id).status == "failed"
    assert service.get_run(db, queued.id).error_message == "test failure"


def test_changed_source_is_not_silently_skipped(source):
    db, cfg, _, _ = source
    run = service.enqueue(db, cfg, wake_scheduler=False)
    manifest = json.loads((service.artifact_dir(run.id) / "manifest.json").read_text())
    Path(manifest["samples"][0]["file_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="verändert"):
        service.calculate(db.get(models.RepresentationRun, run.id), lambda *a: None, lambda *a: None, FakeEncoder)


def test_api_preview_run_status_abort_delete(source, monkeypatch):
    from fastapi.testclient import TestClient
    from app.database import get_db
    from app.main import app
    db, cfg, _, _ = source
    monkeypatch.setattr(service.scheduler, "wake", lambda: None)
    app.dependency_overrides[get_db] = lambda: db
    try:
        client = TestClient(app)
        preview = client.post("/api/dinov3-analysis/preview", json=cfg.model_dump(mode="json"))
        assert preview.status_code == 200
        assert preview.json()["label_counts"]["buffer"] == 4
        created = client.post("/api/dinov3-analysis/runs", json=cfg.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        id = created.json()["id"]
        base = f"/api/dinov3-analysis/runs/{id}"
        assert client.get(base).json()["config"]["cluster_count"] == 3
        assert client.get("/api/dinov3-analysis/runs").json()[0]["id"] == id
        assert client.get(base + "/results").status_code == 404
        assert client.get(base + "/log").json() == {"log": ""}
        assert client.delete(base).status_code == 409
        assert client.post(base + "/abort").json()["status"] == "aborted"
        assert client.delete(base).status_code == 204
        assert client.get(base).status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_project_api_and_artifacts_are_isolated(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import database
    from app.database import project_context
    from app.main import app
    from app.projects import create_project, initialize_catalog
    from tests.test_projects import configure_catalog
    configure_catalog(monkeypatch, tmp_path)
    initialize_catalog()
    one = create_project("Representation One", "Test")
    two = create_project("Representation Two", "Test")
    for project in (one, two):
        with project_context(project.database_url, project.artifact_dir):
            with database.SessionLocal() as session:
                dataset = models.TrainingDataset(name=project.name, usage_label="test")
                session.add(dataset); session.flush()
                run = models.RepresentationRun(training_dataset_id=dataset.id, training_dataset_name=project.name,
                    config=config(training_dataset_id=dataset.id).model_dump(mode="json"), dataset_snapshot={},
                    pipeline_snapshot={}, status="finished", current_step="finished")
                session.add(run); session.commit()
                service.write_json(service.artifact_dir(run.id) / "manifest.json", {"project": project.id})
    client = TestClient(app)
    assert client.get("/api/dinov3-analysis/runs").status_code == 400
    for project in (one, two):
        headers = {"X-MLTrace-Project-ID": project.id}
        runs = client.get("/api/dinov3-analysis/runs", headers=headers)
        assert runs.status_code == 200, runs.text
        assert runs.json()[0]["training_dataset_name"] == project.name
        artifact = client.get("/api/dinov3-analysis/runs/1/artifacts/manifest.json", headers=headers)
        assert artifact.json() == {"project": project.id}
    assert client.get("/api/dinov3-analysis/runs", headers={"X-MLTrace-Project-ID": "missing"}).status_code == 404


def test_worker_success_updates_status_and_persists_metadata(source, monkeypatch):
    db, cfg, _, _ = source
    factory = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr("app.database.SessionLocal", factory)
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    real_calculate = service.calculate
    monkeypatch.setattr(service, "calculate", lambda run, report, save: real_calculate(run, report, save, FakeEncoder))
    service.run_scheduled(queued.id)
    db.expire_all()
    run = service.get_run(db, queued.id)
    assert run.status == "finished" and run.current_step == "finished"
    assert run.processed_images == run.total_images == 12
    assert run.model_snapshot["revision"] == "fixed"
    assert run.duration_seconds is not None and run.ended_at is not None
    assert run.result.label_counts["buffer"] == 4


def test_registry_tracks_representation_dependency_and_cleanup(source):
    from app.registry.specs import ENTITY_SPECS
    from app import services
    db, cfg, _, _ = source
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    dependencies = ENTITY_SPECS["training_dataset"].dependents(db, cfg.training_dataset_id)
    assert any(item.entity_type == "representation_run" and item.id == queued.id for item in dependencies)
    with pytest.raises(ValueError, match="representation analyses"):
        services.delete_training_dataset(db, cfg.training_dataset_id)
    service.abort_run(db, queued.id)
    assert ENTITY_SPECS["representation_run"].deleter(db, queued.id)
    assert not service.artifact_dir(queued.id).exists()


def test_dispatch_waits_for_parent_pid_and_device(source, monkeypatch):
    from app.analysis import dinov3_worker as worker
    db, cfg, _, _ = source
    factory = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr("app.database.SessionLocal", factory)
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    def dispatch(_):
        with factory() as session:
            row = session.get(models.RepresentationRun, queued.id)
            row.status = "running"; row.pid = 12345; row.device = "GPU:2"
            session.commit()
    monkeypatch.setattr(worker.time, "sleep", dispatch)
    assert worker.wait_for_dispatch(queued.id, 12345)
    db.expire_all()
    assert db.get(models.RepresentationRun, queued.id).device == "GPU:2"
    with factory() as session:
        row = session.get(models.RepresentationRun, queued.id)
        row.status = "aborted"; session.commit()
    assert not worker.wait_for_dispatch(queued.id, 12345)


def test_running_abort_is_durable_and_never_signals_without_pid(source, monkeypatch):
    db, cfg, _, _ = source
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    row = db.get(models.RepresentationRun, queued.id)
    row.status = "running"; db.commit()
    signals = []
    monkeypatch.setattr(service.scheduler, "request_abort", lambda *args: signals.append(args))
    result = service.abort_run(db, queued.id)
    assert result.cancel_requested and not signals
    monkeypatch.setattr("app.database.SessionLocal", sessionmaker(bind=db.get_bind()))
    service.run_scheduled(queued.id)
    db.expire_all()
    assert service.get_run(db, queued.id).status == "aborted"


def test_restart_reconciles_workers_without_restarting_live_jobs(source, monkeypatch):
    import importlib
    scheduler_module = importlib.import_module('app.training.scheduler')
    db, cfg, _, root = source
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    row = db.get(models.RepresentationRun, queued.id)
    row.status = "running"; row.pid = 12345; db.commit()
    monkeypatch.setattr(scheduler_module, "SessionLocal", sessionmaker(bind=db.get_bind()))
    monkeypatch.setattr(scheduler_module, "list_projects", lambda: [SimpleNamespace(id="test", database_url="sqlite:///:memory:", artifact_dir=str(root))])
    monkeypatch.setattr(scheduler_module, "_pid_alive", lambda pid: True)
    scheduler = scheduler_module.JobScheduler()
    monkeypatch.setattr(scheduler, "_apply_worker_results", lambda *a, **kw: None)
    monkeypatch.setattr("app.analysis.spatial_abort.reconcile_project", lambda *a: None)
    scheduler._reconcile_on_startup()
    db.expire_all()
    assert service.get_run(db, queued.id).status == "running"
    monkeypatch.setattr(scheduler_module, "_pid_alive", lambda pid: False)
    scheduler._reconcile_on_startup()
    db.expire_all()
    assert service.get_run(db, queued.id).status == "failed"
