from datetime import datetime, timedelta
import json
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from app import models
from app.reference_image import engine, service
from app.reference_image.schemas import ReferenceImageConfig
from app.training.data import ResolvedDatasetImage
from tests.test_training_scheduler import make_db

START = datetime(2026, 1, 1)


def interval(start, end, rate=1):
    return {"start": (START + timedelta(seconds=start)).isoformat(),
            "end": (START + timedelta(seconds=end)).isoformat(), "sampling_rate": rate}


def config(**changes):
    return ReferenceImageConfig.model_validate({"training_dataset_id": 1, "preprocessing_pipeline_id": 1,
        "reference": interval(0, 6), "anomaly": interval(8, 22), "fps": 7, **changes})


def records(count):
    return [ResolvedDatasetImage(str(i), START + timedelta(seconds=i), "images", "/images", 1, ".", str(i)) for i in range(count)]


def test_regular_sampling_inclusive_bounds_deduplication_and_remainder():
    cfg = config(reference=interval(0, 45, 15), anomaly=interval(1, 31, 15))
    rows, preview = engine.select_records(list(reversed(records(47))) + records(47), cfg)
    assert [row["file_path"] for row in rows["reference"]] == ["14", "29", "44"]
    assert [row["file_path"] for row in rows["anomaly"]] == ["15", "30"]
    assert preview.reference.available == 46 and preview.reference.remainder == 1
    assert preview.anomaly.available == 31 and not preview.errors
    rows, preview = engine.select_records(records(47), config(reference=interval(12, 12)))
    assert rows["reference"][0]["file_path"] == "12"


def test_random_sampling_is_exact_unique_seeded_and_order_independent():
    cfg = config(reference={**interval(0, 39), "mode": "random", "count": 9, "seed": 42})
    one, _ = engine.select_records(records(40), cfg)
    two, _ = engine.select_records(list(reversed(records(40))), cfg)
    assert one == two
    assert len({row["file_path"] for row in one["reference"]}) == 9
    cfg.reference.seed = 43
    three, _ = engine.select_records(records(40), cfg)
    assert one["reference"] != three["reference"]
    cfg.reference.count = 41
    _, preview = engine.select_records(records(40), cfg)
    assert preview.reference.selected == 0 and preview.errors


@pytest.mark.parametrize("changes", [
    {"reference": interval(6, 0)}, {"anomaly": interval(0, 5, 0)},
    {"anomaly": {**interval(0, 5), "mode": "random"}},
    {"scale_mode": "manual"}, {"scale_limit": float("inf")}, {"scale_limit": 0}, {"fps": 0},
    {"reference": {**interval(0, 5), "mode": "random", "count": 0}},
])
def test_invalid_configuration(changes):
    with pytest.raises(ValidationError):
        config(**changes)


def test_signed_grayscale_constant_frames_manual_clipping_and_uint16():
    reference = engine.grayscale(np.array([[1000, 40000, 65000]], dtype=np.uint16))
    current = engine.grayscale(np.array([[0, 40000, 65535]], dtype=np.uint16))
    difference = current - reference
    np.testing.assert_array_equal(difference, [[-1000, 0, 535]])
    np.testing.assert_array_equal(engine.render_difference(difference, 535), [[0, 128, 255]])
    np.testing.assert_array_equal(engine.render_difference(difference, 1000), [[0, 128, 196]])
    np.testing.assert_array_equal(engine.render_difference(np.zeros((2, 2)), 0), np.full((2, 2), 128))


@pytest.mark.parametrize("array", [np.zeros((4, 4, 3)), np.array([[np.nan]]), np.array([[np.inf]]), np.empty((0, 0))])
def test_invalid_pipeline_outputs(array):
    with pytest.raises(ValueError):
        engine.grayscale(array)
    with pytest.raises(ValueError, match="Bildgrößen"):
        engine.grayscale(np.zeros((4, 4)), (3, 4))


@pytest.mark.parametrize("second, expected, exact", [(10, 0, True), (11, 1, False), (0, 0, False), (50, 2, False)])
def test_timestamp_lookup(second, expected, exact):
    frames = [{"index": i, "timestamp": (START + timedelta(seconds=s)).isoformat(), "distance": i}
              for i, s in enumerate([10, 20, 30])]
    result = engine.resolve_frame(frames, START + timedelta(seconds=second))
    assert result["frame"]["index"] == expected and result["exact"] is exact


@pytest.fixture
def source(tmp_path, monkeypatch):
    db = make_db()
    root = tmp_path / "images"
    root.mkdir()
    for i in range(24):
        Image.fromarray(np.full((181, 321), 1000 + i * 100, dtype=np.uint16)).save(root / f"{(START + timedelta(seconds=i)):%Y%m%d_%H%M%S}.tif")
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
    monkeypatch.setattr("app.database.SessionLocal", sessionmaker(bind=db.get_bind()))
    yield db, config(training_dataset_id=selected.id, preprocessing_pipeline_id=pipeline.id), pipeline
    db.close()


def test_saved_stride_is_applied_before_clipping_and_additional_sampling(source):
    db, cfg, _ = source
    cfg.reference.start = START + timedelta(seconds=1)
    cfg.reference.end = START + timedelta(seconds=9)
    cfg.reference.sampling_rate = 2
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    manifest = json.loads((service.artifact_dir(queued.id) / "manifest.json").read_text())
    assert [row["timestamp"][-2:] for row in manifest["samples"]["reference"]] == ["04", "08"]
    assert manifest["selection"]["reference"]["available"] == 4
    cfg.reference.sampling_rate = 15
    assert service.preview(db, cfg).errors
    with pytest.raises(ValueError, match="keine Bilder"):
        service.enqueue(db, cfg, wake_scheduler=False)


def test_worker_end_to_end_frozen_pipeline_video_and_png(source):
    db, cfg, pipeline = source
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    pipeline.graph = {"nodes": [], "edges": []}
    db.commit()
    assert service.artifact_path(db, queued.id, "reference.png") is None
    service.run_scheduled(queued.id)
    db.expire_all()
    run = service.get_run(db, queued.id)
    assert run.status == "finished", run.error_message
    assert run.result["reference_count"] == 4 and run.result["frame_count"] == 8
    assert run.result["scale_limit"] == 1900
    assert run.processed_images == run.total_images == 8
    np.testing.assert_allclose(np.load(service.artifact_dir(run.id) / "reference.npy"), 1300)
    results = service.results(db, run.id)
    assert [frame["distance"] for frame in results["frames"]] == list(range(500, 2000, 200))
    png = np.array(Image.open(service.artifact_path(db, run.id, "frame_000000.png")))
    assert png.shape == (182, 322, 3)
    expected = round(128 + 127 * 500 / 1900)
    assert np.all(png[100:, :100] == expected)
    assert np.any(png[:50, 100:] != expected)  # Timestamp is at the top right.
    last = np.array(Image.open(service.artifact_path(db, run.id, "frame_000007.png")))
    assert np.all(last[100:, :100] == 255)
    path = service.artifact_path(db, run.id, "video.mp4")
    encoded = path.read_bytes()
    assert b"avc1" in encoded and encoded.find(b"moov") < encoded.find(b"mdat")
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 7
        ok, frame = capture.read()
        assert ok and frame.shape == png.shape
        assert np.abs(frame[100:, :100].astype(float) - png[100:, :100]).mean() < 5
    finally:
        capture.release()
    assert not (service.artifact_dir(run.id) / "differences").exists()
    assert service.artifact_path(db, run.id, "frame_000008.png") is None
    assert service.artifact_path(db, run.id, "../manifest.json") is None
    assert service.artifact_path(db, run.id, "manifest.json") is None


def test_manual_scale_and_all_unchanged_runs(source):
    db, cfg, _ = source
    cfg.reference = cfg.reference.model_copy(update={"end": START})
    cfg.anomaly = cfg.anomaly.model_copy(update={"start": START, "end": START})
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    service.run_scheduled(queued.id)
    db.expire_all()
    assert service.get_run(db, queued.id).result["scale_limit"] == 0
    png = np.array(Image.open(service.artifact_path(db, queued.id, "frame_000000.png")))
    assert np.all(png[100:, :100] == 128)
    cfg.anomaly.end = START + timedelta(seconds=4)
    cfg.scale_mode = "manual"; cfg.scale_limit = 100
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    service.run_scheduled(queued.id)
    db.expire_all()
    assert service.get_run(db, queued.id).result["scale_limit"] == 100
    png = np.array(Image.open(service.artifact_path(db, queued.id, "frame_000001.png")))
    assert np.all(png[100:, :100] == 255)


@pytest.mark.parametrize("failure", ["changed", "missing", "size", "color"])
def test_failed_runs_do_not_publish_outputs(source, failure):
    db, cfg, _ = source
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    directory = service.artifact_dir(queued.id)
    manifest = json.loads((directory / "manifest.json").read_text())
    sample = manifest["samples"]["anomaly"][0]
    path = Path(sample["file_path"])
    if failure == "missing":
        path.unlink()
    else:
        shape = (8, 8, 3) if failure == "color" else (8, 8)
        Image.fromarray(np.ones(shape, dtype=np.uint8)).save(path)
        if failure in {"size", "color"}:
            sample.update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            service.write_json(directory / "manifest.json", manifest)
    service.run_scheduled(queued.id)
    db.expire_all()
    run = service.get_run(db, queued.id)
    assert run.status == "failed" and run.error_message
    assert service.artifact_path(db, queued.id, "reference.png") is None
    assert service.results(db, queued.id) is None
    assert not (directory / "differences").exists()


def test_abort_queue_running_and_registry_cleanup(source, monkeypatch):
    from app.registry.specs import ENTITY_SPECS
    from app.registry.service import _used_ids
    from app import services
    from app.training.scheduler import _KINDS, normalize_queue_ranks
    db, cfg, _ = source
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    assert _KINDS["reference_image"]["force_cpu"]
    normalize_queue_ranks(db)
    assert service.get_run(db, queued.id).queue_rank == 1
    with pytest.raises(ValueError):
        service.delete_run(db, queued.id)
    assert cfg.training_dataset_id in _used_ids(db, "training_dataset")
    assert any(item.entity_type == "reference_image_run" for item in ENTITY_SPECS["training_dataset"].dependents(db, cfg.training_dataset_id))
    with pytest.raises(ValueError, match="reference image"):
        services.delete_training_dataset(db, cfg.training_dataset_id)
    assert service.abort_run(db, queued.id).status == "aborted"
    service.run_scheduled(queued.id)
    assert ENTITY_SPECS["reference_image_run"].deleter(db, queued.id)
    assert not service.artifact_dir(queued.id).exists()
    running = service.enqueue(db, cfg, wake_scheduler=False)
    original = service.calculate
    def abort_during_render(run, report, event):
        def intercept(step, done, total):
            if step == "rendering":
                event.set()
            report(step, done, total)
        return original(run, intercept, event)
    monkeypatch.setattr(service, "calculate", abort_during_render)
    service.run_scheduled(running.id)
    db.expire_all()
    assert service.get_run(db, running.id).status == "aborted"
    assert service.results(db, running.id) is None


def test_api_preview_lifecycle_lookup_and_download(source, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.database import get_db
    from app.reference_image.api import router
    db, cfg, _ = source
    app = FastAPI(); app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(service.scheduler, "wake", lambda: None)
    client = TestClient(app)
    base = "/api/reference-image-analysis"
    assert client.post(base + "/preview", json=cfg.model_dump(mode="json")).json()["reference"]["selected"] == 4
    created = client.post(base + "/runs", json=cfg.model_dump(mode="json"))
    assert created.status_code == 200, created.text
    id = created.json()["id"]
    url = f"{base}/runs/{id}"
    assert client.get(url + "/results").status_code == 404
    assert client.get(url + "/artifacts/video.mp4").status_code == 404
    assert client.delete(url).status_code == 409
    service.run_scheduled(id); db.expire_all()
    assert client.get(url).json()["status"] == "finished"
    assert client.get(base + "/runs").json()[0]["id"] == id
    found = client.get(url + "/frame", params={"timestamp": (START + timedelta(seconds=9)).isoformat()}).json()
    assert found["frame"]["index"] == 1 and not found["exact"]
    download = client.get(url + "/artifacts/frame_000001.png?download=true")
    assert download.headers["content-type"] == "image/png"
    assert download.headers["content-disposition"].startswith("attachment")
    assert client.get(url + "/artifacts/video.mp4").headers["content-type"] == "video/mp4"
    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404


def test_migration_project_isolation_and_media_links(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import database
    from app.database import project_context
    from app.main import app
    from app.projects import create_project, initialize_catalog
    from tests.test_projects import configure_catalog
    configure_catalog(monkeypatch, tmp_path)
    initialize_catalog()
    projects = [create_project(name, "Test") for name in ("Reference One", "Reference Two")]
    for project in projects:
        with project_context(project.database_url, project.artifact_dir):
            with database.SessionLocal() as db:
                dataset = models.TrainingDataset(name=project.name, usage_label="test")
                db.add(dataset); db.flush()
                run = models.ReferenceImageRun(training_dataset_id=dataset.id, training_dataset_name=project.name,
                    config=config(training_dataset_id=dataset.id).model_dump(mode="json"), dataset_snapshot={},
                    pipeline_snapshot={}, status="finished", current_step="finished")
                db.add(run); db.commit()
                service.write_json(service.artifact_dir(run.id) / "results.json", {"project": project.id})
    client = TestClient(app)
    base = "/api/reference-image-analysis/runs"
    assert client.get(base).status_code == 400
    for project in projects:
        response = client.get(base, headers={"X-MLTrace-Project-ID": project.id})
        assert response.status_code == 200, response.text
        assert response.json()[0]["training_dataset_name"] == project.name
        assert client.get(base + "/1/artifacts/results.json", params={"project_id": project.id}).json() == {"project": project.id}
        jobs = client.get('/api/scheduler/jobs?scope=project', headers={"X-MLTrace-Project-ID": project.id})
        assert jobs.status_code == 200, jobs.text
        assert any(job["kind"] == "reference_image" for job in jobs.json())
    assert client.get(base, headers={"X-MLTrace-Project-ID": "missing"}).status_code == 404


def test_ranges_must_stay_inside_dataset(source):
    db, cfg, _ = source
    cfg.reference.start = START - timedelta(seconds=1)
    with pytest.raises(ValueError, match="Datensatzgrenzen"):
        service.preview(db, cfg)


def test_encoder_failure_hides_partial_files(source, monkeypatch):
    db, cfg, _ = source
    queued = service.enqueue(db, cfg, wake_scheduler=False)
    def failed_encoder(*args, **kwargs):
        raise ValueError("Could not encode browser-compatible MP4")
    monkeypatch.setattr("app.video.finalize_browser_mp4", failed_encoder)
    service.run_scheduled(queued.id)
    db.expire_all()
    assert service.get_run(db, queued.id).status == "failed"
    assert service.artifact_path(db, queued.id, "video.mp4") is None
    assert service.artifact_path(db, queued.id, "frame_000000.png") is None
    assert not (service.artifact_dir(queued.id) / "differences").exists()
