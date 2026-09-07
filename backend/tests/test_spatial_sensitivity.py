from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.analysis import spatial_sensitivity
from app.analysis.spatial_sensitivity import build_roi_mask, calculate, compute_maps, configuration_signature, load_valid_uint16, map_metrics, nearest_valid
from app.database import Base
from app.database import get_db
from app.main import app
from app.schemas import SpatialSensitivityConfigurationCreate, SpatialSensitivityRunCreate, SpatialSensitivityWarpPreviewRequest
from app.training.data import ResolvedDatasetImage


def test_robust_change_maps_and_unclipped_metrics() -> None:
    normal = np.array([[[1, 1], [4, 4]], [[1, 3], [4, 8]], [[1, 5], [4, 12]]], dtype=np.uint16)
    event = np.array([[[11, 5], [4, 20]], [[11, 5], [4, 20]]], dtype=np.uint16)
    maps = compute_maps(normal, event, epsilon=1.0)
    np.testing.assert_array_equal(maps["median_normal"], [[1, 3], [4, 8]])
    np.testing.assert_array_equal(maps["difference"], [[10, 2], [0, 12]])
    np.testing.assert_array_equal(maps["mad_normal"], [[0, 2], [0, 4]])
    assert maps["z_map"][0, 0] == pytest.approx(10)
    mask = np.array([[True, False], [True, False]])
    metrics = map_metrics(maps["difference"], maps["z_map"], mask, 1.0)
    assert metrics["mean_d_in"] == 5
    assert metrics["mean_d_out"] == 7
    assert metrics["q_d"] == pytest.approx(5 / 8)
    assert metrics["p_in"] == pytest.approx(10 / 24 * 100)


def test_polygon_mask_keeps_inside_and_outside() -> None:
    mask = build_roi_mask([{"x": 1, "y": 1}, {"x": 4, "y": 1}, {"x": 4, "y": 4}, {"x": 1, "y": 4}], (6, 6))
    assert mask.shape == (6, 6)
    assert mask[2, 2]
    assert not mask[0, 0]
    with pytest.raises(ValueError, match="leave pixels outside"):
        build_roi_mask([{"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 5}, {"x": 0, "y": 5}], (6, 6))


def test_loader_requires_uint16_grayscale_1280x960(tmp_path: Path) -> None:
    valid = tmp_path / "valid.tif"
    Image.fromarray(np.full((960, 1280), 1234, dtype=np.uint16)).save(valid)
    loaded = load_valid_uint16(str(valid))
    assert loaded.dtype == np.uint16 and loaded.shape == (960, 1280)
    rgb = tmp_path / "rgb.tif"
    Image.fromarray(np.zeros((960, 1280, 3), dtype=np.uint8)).save(rgb)
    with pytest.raises(ValueError, match="two-dimensional"):
        load_valid_uint16(str(rgb))


def test_nearest_valid_uses_earlier_timestamp_as_tie_break(tmp_path: Path, monkeypatch) -> None:
    base = datetime(2026, 1, 1, 12)
    records = [ResolvedDatasetImage(str(tmp_path / name), base + timedelta(minutes=offset), "d", str(tmp_path), 1, ".", name)
               for name, offset in (("later.tif", 5), ("earlier.tif", -5))]
    monkeypatch.setattr("app.analysis.spatial_sensitivity.load_valid_uint16", lambda _: np.zeros((960, 1280), dtype=np.uint16))
    selected, _ = nearest_valid(records, base)
    assert selected.file_name == "earlier.tif"


def test_calculate_writes_reproducible_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(spatial_sensitivity, "HEIGHT", 60)
    monkeypatch.setattr(spatial_sensitivity, "WIDTH", 80)
    monkeypatch.setattr(spatial_sensitivity, "TILE_ROWS", 16)
    base = datetime(2026, 1, 1, 12)
    records = []
    for index, (minutes, value) in enumerate(((-60, 100), (-30, 100), (0, 140), (30, 140))):
        path = tmp_path / f"image_{index}.tif"
        Image.fromarray(np.full((60, 80), value, dtype=np.uint16)).save(path)
        records.append(ResolvedDatasetImage(str(path), base + timedelta(minutes=minutes), "dataset", str(tmp_path), 1, ".", path.name))
    dataset = models.TrainingDataset(id=1, name="dataset", usage_label="test")
    run = models.SpatialSensitivityRun(id=7, config={
        "training_dataset_ids": [1], "events": [{"id": "U1", "training_dataset_id": 1, "normal_start": (base - timedelta(minutes=30)).isoformat(), "start": base.isoformat(), "end": (base + timedelta(minutes=30)).isoformat()}],
        "normal_window_hours": 2, "epsilon": 1.0,
        "roi_points": [{"x": 10, "y": 10}, {"x": 60, "y": 10}, {"x": 60, "y": 50}, {"x": 10, "y": 50}],
        "roi_source_dataset_id": 1, "roi_source_timestamp": (base - timedelta(minutes=60)).isoformat(),
        "example_event_id": "U1", "example_normal_timestamp": (base - timedelta(minutes=30)).isoformat(), "example_event_timestamp": base.isoformat(),
    })
    output = tmp_path / "artifacts"
    monkeypatch.setattr(spatial_sensitivity, "_artifact_dir", lambda _: output)
    monkeypatch.setattr(spatial_sensitivity, "enumerate_training_dataset_image_records", lambda _: records)
    monkeypatch.setattr(spatial_sensitivity, "_save_figure", lambda fig, base: [])
    result, csv_path, archive, successful, failed = calculate(run, {1: dataset}, __import__("threading").Event(), lambda *_: None)
    assert result["events"][0]["mean_d_in"] == 40
    assert result["events"][0]["p_in"] == pytest.approx(result["events"][0]["area_in"] / (60 * 80) * 100)
    assert csv_path.is_file() and archive.is_file()
    assert (output / "event_001_arrays.npz").is_file()
    assert successful == 3 and failed == 0
    assert result["events"][0]["normal_image_count"] == 1


def _analysis_config() -> dict:
    base = datetime(2026, 1, 1, 12)
    return {
        "training_dataset_ids": [1], "events": [{"id": "U1", "training_dataset_id": 1, "start": base.isoformat(), "end": (base + timedelta(hours=1)).isoformat()}],
        "normal_window_hours": 2, "epsilon": 1.0,
        "roi_points": [{"x": 10, "y": 10}, {"x": 100, "y": 10}, {"x": 100, "y": 100}, {"x": 10, "y": 100}],
        "roi_source_dataset_id": 1, "roi_source_timestamp": (base - timedelta(hours=1)).isoformat(),
        "example_event_id": "U1", "example_normal_timestamp": (base - timedelta(minutes=30)).isoformat(), "example_event_timestamp": base.isoformat(),
    }


def test_configuration_signature_is_canonical_and_covers_analysis_fields() -> None:
    config = _analysis_config()
    reordered = dict(reversed(list(config.items())))
    assert configuration_signature(config) == configuration_signature(reordered)
    changed = {**config, "epsilon": 2.0}
    assert configuration_signature(config) != configuration_signature(changed)
    warped = {**config, "warp_preview_config": {"source_points": config["roi_points"], "output_shape_mode": "manual", "output_width": 320, "output_height": 240, "interpolation": "cubic"}}
    assert configuration_signature(config) != configuration_signature(warped)


def test_warp_preview_reuses_preprocessing_transformation(monkeypatch) -> None:
    monkeypatch.setattr(spatial_sensitivity, "HEIGHT", 60); monkeypatch.setattr(spatial_sensitivity, "WIDTH", 80)
    base = datetime(2026, 1, 1, 12); dataset = models.TrainingDataset(id=1, name="dataset", usage_label="test")
    record = ResolvedDatasetImage("image.tif", base, "dataset", ".", 1, ".", "image.tif")
    monkeypatch.setattr(spatial_sensitivity, "enumerate_training_dataset_image_records", lambda _: [record])
    monkeypatch.setattr(spatial_sensitivity, "load_valid_uint16", lambda _: np.arange(60 * 80, dtype=np.uint16).reshape(60, 80))
    class FakeDb:
        def scalar(self, _): return dataset
    payload = SpatialSensitivityWarpPreviewRequest(training_dataset_id=1, target_timestamp=base, warp={
        "source_points": [{"x": 5, "y": 5}, {"x": 70, "y": 5}, {"x": 70, "y": 50}, {"x": 5, "y": 50}],
        "output_shape_mode": "manual", "output_width": 32, "output_height": 24, "interpolation": "nearest",
    })
    result = spatial_sensitivity.warp_preview(FakeDb(), payload)
    assert (result.input_width, result.input_height) == (80, 60)
    assert (result.output_width, result.output_height) == (32, 24)
    assert result.interpolation == "nearest"


def test_saved_configuration_finds_exact_latest_finished_run_and_preserves_history() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(engine); db = sessionmaker(bind=engine)()
    try:
        payload = SpatialSensitivityConfigurationCreate(name="ROI Study", description="saved", config=_analysis_config())
        saved = spatial_sensitivity.create_configuration(db, payload)
        assert saved.latest_finished_run_id is None
        older = models.SpatialSensitivityRun(status="finished", current_step="finished", config=payload.config,
            config_signature=saved.config_signature, configuration_id=saved.id, dataset_snapshot=[], ended_at=datetime(2026, 1, 2))
        newer = models.SpatialSensitivityRun(status="finished", current_step="finished", config=payload.config,
            config_signature=saved.config_signature, configuration_id=None, dataset_snapshot=[], ended_at=datetime(2026, 1, 3))
        failed = models.SpatialSensitivityRun(status="failed", current_step="failed", config=payload.config,
            config_signature=saved.config_signature, dataset_snapshot=[], ended_at=datetime(2026, 1, 4))
        db.add_all([older, newer, failed]); db.commit()
        loaded = spatial_sensitivity.get_configuration(db, saved.id)
        assert loaded and loaded.latest_finished_run_id == newer.id
        changed = SpatialSensitivityConfigurationCreate(name="ROI Study", description="changed", config={**payload.config, "epsilon": 2})
        updated = spatial_sensitivity.update_configuration(db, saved.id, changed)
        assert updated and updated.latest_finished_run_id is None
        assert spatial_sensitivity.delete_configuration(db, saved.id)
        assert db.get(models.SpatialSensitivityRun, older.id) is not None
    finally:
        db.close()


def test_enqueue_records_saved_configuration_and_rejects_dirty_payload() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(engine); db = sessionmaker(bind=engine)()
    try:
        db.add(models.TrainingDataset(id=1, name="dataset", usage_label="test")); db.commit()
        saved = spatial_sensitivity.create_configuration(db, SpatialSensitivityConfigurationCreate(name="Saved", config=_analysis_config()))
        created = spatial_sensitivity.enqueue(db, SpatialSensitivityRunCreate(**_analysis_config(), configuration_id=saved.id), wake_scheduler=False)
        assert created.configuration_id == saved.id
        assert created.config_signature == saved.config_signature
        with pytest.raises(ValueError, match="differs"):
            spatial_sensitivity.enqueue(db, SpatialSensitivityRunCreate(**{**_analysis_config(), "epsilon": 3}, configuration_id=saved.id), wake_scheduler=False)
    finally:
        db.close()


def test_spatial_configuration_api_crud_and_latest_result() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine); factory = sessionmaker(bind=engine)
    def override_db():
        db = factory()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_db
    client = TestClient(app)
    try:
        created = client.post("/api/spatial-sensitivity/configurations", json={"name": "API config", "description": "first", "config": _analysis_config()})
        assert created.status_code == 200, created.text
        body = created.json(); assert body["latest_finished_run_id"] is None
        duplicate = client.post("/api/spatial-sensitivity/configurations", json={"name": "api CONFIG", "config": _analysis_config()})
        assert duplicate.status_code == 409
        db = factory(); run = models.SpatialSensitivityRun(status="finished", current_step="finished", config=_analysis_config(),
            config_signature=body["config_signature"], configuration_id=body["id"], dataset_snapshot=[], ended_at=datetime(2026, 2, 1)); db.add(run); db.commit(); run_id = run.id; db.close()
        loaded = client.get(f"/api/spatial-sensitivity/configurations/{body['id']}")
        assert loaded.status_code == 200 and loaded.json()["latest_finished_run_id"] == run_id
        updated = client.put(f"/api/spatial-sensitivity/configurations/{body['id']}", json={"name": "API config", "description": "changed", "config": {**_analysis_config(), "epsilon": 2}})
        assert updated.status_code == 200 and updated.json()["latest_finished_run_id"] is None
        assert client.delete(f"/api/spatial-sensitivity/configurations/{body['id']}").status_code == 204
        assert client.get(f"/api/spatial-sensitivity-runs/{run_id}").status_code == 200
    finally:
        app.dependency_overrides.clear()
