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
from app.training import folder_time_index


def test_robust_change_maps_and_unclipped_metrics() -> None:
    normal = np.array([[[1, 1], [4, 4]], [[1, 3], [4, 8]], [[1, 5], [4, 12]]], dtype=np.uint16)
    event = np.array([[[11, 5], [4, 20]], [[11, 5], [4, 20]]], dtype=np.uint16)
    maps = compute_maps(normal, event, epsilon=1.0)
    np.testing.assert_array_equal(maps["median_normal"], [[1, 3], [4, 8]])
    np.testing.assert_array_equal(maps["D_med"], [[10, 2], [0, 12]])
    np.testing.assert_array_equal(maps["mad_normal"], [[0, 2], [0, 4]])
    assert maps["R_med"][0, 0] == pytest.approx(10)
    mask = np.array([[True, False], [True, False]])
    metrics = map_metrics(maps, mask)
    assert metrics["D_med_in"] == 5
    assert metrics["D_med_out"] == 7
    assert metrics["Q_D_med"] == pytest.approx(5 / 7)
    assert metrics["P_in_D_med"] == pytest.approx(10 / 24 * 100)


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
    assert result["events"][0]["D_med_in"] == 40
    assert result["events"][0]["P_in_D_med"] == pytest.approx(result["events"][0]["area_in"] / (60 * 80) * 100)
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
    changed = {**config, "normal_sample_size": 500}
    assert configuration_signature(config) != configuration_signature(changed)
    warped = {**config, "warp_preview_config": {"source_points": config["roi_points"], "output_shape_mode": "manual", "output_width": 320, "output_height": 240, "interpolation": "cubic"}}
    assert configuration_signature(config) != configuration_signature(warped)
    sampled = SpatialSensitivityRunCreate(**config)
    assert sampled.normal_sample_size == 1000 and sampled.event_sample_size == 1000 and sampled.sampling_seed == 42
    with pytest.raises(ValueError):
        SpatialSensitivityRunCreate(**{**config, "sampling_seed": 43})


def test_deterministic_sample_is_bounded_and_backfills_invalid_images(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(spatial_sensitivity, "HEIGHT", 2); monkeypatch.setattr(spatial_sensitivity, "WIDTH", 3)
    base = datetime(2026, 1, 1)
    records = [ResolvedDatasetImage(str(tmp_path / f"{index}.tif"), base + timedelta(minutes=index), "d", str(tmp_path), 1, ".", f"{index}.tif") for index in range(8)]
    dataset = models.TrainingDataset(id=4, name="d", usage_label="train")
    monkeypatch.setattr(spatial_sensitivity, "enumerate_training_dataset_image_records", lambda _: records)
    invalid_paths: set[str] = set()
    def fake_load(path: str):
        if path in invalid_paths: raise ValueError("broken")
        return np.full((2, 3), int(Path(path).stem), dtype=np.uint16)
    monkeypatch.setattr(spatial_sensitivity, "load_valid_uint16", fake_load)
    first = tmp_path / "first.sqlite3"; second = tmp_path / "second.sqlite3"; changed = tmp_path / "changed.sqlite3"
    for path, seed in ((first, 42), (second, 42), (changed, 43)):
        spatial_sensitivity._build_candidate_table(dataset, base, base + timedelta(minutes=7), end_inclusive=True,
            event_id="U1", window_kind="normal", seed=seed, path=path, abort_event=__import__("threading").Event())
    order = lambda path: [record.file_path for record in spatial_sensitivity._candidate_records(path, dataset)]
    assert order(first) == order(second)
    assert order(first) != order(changed)
    invalid_paths.add(order(first)[0])
    stack, summary, rejected = spatial_sensitivity._stage_sample(first, dataset, 7, tmp_path / "stack.npy",
        event_id="U1", window_kind="normal", abort_event=__import__("threading").Event(), progress=lambda *_: None)
    assert stack.shape == (7, 2, 3)
    assert summary["valid_sample_size"] == 7 and summary["attempted_count"] == 8
    selected_paths = [item["path"] for item in summary["selected_valid_images"]]
    assert len(selected_paths) == len(set(selected_paths)) == 7
    assert len(rejected) == 1 and rejected[0]["path"] in invalid_paths


def test_disk_time_index_preserves_rule_stride_phase_for_clipped_range(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"; image_dir.mkdir()
    base = datetime(2026, 1, 1)
    for minute in range(10):
        (image_dir / f"image_20260101_00{minute:02d}00.tif").touch()
    source = models.Dataset(id=1, name="source", root_path=str(image_dir), status="ready",
        timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})", timestamp_format="%Y%m%d_%H%M%S")
    folder = models.DatasetFolder(id=2, dataset=source, relative_path=".", image_count=10,
        first_timestamp=base, last_timestamp=base + timedelta(minutes=9))
    selected = models.TrainingDataset(id=3, name="selected", usage_label="train")
    rule = models.TrainingDatasetRule(id=4, training_dataset=selected, folder=folder,
        start_timestamp=base, end_timestamp=base + timedelta(minutes=9), stride=2)
    selected.rules = [rule]
    monkeypatch.setattr(folder_time_index, "data_dir", lambda: tmp_path / "data")
    records = list(folder_time_index.iter_training_dataset_range_records(
        selected, base + timedelta(minutes=3), base + timedelta(minutes=7), end_inclusive=True))
    assert [record.timestamp_parsed.minute for record in records] == [4, 6]


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
        changed = SpatialSensitivityConfigurationCreate(name="ROI Study", description="changed", config={**payload.config, "normal_sample_size": 500})
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
            spatial_sensitivity.enqueue(db, SpatialSensitivityRunCreate(**{**_analysis_config(), "normal_sample_size": 300}, configuration_id=saved.id), wake_scheduler=False)
    finally:
        db.close()


def test_queued_spatial_run_can_be_aborted_immediately() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(engine); db = sessionmaker(bind=engine)()
    try:
        run = models.SpatialSensitivityRun(status="queued", current_step="queued", config=_analysis_config(),
            config_signature=configuration_signature(_analysis_config()), dataset_snapshot=[])
        db.add(run); db.commit()
        aborted = spatial_sensitivity.abort_run(db, run.id)
        assert aborted and aborted.status == "aborted" and aborted.current_step == "aborted"
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
        updated = client.put(f"/api/spatial-sensitivity/configurations/{body['id']}", json={"name": "API config", "description": "changed", "config": {**_analysis_config(), "normal_sample_size": 500}})
        assert updated.status_code == 200 and updated.json()["latest_finished_run_id"] is None
        assert client.delete(f"/api/spatial-sensitivity/configurations/{body['id']}").status_code == 204
        assert client.get(f"/api/spatial-sensitivity-runs/{run_id}").status_code == 200
    finally:
        app.dependency_overrides.clear()
