from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import models
from app.analysis import spatial_sensitivity
from app.analysis.spatial_sensitivity import build_roi_mask, calculate, compute_maps, load_valid_uint16, map_metrics, nearest_valid
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
        "training_dataset_ids": [1], "events": [{"id": "U1", "training_dataset_id": 1, "start": base.isoformat(), "end": (base + timedelta(minutes=30)).isoformat()}],
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
    assert successful == 4 and failed == 0
