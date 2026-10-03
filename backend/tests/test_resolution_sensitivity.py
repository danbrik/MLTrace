from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
import numpy as np

from app import models
from app.analysis import resolution_sensitivity
from app.schemas import ResolutionSensitivityRunCreate
from app.training.data import ResolvedDatasetImage
from tests.test_testing_service import make_db


def _record(path: str, timestamp: datetime) -> ResolvedDatasetImage:
    return ResolvedDatasetImage(
        file_path=path, timestamp_parsed=timestamp, dataset_name="camera",
        dataset_root_path="/images", folder_id=1, folder_relative_path=".", file_name=Path(path).name,
    )


def _payload(**changes) -> ResolutionSensitivityRunCreate:
    base = {
        "training_dataset_id": 1,
        "pipeline_ids": [1, 2, 3, 4],
        "samples_per_interval": 2,
        "intervals": [
            {"id": "normal", "name": "Normal", "type": "normal", "start": "2026-01-01T00:00:00", "end": "2026-01-01T01:00:00"},
            {"id": "event", "name": "Event", "type": "event", "start": "2026-01-01T02:00:00", "end": "2026-01-01T03:00:00"},
        ],
    }
    base.update(changes)
    return ResolutionSensitivityRunCreate(**base)


def test_configuration_requires_distinct_pipelines_and_non_overlapping_normal_and_event() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        _payload(pipeline_ids=[1, 1, 2, 3])
    with pytest.raises(ValidationError, match="overlap"):
        _payload(intervals=[
            {"id": "normal", "name": "Normal", "type": "normal", "start": "2026-01-01T00:00:00", "end": "2026-01-01T02:30:00"},
            {"id": "event", "name": "Event", "type": "event", "start": "2026-01-01T02:00:00", "end": "2026-01-01T03:00:00"},
        ])


def test_sampling_uses_midpoints_nearest_image_duplicates_and_earlier_tie() -> None:
    start = datetime(2026, 1, 1)
    images = [_record("/images/a.tif", start + timedelta(minutes=10)), _record("/images/b.tif", start + timedelta(minutes=50))]
    rows = resolution_sensitivity.sample_intervals(images, [{
        "id": "normal", "name": "Normal", "type": "normal",
        "start": start.isoformat(), "end": (start + timedelta(hours=1)).isoformat(),
    }], 4)
    assert [row["image"].file_path for row in rows] == ["/images/a.tif", "/images/a.tif", "/images/b.tif", "/images/b.tif"]
    assert [row["duplicate"] for row in rows] == [False, True, False, True]

    tie = resolution_sensitivity.sample_intervals(images, [{
        "id": "normal", "name": "Normal", "type": "normal",
        "start": start.isoformat(),
        "end": (start + timedelta(hours=1)).isoformat(),
    }], 1)
    assert tie[0]["image"].file_path == "/images/a.tif"


def test_pipeline_mapping_and_enqueue_snapshot() -> None:
    db = make_db()
    try:
        dataset = models.Dataset(name="camera", root_path="/tmp", status="ready")
        db.add(dataset); db.flush()
        training = models.TrainingDataset(dataset_id=dataset.id, name="sensitivity", usage_label="test")
        db.add(training); db.flush()
        pipelines = []
        for resolution in resolution_sensitivity.RESOLUTIONS:
            pipeline = models.PreprocessingPipeline(
                name=f"p{resolution}", graph={"nodes": [{"id": "load", "type": "load_image", "config": {}}], "edges": []},
                output_width=resolution, output_height=resolution,
            )
            db.add(pipeline); pipelines.append(pipeline)
        db.commit()
        payload = _payload(training_dataset_id=training.id, pipeline_ids=[pipeline.id for pipeline in pipelines])
        run = resolution_sensitivity.enqueue(db, payload, wake_scheduler=False)
        assert run.status == "queued"
        assert [item["resolution"] for item in run.pipeline_snapshot] == [840, 512, 256, 128]
        assert run.config["samples_per_interval"] == 2
        assert run.config["interval_end_inclusive"] is True
    finally:
        db.close()


def test_metric_helpers() -> None:
    summary = resolution_sensitivity._metric_summary([1.0, 2.0, 3.0, 4.0])
    assert summary == pytest.approx({"median": 2.5, "q25": 1.75, "q75": 3.25, "iqr": 1.5})
    values = resolution_sensitivity._features(__import__("numpy").array([[0.0, 2.0], [4.0, 6.0]]))
    assert values["mean_intensity"] == pytest.approx(3.0)
    assert values["q95_intensity"] == pytest.approx(5.7)
    assert values["spatial_std_intensity"] == pytest.approx(2.2360679)


def test_calculation_keeps_sources_paired_and_computes_retention(tmp_path, monkeypatch) -> None:
    start = datetime(2026, 1, 1)
    records = [
        _record("/images/normal.tif", start + timedelta(minutes=30)),
        _record("/images/event.tif", start + timedelta(hours=2, minutes=30)),
    ]

    class FakePipeline:
        def __init__(self, resolution: int): self.resolution = resolution
        def run(self, path: str):
            base = 0.0 if "normal" in path else 10.0 * self.resolution / 840.0
            return np.full((self.resolution, self.resolution), base, dtype=np.float32)

    monkeypatch.setattr(resolution_sensitivity, "enumerate_training_dataset_image_records", lambda dataset: records)
    monkeypatch.setattr(
        resolution_sensitivity, "compile_pipeline",
        lambda graph: FakePipeline(int(graph.nodes[0].config["resolution"])),
    )
    monkeypatch.setattr(resolution_sensitivity, "_artifact_dir", lambda run_id: tmp_path / str(run_id))
    run = models.ResolutionSensitivityRun(
        id=77, training_dataset_id=1, training_dataset_name="dataset", status="running",
        config=_payload(ssim_data_range=10.0, samples_per_interval=1).model_dump(mode="json"),
        pipeline_snapshot=[{
            "resolution": resolution, "pipeline_id": index, "name": f"p{resolution}",
            "graph": {"nodes": [{"id": "load", "type": "load_image", "config": {"resolution": resolution}}], "edges": []},
        } for index, resolution in enumerate(resolution_sensitivity.RESOLUTIONS, 1)],
    )
    result, detail, summary, data_range, successful, failed = resolution_sensitivity.calculate(
        run, object(), __import__("threading").Event(), lambda *args: None,
    )
    assert data_range == 10.0
    assert successful == 2 and failed == 0
    assert detail.is_file() and summary.is_file()
    mean_rows = [row for row in result["separations"] if row["event_id"] == "event" and row["feature"] == "mean_intensity"]
    by_resolution = {row["resolution"]: row for row in mean_rows}
    assert by_resolution[840]["retention"] == pytest.approx(1.0)
    assert by_resolution[512]["retention"] == pytest.approx(512 / 840)
    assert result["overview"][0]["ssim"]["median"] == pytest.approx(1.0)


def test_new_runs_allow_single_points_but_reject_shared_boundaries():
    payload = _payload()
    values = payload.model_dump(mode='json')
    values['intervals'][0]['end'] = values['intervals'][0]['start']
    assert ResolutionSensitivityRunCreate(**values).intervals[0].start == payload.intervals[0].start
    values = payload.model_dump(mode='json')
    values['intervals'][1]['start'] = values['intervals'][0]['end']
    with pytest.raises(ValidationError, match='overlap'):
        ResolutionSensitivityRunCreate(**values)
    values['intervals'][0]['end'] = '2025-12-31T23:00:00'
    with pytest.raises(ValidationError, match='before start'):
        ResolutionSensitivityRunCreate(**values)


def test_inclusive_boundaries_single_point_and_legacy_sampling():
    start = datetime(2026, 1, 1)
    end = start + timedelta(seconds=10)
    images = [_record('/first.tif', start), _record('/last.tif', end)]
    interval = {'id': 'normal', 'name': 'Normal', 'type': 'normal', 'start': start.isoformat(), 'end': end.isoformat()}
    sampled = resolution_sensitivity.sample_intervals(images, [interval], 2)
    assert [row['image'].file_path for row in sampled] == ['/first.tif', '/last.tif']
    old = resolution_sensitivity.sample_intervals(images, [interval], 2, end_inclusive=False)
    assert [row['image'].file_path for row in old] == ['/first.tif', '/first.tif']
    point = {**interval, 'start': end.isoformat()}
    sampled = resolution_sensitivity.sample_intervals(images, [point], 3)
    assert [row['image'].file_path for row in sampled] == ['/last.tif'] * 3
    assert [row['duplicate'] for row in sampled] == [False, True, True]
    with pytest.raises(ValueError, match='no available image'):
        resolution_sensitivity.sample_intervals(images, [point], 1, end_inclusive=False)
    missing = {**point, 'start': (end + timedelta(seconds=1)).isoformat(), 'end': (end + timedelta(seconds=1)).isoformat()}
    with pytest.raises(ValueError, match='no available image'):
        resolution_sensitivity.sample_intervals(images, [missing], 1)


@pytest.mark.parametrize('stored_flag, expected', [(None, False), (True, True), (False, False)])
def test_worker_preserves_legacy_rule_and_honors_new_marker(monkeypatch, stored_flag, expected):
    config = _payload().model_dump(mode='json')
    if stored_flag is not None:
        config['interval_end_inclusive'] = stored_flag
    run = models.ResolutionSensitivityRun(config=config)
    monkeypatch.setattr(resolution_sensitivity, 'enumerate_training_dataset_image_records', lambda _: [])
    class SamplingReached(Exception): pass
    def capture(*args, **kwargs):
        assert kwargs['end_inclusive'] is expected
        raise SamplingReached()
    monkeypatch.setattr(resolution_sensitivity, 'sample_intervals', capture)
    with pytest.raises(SamplingReached):
        resolution_sensitivity.calculate(run, object(), __import__('threading').Event(), lambda *args: None)
