from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.artifact_signatures import artifact_signature
from app.database import Base
from app.schemas import (
    AnalysisImageComparisonRequest,
    HeatmapRunCreate,
    HeatmapVisualizationConfig,
    RoiDefinitionCreate,
    RoiPreviewRequest,
    TestingRunBulkCreate,
    TestingRunCreate as TestingRunCreatePayload,
)
from app.testing import service as testing_service
from app.testing.service import (
    ArtifactEvaluator,
    CURRENT_HEATMAP_RENDER_VERSION,
    _heatmap_overlay,
    _pixel_error_map,
    compute_heatmap_run,
    create_roi,
    create_testing_run,
    get_testing_run_results,
    preview_roi_image,
)


def write_tiff(path: Path, value: int, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def make_db():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def test_heatmap_overlay_uses_bounded_error_dependent_alpha() -> None:
    overlay = _heatmap_overlay(np.array([[0.0, 0.5, 1.0]], dtype=np.float32), vmax=1.0)

    assert overlay[0, 0, 3] == 0
    assert 0 < overlay[0, 1, 3] < overlay[0, 2, 3]
    assert overlay[0, 2, 3] <= 140


def test_heatmap_error_modes_threshold_and_signed_weights() -> None:
    source = np.array([[0.0, 0.2, 0.8]], dtype=np.float32)
    reconstruction = np.array([[0.1, 0.1, 0.5]], dtype=np.float32)

    thresholded = _pixel_error_map(
        source,
        reconstruction,
        HeatmapVisualizationConfig(
            error_mode="absolute",
            threshold_enabled=True,
            threshold=0.15,
        ),
    )
    signed = _pixel_error_map(
        source,
        reconstruction,
        HeatmapVisualizationConfig(
            error_mode="absolute",
            signed_deviations=True,
            positive_weight=2.0,
            negative_weight=3.0,
        ),
    )

    assert np.allclose(thresholded, [[0.0, 0.0, 0.3]])
    assert np.allclose(signed, [[-0.3, 0.2, 0.6]])


def test_heatmap_max_clip_saturates_while_opacity_mode_stays_bounded() -> None:
    error = np.array([[0.0, 0.33, 1.0]], dtype=np.float32)
    clipped = _heatmap_overlay(
        error,
        vmax=1.0,
        config=HeatmapVisualizationConfig(max_clip_enabled=True, max_clip=0.33),
    )
    bounded = _heatmap_overlay(
        error,
        vmax=1.0,
        config=HeatmapVisualizationConfig(max_opacity=0.2),
    )

    assert clipped[0, 0, 3] == 0
    assert clipped[0, 1, 3] == 255
    assert clipped[0, 2, 3] == 255
    assert bounded[0, 2, 3] == 51


def test_fixed_ceiling_does_not_stretch_tiny_errors() -> None:
    error = np.array([[-1.0, -0.0001, 0.0001, 0.5, 1.0, 2.0]], dtype=np.float64)
    config = HeatmapVisualizationConfig(
        fixed_ceiling_enabled=True,
        fixed_ceiling=1.0,
        max_opacity=0.55,
        signed_deviations=True,
    )
    overlay = _heatmap_overlay(error, config=config)

    assert overlay[0, 1, 3] == 0
    assert overlay[0, 2, 3] == 0
    assert overlay[0, 0, 3] == overlay[0, 4, 3]
    assert overlay[0, 3, 3] < overlay[0, 4, 3]
    assert overlay[0, 4, 3] == overlay[0, 5, 3]


def test_fixed_ceiling_uses_selected_error_units() -> None:
    source = np.array([[0.5]], dtype=np.float32)
    reconstruction = np.array([[0.0]], dtype=np.float32)
    absolute_config = HeatmapVisualizationConfig(
        error_mode="absolute", fixed_ceiling_enabled=True, fixed_ceiling=1.0
    )
    squared_config = HeatmapVisualizationConfig(
        error_mode="squared", fixed_ceiling_enabled=True, fixed_ceiling=1.0
    )

    absolute_overlay = _heatmap_overlay(
        _pixel_error_map(source, reconstruction, absolute_config), config=absolute_config
    )
    squared_overlay = _heatmap_overlay(
        _pixel_error_map(source, reconstruction, squared_config), config=squared_config
    )

    assert absolute_overlay[0, 0, 3] > squared_overlay[0, 0, 3]


def test_fixed_ceiling_and_max_clip_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="Fixed ceiling and max clip"):
        HeatmapVisualizationConfig(fixed_ceiling_enabled=True, max_clip_enabled=True)


def seed_finished_mean_image_run(db, tmp_path: Path):
    root = tmp_path / "test_images"
    for index, value in enumerate([100, 110, 120]):
        timestamp = datetime(2026, 4, 1, 12, 0, 0) + timedelta(seconds=index * 10)
        write_tiff(root / f"frame_{timestamp:%Y%m%d_%H%M%S}.tiff", value)

    dataset = models.Dataset(
        name="Test root",
        root_path=str(root),
        status="ready",
        timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})",
        timestamp_format="%Y%m%d_%H%M%S",
    )
    db.add(dataset)
    db.flush()
    folder = models.DatasetFolder(
        dataset_id=dataset.id,
        relative_path=".",
        image_count=3,
        first_timestamp=datetime(2026, 4, 1, 12, 0, 0),
        last_timestamp=datetime(2026, 4, 1, 12, 0, 20),
        extension_summary={".tiff": 3},
        resolution_summary={"8x6": 3},
        image_metadata={"format": "TIFF", "mode": "L", "dtype": "uint8", "channels": 1},
        cadence_summary={"median_seconds": 10},
    )
    db.add(folder)
    db.flush()
    first_timestamp = datetime(2026, 4, 1, 12, 0, 0)
    db.add(
        models.DatasetImage(
            dataset_id=dataset.id,
            folder_id=folder.id,
            file_path=str(root / f"frame_{first_timestamp:%Y%m%d_%H%M%S}.tiff"),
            relative_path=f"frame_{first_timestamp:%Y%m%d_%H%M%S}.tiff",
            file_name=f"frame_{first_timestamp:%Y%m%d_%H%M%S}.tiff",
            extension=".tiff",
            width=8,
            height=6,
            timestamp_raw=f"{first_timestamp:%Y%m%d_%H%M%S}",
            timestamp_parsed=first_timestamp,
        )
    )
    test_set = models.TrainingDataset(name="Test Set", usage_label="test")
    db.add(test_set)
    db.flush()
    db.add(
        models.TrainingDatasetRule(
            training_dataset_id=test_set.id,
            folder_id=folder.id,
            start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
            end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
            stride=1,
        )
    )
    preprocessing = models.PreprocessingPipeline(
        name="Load only",
        graph={"nodes": [{"id": "load", "type": "load_image", "config": {}}], "edges": []},
        input_width=8,
        input_height=6,
        output_width=8,
        output_height=6,
    )
    method = models.MethodConfiguration(
        name="Mean",
        method_type="mean_image",
        method_family="statistical_baseline",
        method_version="1",
        training_mode="fit",
        requires_training=True,
        supports_training_pipeline=False,
        artifact_kind="mean_image",
        builder_kind="form",
        method_graph={},
        method_config={
            "aggregation": "mean",
            "accumulator_dtype": "float32",
            "output_dtype_policy": "source",
            "normalization_mode": "none",
        },
        training_config={},
        inference_config={},
        diagram={},
    )
    db.add_all([preprocessing, method])
    db.flush()
    pipeline = models.TrainingPipeline(
        name="Mean pipeline",
        preprocessing_pipeline_id=preprocessing.id,
        method_configuration_id=method.id,
        training_parameters={},
    )
    db.add(pipeline)
    db.flush()
    db.add(models.TrainingPipelineDataset(training_pipeline_id=pipeline.id, training_dataset_id=test_set.id, position=0))
    artifact_path = tmp_path / "mean.npy"
    np.save(artifact_path, np.full((6, 8), 100, dtype=np.uint8))
    run = models.TrainingRun(
        training_pipeline_id=pipeline.id,
        status="finished",
        artifact_kind="mean_image",
        artifact_path=str(artifact_path),
        artifact_size_bytes=artifact_path.stat().st_size,
        training_pipeline_name="Mean pipeline",
        method_type="mean_image",
        method_family="statistical_baseline",
        training_mode="fit",
        builder_kind="form",
        preprocessing_pipeline_name="Load only",
        dataset_names=["Test Set"],
        dataset_names_text="Test Set",
        shuffle=False,
        training_parameters={},
    )
    db.add(run)
    db.commit()
    return run.id, test_set.id


def test_analysis_image_comparison_supports_input_and_reconstruction_modes(tmp_path: Path) -> None:
    db = make_db()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        testing_run = create_testing_run(
            db,
            TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id),
        )
        assert testing_run.model_training_dataset_names == ["Test Set"]
        assert testing_run.artifact_signature == artifact_signature(tmp_path / "mean.npy")
        assert db.get(models.TrainingRun, training_run_id).artifact_signature == testing_run.artifact_signature
        root = tmp_path / "test_images"
        timestamps = [datetime(2026, 4, 1, 12, 0, index * 10) for index in range(3)]
        rows = []
        for index, timestamp in enumerate(timestamps):
            row = models.TestingRunResult(
                testing_run_id=testing_run.id,
                position=index,
                image_path=str(root / f"frame_{timestamp:%Y%m%d_%H%M%S}.tiff"),
                timestamp=timestamp,
                score=0.0,
                full_mse=0.0,
                width=8,
                height=6,
            )
            db.add(row)
            rows.append(row)
        testing_run.status = "finished"
        db.commit()

        inputs = testing_service.calculate_analysis_image_comparison(
            db,
            AnalysisImageComparisonRequest(
                testing_run_id=testing_run.id,
                reference_result_id=rows[0].id,
                comparison_result_ids=[rows[1].id, rows[2].id],
                image_source="input",
            ),
        )
        assert inputs.image_source == "input"
        assert inputs.shared_max_difference == pytest.approx(20 / 255)
        assert [item.mean_difference for item in inputs.comparisons] == pytest.approx([10 / 255, 20 / 255])
        assert inputs.reference_image_data_url.startswith("data:image/png;base64,")
        assert all(item.heatmap_image_data_url.startswith("data:image/png;base64,") for item in inputs.comparisons)

        reconstructions = testing_service.calculate_analysis_image_comparison(
            db,
            AnalysisImageComparisonRequest(
                testing_run_id=testing_run.id,
                reference_result_id=rows[0].id,
                comparison_result_ids=[rows[1].id],
                image_source="reconstruction",
            ),
        )
        assert reconstructions.shared_max_difference == 0
        assert reconstructions.comparisons[0].mean_difference == 0
    finally:
        db.close()


def test_analysis_image_comparison_rejects_results_from_another_run(tmp_path: Path) -> None:
    db = make_db()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        first = create_testing_run(db, TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id))
        second = models.TestingRun(
            name="Other test",
            training_run_id=training_run_id,
            training_dataset_id=test_set_id,
            status="finished",
            training_run_name="Mean run",
            training_pipeline_name="Mean pipeline",
            training_dataset_name="Test Set",
            preprocessing_pipeline_name="Load only",
            method_type="mean_image",
            method_family="statistical_baseline",
            training_mode="fit",
            artifact_kind="mean_image",
            artifact_path=str(tmp_path / "mean.npy"),
            inference_config={"variant": "other"},
        )
        db.add(second)
        db.flush()
        timestamp = datetime(2026, 4, 1, 12, 0, 0)
        first_row = models.TestingRunResult(testing_run_id=first.id, position=0, image_path="/first.tiff", timestamp=timestamp, score=0, full_mse=0, width=8, height=6)
        second_row = models.TestingRunResult(testing_run_id=second.id, position=0, image_path="/second.tiff", timestamp=timestamp, score=0, full_mse=0, width=8, height=6)
        db.add_all([first_row, second_row])
        first.status = "finished"
        second.status = "finished"
        db.commit()
        with pytest.raises(ValueError, match="was not found"):
            testing_service.calculate_analysis_image_comparison(
                db,
                AnalysisImageComparisonRequest(
                    testing_run_id=first.id,
                    reference_result_id=first_row.id,
                    comparison_result_ids=[second_row.id],
                ),
            )
    finally:
        db.close()


def test_analysis_image_comparison_request_validates_reference_and_comparisons() -> None:
    with pytest.raises(ValueError, match="cannot also be"):
        AnalysisImageComparisonRequest(
            testing_run_id=1,
            reference_result_id=2,
            comparison_result_ids=[2],
        )

    payload = AnalysisImageComparisonRequest(
        testing_run_id=1,
        reference_result_id=2,
        comparison_result_ids=[3, 3, 4],
    )
    assert payload.comparison_result_ids == [3, 4]


def seed_finished_stae_heatmap_run(db, tmp_path: Path):
    training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
    training_run = db.get(models.TrainingRun, training_run_id)
    method = training_run.training_pipeline.method_configuration
    method.method_type = "spatiotemporal_autoencoder"
    method.method_family = "spatiotemporal_reconstruction"
    method.builder_kind = "spatiotemporal_autoencoder"
    method.artifact_kind = "torch_state_dict"
    method.method_config = {
        "input_channels": 1,
        "input_height": 6,
        "input_width": 8,
        "clip_length": 1,
        "future_length": 2,
        "prediction_branch": True,
    }
    training_run.method_type = "spatiotemporal_autoencoder"
    training_run.method_family = "spatiotemporal_reconstruction"
    training_run.builder_kind = "spatiotemporal_autoencoder"
    training_run.artifact_kind = "torch_state_dict"

    root = tmp_path / "test_images"
    timestamps = [datetime(2026, 4, 1, 12, 0, index * 10) for index in range(3)]
    paths = [str(root / f"frame_{timestamp:%Y%m%d_%H%M%S}.tiff") for timestamp in timestamps]
    testing_run = models.TestingRun(
        name="STAE test",
        training_run_id=training_run.id,
        training_dataset_id=test_set_id,
        status="finished",
        image_count=1,
        expected_image_count=1,
        training_run_name="STAE run",
        training_pipeline_name=training_run.training_pipeline_name,
        training_dataset_name="Test Set",
        preprocessing_pipeline_name=training_run.preprocessing_pipeline_name,
        method_type="spatiotemporal_autoencoder",
        method_family="spatiotemporal_reconstruction",
        training_mode="gradient",
        artifact_kind="torch_state_dict",
        artifact_path=training_run.artifact_path,
        inference_config={},
    )
    db.add(testing_run)
    db.flush()
    result = models.TestingRunResult(
        testing_run_id=testing_run.id,
        position=0,
        image_path=paths[0],
        timestamp=timestamps[0],
        score=0.0,
        full_mse=0.0,
        width=8,
        height=6,
        result_metadata={
            "sample_kind": "clip",
            "input_frames": [
                {"path": paths[0], "timestamp": timestamps[0].isoformat(), "file_name": Path(paths[0]).name}
            ],
            "future_frames": [
                {"path": paths[index], "timestamp": timestamps[index].isoformat(), "file_name": Path(paths[index]).name}
                for index in (1, 2)
            ],
        },
    )
    db.add(result)
    db.commit()
    return training_run.id, test_set_id, testing_run.id, result.id


class FakeStaeHeatmapEvaluator:
    clip_shapes: list[tuple[int, ...]] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.mean_image = None

    def reconstruct_batch(self, _images):
        raise AssertionError("STAE heatmaps must not use 4D image reconstruction.")

    def reconstruct_clip_batch(self, clips):
        outputs = []
        for clip in clips:
            self.clip_shapes.append(tuple(clip.shape))
            assert clip.ndim == 4
            outputs.append(
                {
                    "reconstruction": clip.copy(),
                    "prediction": np.zeros((clip.shape[0], 2, clip.shape[2], clip.shape[3]), dtype=clip.dtype),
                }
            )
        return outputs


def test_stae_single_heatmaps_use_clip_reconstruction_and_prediction(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        _, _, testing_run_id, result_id = seed_finished_stae_heatmap_run(db, tmp_path)
        FakeStaeHeatmapEvaluator.clip_shapes = []
        monkeypatch.setattr(testing_service, "ArtifactEvaluator", FakeStaeHeatmapEvaluator)
        monkeypatch.setattr(testing_service, "data_dir", lambda: tmp_path / "artifacts")

        reconstruction = compute_heatmap_run(
            db,
            HeatmapRunCreate(testing_run_id=testing_run_id, testing_result_id=result_id),
        )
        prediction = compute_heatmap_run(
            db,
            HeatmapRunCreate(
                testing_run_id=testing_run_id,
                testing_result_id=result_id,
                stae_view="prediction",
                prediction_horizon=2,
            ),
        )

        assert FakeStaeHeatmapEvaluator.clip_shapes == [(1, 1, 6, 8), (1, 1, 6, 8)]
        assert reconstruction.image_path.endswith("frame_20260401_120000.tiff")
        assert prediction.image_path.endswith("frame_20260401_120020.tiff")
        assert reconstruction.config_signature != prediction.config_signature
    finally:
        db.close()


def test_stae_single_heatmap_rejects_timestamp_without_clip_result(tmp_path: Path) -> None:
    db = make_db()
    try:
        _, _, testing_run_id, _ = seed_finished_stae_heatmap_run(db, tmp_path)
        with pytest.raises(ValueError, match="stored clip result"):
            compute_heatmap_run(
                db,
                HeatmapRunCreate(
                    testing_run_id=testing_run_id,
                    timestamp=datetime(2026, 4, 1, 12, 0, 10),
                ),
            )
    finally:
        db.close()


def test_stae_single_heatmap_rejects_legacy_result_without_clip_metadata(tmp_path: Path) -> None:
    db = make_db()
    try:
        _, _, testing_run_id, result_id = seed_finished_stae_heatmap_run(db, tmp_path)
        result = db.get(models.TestingRunResult, result_id)
        result.result_metadata = {"sample_kind": "image"}
        db.commit()

        with pytest.raises(ValueError, match="Rerun inference"):
            compute_heatmap_run(
                db,
                HeatmapRunCreate(testing_run_id=testing_run_id, testing_result_id=result_id),
            )
    finally:
        db.close()


def test_roi_preview_and_mean_image_testing_run(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)

        preview = preview_roi_image(db, RoiPreviewRequest(training_run_id=training_run_id, training_dataset_id=test_set_id))
        assert preview.width == 8
        assert preview.height == 6
        assert preview.image_data_url.startswith("data:image/png;base64,")

        roi = create_roi(
            db,
            RoiDefinitionCreate(name="Center", image_width=8, image_height=6, x=2, y=1, width=4, height=3),
        )
        testing_run = create_testing_run(
            db,
            TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id, roi_id=roi.id),
        )
        assert testing_run.status == "finished"
        assert testing_run.image_count == 3
        assert testing_run.roi_name == "Center"
        assert testing_run.results_path is not None
        assert Path(testing_run.results_path).exists()
        assert testing_run.score_mean == testing_run.roi_mse_mean

        details = get_testing_run_results(db, testing_run.id)
        assert details is not None
        assert len(details.results) == 3
        assert [result.score for result in details.results] == [0.0, 100.0, 400.0]
        assert db.scalar(select(models.TestingRunResult).where(models.TestingRunResult.testing_run_id == testing_run.id))

        middle_result = details.results[1]
        db.delete(db.get(models.TestingRunResult, middle_result.id))
        db.commit()

        def fail_if_enumerating(*_args, **_kwargs):
            raise AssertionError("Direct heatmap lookup must not enumerate all dataset filenames.")

        monkeypatch.setattr(testing_service, "enumerate_training_dataset_image_records", fail_if_enumerating)
        heatmap = compute_heatmap_run(
            db,
            HeatmapRunCreate(testing_run_id=testing_run.id, timestamp=datetime(2026, 4, 1, 12, 0, 10)),
        )
        assert heatmap.status == "finished"
        assert heatmap.testing_result_id is None
        assert heatmap.image_path.endswith("frame_20260401_120010.tiff")
        assert heatmap.max_error == 100.0
        assert heatmap.render_version == CURRENT_HEATMAP_RENDER_VERSION

        cached = compute_heatmap_run(
            db,
            HeatmapRunCreate(testing_run_id=testing_run.id, timestamp=datetime(2026, 4, 1, 12, 0, 10)),
        )
        assert cached.id == heatmap.id

        recomputed = compute_heatmap_run(
            db,
            HeatmapRunCreate(
                testing_run_id=testing_run.id,
                timestamp=datetime(2026, 4, 1, 12, 0, 10),
                force_recompute=True,
            ),
        )
        assert recomputed.id == heatmap.id
        assert recomputed.render_version == CURRENT_HEATMAP_RENDER_VERSION

        absolute = compute_heatmap_run(
            db,
            HeatmapRunCreate(
                testing_run_id=testing_run.id,
                timestamp=datetime(2026, 4, 1, 12, 0, 10),
                visualization_config=HeatmapVisualizationConfig(error_mode="absolute"),
            ),
        )
        assert absolute.id != heatmap.id
        assert absolute.max_error == 10.0
        assert absolute.config_signature != heatmap.config_signature
    finally:
        db.close()


def test_bulk_enqueue_creates_combinations_skips_duplicates_without_enumerating(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        first_rule = db.scalar(
            select(models.TrainingDatasetRule).where(models.TrainingDatasetRule.training_dataset_id == test_set_id)
        )
        second_set = models.TrainingDataset(name="Second Test Set", usage_label="test")
        db.add(second_set)
        db.flush()
        db.add(
            models.TrainingDatasetRule(
                training_dataset_id=second_set.id,
                folder_id=first_rule.folder_id,
                start_timestamp=first_rule.start_timestamp,
                end_timestamp=first_rule.end_timestamp,
                stride=1,
                selected_images=3,
            )
        )
        db.commit()

        def fail_if_enumerating(*_args, **_kwargs):
            raise AssertionError("Bulk enqueue must not enumerate all dataset filenames.")

        monkeypatch.setattr(testing_service, "enumerate_training_dataset_image_records", fail_if_enumerating)
        payload = TestingRunBulkCreate(
            training_run_ids=[training_run_id],
            training_dataset_ids=[test_set_id, second_set.id],
            inference_config={"error_metric": "mse"},
        )

        first = testing_service.bulk_enqueue_testing_runs(db, payload, wake_scheduler=False)
        assert len(first.created) == 2
        assert first.skipped == []
        assert [run.queue_rank for run in first.created] == [1, 2]
        assert first.created[0].expected_image_count is None
        assert first.created[1].expected_image_count == 3

        second = testing_service.bulk_enqueue_testing_runs(db, payload, wake_scheduler=False)
        assert second.created == []
        assert len(second.skipped) == 2
        assert {item.existing_testing_run_id for item in second.skipped} == {run.id for run in first.created}
    finally:
        db.close()


def test_retrained_artifact_allows_new_inference_generation(tmp_path: Path) -> None:
    db = make_db()
    try:
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        payload = TestingRunCreatePayload(
            training_run_id=training_run_id,
            training_dataset_id=test_set_id,
        )
        first = testing_service.enqueue_testing_run(db, payload, wake_scheduler=False)

        training_run = db.get(models.TrainingRun, training_run_id)
        artifact = Path(training_run.artifact_path)
        np.save(artifact, np.full((6, 8), 101, dtype=np.uint8))
        training_run.artifact_signature = None
        db.commit()

        second = testing_service.enqueue_testing_run(db, payload, wake_scheduler=False)

        assert second.id != first.id
        assert second.artifact_signature != first.artifact_signature
    finally:
        db.close()


def test_bulk_enqueue_rejects_incompatible_dataset_resolution(tmp_path: Path) -> None:
    db = make_db()
    try:
        training_run_id, _ = seed_finished_mean_image_run(db, tmp_path)
        dataset = models.Dataset(
            name="Different root",
            root_path=str(tmp_path / "different"),
            status="ready",
            timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})",
            timestamp_format="%Y%m%d_%H%M%S",
        )
        db.add(dataset)
        db.flush()
        folder = models.DatasetFolder(
            dataset_id=dataset.id,
            relative_path=".",
            image_count=1,
            extension_summary={".tiff": 1},
            resolution_summary={"10x6": 1},
            image_metadata={"format": "TIFF", "mode": "L", "dtype": "uint8", "channels": 1},
        )
        db.add(folder)
        db.flush()
        incompatible = models.TrainingDataset(name="Wrong Size", usage_label="test")
        db.add(incompatible)
        db.flush()
        db.add(
            models.TrainingDatasetRule(
                training_dataset_id=incompatible.id,
                folder_id=folder.id,
                start_timestamp=datetime(2026, 4, 1, 12, 0, 0),
                end_timestamp=datetime(2026, 4, 1, 12, 0, 20),
                stride=1,
                selected_images=1,
            )
        )
        db.commit()

        with pytest.raises(ValueError, match="does not match required input size 8x6"):
            testing_service.bulk_enqueue_testing_runs(
                db,
                TestingRunBulkCreate(training_run_ids=[training_run_id], training_dataset_ids=[incompatible.id]),
                wake_scheduler=False,
            )
    finally:
        db.close()


def test_vae_sample_count_averages_monte_carlo_reconstructions() -> None:
    torch = pytest.importorskip("torch")

    class DummyVae:
        def __init__(self) -> None:
            self.deterministic_vae = True
            self.calls = 0

        def to(self, _device):
            return self

        def __call__(self, x):
            self.calls += 1
            offset = 1.0 if self.deterministic_vae else float(self.calls)
            return x + offset, None

    evaluator = ArtifactEvaluator.__new__(ArtifactEvaluator)
    evaluator.mean_image = None
    evaluator.configuration = SimpleNamespace(
        builder_kind="sequential_variational_autoencoder",
        inference_config={"sample_count": 3},
    )
    evaluator.model = DummyVae()
    evaluator.torch = torch

    reconstruction = evaluator.reconstruct_batch([np.zeros((2, 2), dtype=np.float32)])[0]

    assert evaluator.model.calls == 3
    assert evaluator.model.deterministic_vae is True
    assert np.allclose(reconstruction, 2.0)
