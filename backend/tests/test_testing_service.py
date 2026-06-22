from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base
from app.schemas import (
    HeatmapRunCreate,
    HeatmapVisualizationConfig,
    RoiDefinitionCreate,
    RoiPreviewRequest,
    TestingRunCreate as TestingRunCreatePayload,
)
from app.testing import service as testing_service
from app.testing.service import (
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
