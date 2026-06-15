from __future__ import annotations

import csv
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import data_dir
from app.preprocessing.pipeline import encode_png_data_url, image_metadata, run_pipeline_array
from app.schemas import (
    PreprocessingGraph,
    RoiDefinitionCreate,
    RoiDefinitionRead,
    RoiPreviewRequest,
    RoiPreviewResponse,
    TestingRunCreate,
    TestingRunRead,
    TestingRunResultRead,
    TestingRunResultsResponse,
)
from app.training.data import ResolvedDatasetImage, enumerate_training_dataset_image_records
from app.training.engine import _build_model, _to_nchw


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _testing_run_dir(run_id: int) -> Path:
    path = data_dir() / "testing_runs" / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _roi_geometry(roi: models.RoiDefinition | None) -> dict | None:
    if roi is None:
        return None
    return {
        "image_width": roi.image_width,
        "image_height": roi.image_height,
        "x": roi.x,
        "y": roi.y,
        "width": roi.width,
        "height": roi.height,
    }


def _validate_roi_payload(payload: RoiDefinitionCreate) -> None:
    if payload.x + payload.width > payload.image_width:
        raise ValueError("ROI width extends beyond image width.")
    if payload.y + payload.height > payload.image_height:
        raise ValueError("ROI height extends beyond image height.")


def list_rois(db: Session) -> list[RoiDefinitionRead]:
    rois = db.scalars(select(models.RoiDefinition).order_by(models.RoiDefinition.created_at.desc())).all()
    return [RoiDefinitionRead.model_validate(roi) for roi in rois]


def create_roi(db: Session, payload: RoiDefinitionCreate) -> RoiDefinitionRead:
    _validate_roi_payload(payload)
    roi = models.RoiDefinition(**payload.model_dump())
    db.add(roi)
    db.commit()
    db.refresh(roi)
    return RoiDefinitionRead.model_validate(roi)


def delete_roi(db: Session, roi_id: int) -> bool:
    roi = db.get(models.RoiDefinition, roi_id)
    if roi is None:
        return False
    db.delete(roi)
    db.commit()
    return True


def _load_training_run(db: Session, training_run_id: int) -> models.TrainingRun | None:
    return db.scalar(
        select(models.TrainingRun)
        .where(models.TrainingRun.id == training_run_id)
        .options(
            selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.preprocessing_pipeline),
            selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.method_configuration),
        )
    )


def _load_training_dataset(db: Session, training_dataset_id: int) -> models.TrainingDataset | None:
    return db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset_id)
        .options(
            selectinload(models.TrainingDataset.rules)
            .selectinload(models.TrainingDatasetRule.folder)
            .selectinload(models.DatasetFolder.dataset)
        )
    )


def _first_test_record(db: Session, training_dataset_id: int) -> ResolvedDatasetImage:
    dataset = _load_training_dataset(db, training_dataset_id)
    if dataset is None:
        raise ValueError(f"Train/test dataset does not exist: {training_dataset_id}")
    records = enumerate_training_dataset_image_records(dataset)
    if not records:
        raise ValueError("Train/test dataset produced no images.")
    return records[0]


def preview_roi_image(db: Session, payload: RoiPreviewRequest) -> RoiPreviewResponse:
    training_run = _load_training_run(db, payload.training_run_id)
    if training_run is None:
        raise ValueError(f"Training run does not exist: {payload.training_run_id}")
    preprocessing = training_run.training_pipeline.preprocessing_pipeline
    record = _first_test_record(db, payload.training_dataset_id)
    graph = PreprocessingGraph.model_validate(preprocessing.graph)
    image = run_pipeline_array(graph, record.file_path)
    width, height, channels, dtype, _, _ = image_metadata(image)
    return RoiPreviewResponse(
        training_run_id=payload.training_run_id,
        training_dataset_id=payload.training_dataset_id,
        preprocessing_pipeline_id=preprocessing.id,
        source_image_path=record.file_path,
        source_timestamp=record.timestamp_parsed,
        width=width,
        height=height,
        channels=channels,
        dtype=dtype,
        image_data_url=encode_png_data_url(image),
    )


def _as_image(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
        if array.shape[0] == 1:
            return array[0]
        return np.transpose(array, (1, 2, 0))
    return array


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Cannot score arrays with different shapes: {left.shape} vs {right.shape}.")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.mean(delta * delta))


def _crop_roi(image: np.ndarray, roi: models.RoiDefinition) -> np.ndarray:
    height, width = image.shape[:2]
    if roi.image_width != width or roi.image_height != height:
        raise ValueError(
            f"ROI '{roi.name}' is tuned for {roi.image_width}x{roi.image_height}, "
            f"but testing image is {width}x{height}."
        )
    return image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]


class ArtifactEvaluator:
    """Reusable scorer for one trained artifact.

    Mean-image artifacts are numpy arrays. Gradient artifacts are loaded once
    into a CPU torch module and reused across all test images.
    """

    def __init__(self, training_run: models.TrainingRun) -> None:
        self.training_run = training_run
        self.pipeline = training_run.training_pipeline
        self.configuration = self.pipeline.method_configuration
        self.artifact_path = Path(training_run.artifact_path or "")
        self.mean_image: np.ndarray | None = None
        self.torch = None
        self.model = None
        if not self.artifact_path.exists():
            raise ValueError(f"Training artifact does not exist: {self.artifact_path}")
        if training_run.artifact_kind == "mean_image":
            self.mean_image = np.load(self.artifact_path)

    def _ensure_torch_model(self, image: np.ndarray) -> None:
        if self.model is not None:
            return
        try:
            import torch
        except Exception as exc:  # noqa: BLE001 - report as user-facing validation
            raise ValueError("Torch is required to test gradient-trained methods.") from exc

        input_chw = _to_nchw(image)
        expected = (
            int(self.configuration.method_config["input_channels"]),
            int(self.configuration.method_config["input_height"]),
            int(self.configuration.method_config["input_width"]),
        )
        actual = (int(input_chw.shape[0]), int(input_chw.shape[1]), int(input_chw.shape[2]))
        if actual != expected:
            raise ValueError(
                f"Preprocessing output is {actual[0]}x{actual[1]}x{actual[2]} (CxHxW), "
                f"but the trained method expects {expected[0]}x{expected[1]}x{expected[2]}."
            )

        model, _ = _build_model(torch, self.configuration, deterministic_vae=True)
        model.eval()
        dummy = torch.from_numpy(input_chw[np.newaxis])
        with torch.no_grad():
            model(dummy)
        state = torch.load(self.artifact_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self.torch = torch
        self.model = model

    def reconstruct(self, image: np.ndarray) -> np.ndarray:
        if self.mean_image is not None:
            if image.shape != self.mean_image.shape:
                raise ValueError(f"Mean image shape {self.mean_image.shape} does not match test image shape {image.shape}.")
            return self.mean_image

        self._ensure_torch_model(image)
        input_chw = _to_nchw(image)
        x = self.torch.from_numpy(input_chw[np.newaxis])
        with self.torch.no_grad():
            reconstruction, _ = self.model(x)
        return _as_image(reconstruction.squeeze(0).cpu().numpy())

    def score(self, image: np.ndarray, roi: models.RoiDefinition | None) -> tuple[float, float | None, int, int]:
        reconstruction = self.reconstruct(image)
        source = image
        if self.mean_image is None:
            source = _as_image(_to_nchw(image))
        width, height, _, _, _, _ = image_metadata(source)
        full_mse = _mse(source, reconstruction)
        roi_mse = None
        if roi is not None:
            roi_mse = _mse(_crop_roi(source, roi), _crop_roi(reconstruction, roi))
        return full_mse, roi_mse, width, height


def _snapshot_name(training_run: models.TrainingRun) -> str:
    return training_run.training_pipeline_name or training_run.training_pipeline.name


def _serialize_testing_run(run: models.TestingRun) -> TestingRunRead:
    return TestingRunRead(
        id=run.id,
        name=run.name,
        training_run_id=run.training_run_id,
        training_dataset_id=run.training_dataset_id,
        roi_id=run.roi_id,
        status=run.status,
        started_at=run.started_at,
        ended_at=run.ended_at,
        duration_seconds=run.duration_seconds,
        error_message=run.error_message,
        image_count=run.image_count,
        score_mean=run.score_mean,
        score_min=run.score_min,
        score_max=run.score_max,
        full_mse_mean=run.full_mse_mean,
        roi_mse_mean=run.roi_mse_mean,
        results_path=run.results_path,
        results_size_bytes=run.results_size_bytes,
        training_run_name=run.training_run_name,
        training_pipeline_name=run.training_pipeline_name,
        training_dataset_name=run.training_dataset_name,
        preprocessing_pipeline_name=run.preprocessing_pipeline_name,
        method_type=run.method_type,
        method_family=run.method_family,
        training_mode=run.training_mode,
        artifact_kind=run.artifact_kind,
        artifact_path=run.artifact_path,
        roi_name=run.roi_name,
        roi_geometry=run.roi_geometry,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _serialize_result(result: models.TestingRunResult) -> TestingRunResultRead:
    return TestingRunResultRead(
        id=result.id,
        position=result.position,
        image_path=result.image_path,
        timestamp=result.timestamp,
        score=result.score,
        full_mse=result.full_mse,
        roi_mse=result.roi_mse,
        width=result.width,
        height=result.height,
    )


def list_testing_runs(db: Session) -> list[TestingRunRead]:
    runs = db.scalars(select(models.TestingRun).order_by(models.TestingRun.created_at.desc())).all()
    return [_serialize_testing_run(run) for run in runs]


def get_testing_run(db: Session, run_id: int) -> TestingRunRead | None:
    run = db.get(models.TestingRun, run_id)
    return _serialize_testing_run(run) if run else None


def get_testing_run_results(db: Session, run_id: int) -> TestingRunResultsResponse | None:
    run = db.scalar(
        select(models.TestingRun)
        .where(models.TestingRun.id == run_id)
        .options(selectinload(models.TestingRun.results))
    )
    if run is None:
        return None
    return TestingRunResultsResponse(
        testing_run=_serialize_testing_run(run),
        results=[_serialize_result(result) for result in run.results],
    )


def delete_testing_run(db: Session, run_id: int) -> bool:
    run = db.get(models.TestingRun, run_id)
    if run is None:
        return False
    shutil.rmtree(_testing_run_dir(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return True


def _write_results_csv(testing_run: models.TestingRun, rows: list[models.TestingRunResult]) -> Path:
    path = _testing_run_dir(testing_run.id) / "reconstruction_errors.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "timestamp", "image_path", "score", "full_mse", "roi_mse", "width", "height"])
        for row in rows:
            writer.writerow([
                row.position,
                row.timestamp.isoformat(),
                row.image_path,
                row.score,
                row.full_mse,
                "" if row.roi_mse is None else row.roi_mse,
                row.width,
                row.height,
            ])
    return path


def create_testing_run(db: Session, payload: TestingRunCreate) -> TestingRunRead:
    training_run = _load_training_run(db, payload.training_run_id)
    if training_run is None:
        raise ValueError(f"Training run does not exist: {payload.training_run_id}")
    if training_run.status != "finished":
        raise ValueError("Testing requires a finished training run.")
    if not training_run.artifact_path or not training_run.artifact_kind:
        raise ValueError("Training run has no artifact to test.")

    training_dataset = _load_training_dataset(db, payload.training_dataset_id)
    if training_dataset is None:
        raise ValueError(f"Train/test dataset does not exist: {payload.training_dataset_id}")

    roi = db.get(models.RoiDefinition, payload.roi_id) if payload.roi_id is not None else None
    if payload.roi_id is not None and roi is None:
        raise ValueError(f"ROI does not exist: {payload.roi_id}")

    pipeline = training_run.training_pipeline
    configuration = pipeline.method_configuration
    name = payload.name or f"{training_dataset.name} on {_snapshot_name(training_run)}"
    started = time.perf_counter()
    now = _utcnow()
    testing_run = models.TestingRun(
        name=name,
        training_run_id=training_run.id,
        training_dataset_id=training_dataset.id,
        roi_id=roi.id if roi else None,
        status="running",
        started_at=now,
        training_run_name=_snapshot_name(training_run),
        training_pipeline_name=pipeline.name,
        training_dataset_name=training_dataset.name,
        preprocessing_pipeline_name=pipeline.preprocessing_pipeline.name,
        method_type=configuration.method_type,
        method_family=configuration.method_family,
        training_mode=configuration.training_mode,
        artifact_kind=training_run.artifact_kind,
        artifact_path=training_run.artifact_path,
        roi_name=roi.name if roi else None,
        roi_geometry=_roi_geometry(roi),
    )
    db.add(testing_run)
    db.commit()
    db.refresh(testing_run)

    try:
        graph = PreprocessingGraph.model_validate(pipeline.preprocessing_pipeline.graph)
        records = enumerate_training_dataset_image_records(training_dataset)
        if not records:
            raise ValueError("Train/test dataset produced no images.")
        evaluator = ArtifactEvaluator(training_run)
        result_rows: list[models.TestingRunResult] = []
        scores: list[float] = []
        full_scores: list[float] = []
        roi_scores: list[float] = []

        for position, record in enumerate(records):
            image = run_pipeline_array(graph, record.file_path)
            full_mse, roi_mse, width, height = evaluator.score(image, roi)
            score = roi_mse if roi_mse is not None else full_mse
            scores.append(score)
            full_scores.append(full_mse)
            if roi_mse is not None:
                roi_scores.append(roi_mse)
            row = models.TestingRunResult(
                testing_run_id=testing_run.id,
                position=position,
                image_path=record.file_path,
                timestamp=record.timestamp_parsed,
                score=score,
                full_mse=full_mse,
                roi_mse=roi_mse,
                width=width,
                height=height,
            )
            db.add(row)
            result_rows.append(row)

        db.flush()
        results_path = _write_results_csv(testing_run, result_rows)
        testing_run.status = "finished"
        testing_run.ended_at = _utcnow()
        testing_run.duration_seconds = round(time.perf_counter() - started, 3)
        testing_run.image_count = len(result_rows)
        testing_run.score_mean = float(np.mean(scores))
        testing_run.score_min = float(np.min(scores))
        testing_run.score_max = float(np.max(scores))
        testing_run.full_mse_mean = float(np.mean(full_scores))
        testing_run.roi_mse_mean = float(np.mean(roi_scores)) if roi_scores else None
        testing_run.results_path = str(results_path)
        testing_run.results_size_bytes = results_path.stat().st_size
    except Exception as exc:
        testing_run.status = "failed"
        testing_run.ended_at = _utcnow()
        testing_run.duration_seconds = round(time.perf_counter() - started, 3)
        testing_run.error_message = str(exc)
    db.commit()
    db.refresh(testing_run)
    return _serialize_testing_run(testing_run)


def clear_testing_rows_for_training_run(db: Session, training_run_id: int) -> None:
    testing_ids = list(db.scalars(select(models.TestingRun.id).where(models.TestingRun.training_run_id == training_run_id)))
    if not testing_ids:
        return
    db.execute(delete(models.TestingRunResult).where(models.TestingRunResult.testing_run_id.in_(testing_ids)))
    db.execute(delete(models.TestingRun).where(models.TestingRun.id.in_(testing_ids)))
