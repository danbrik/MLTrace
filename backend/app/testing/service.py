from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.database import data_dir
from app.preprocessing.pipeline import (
    encode_absolute_image_data_url,
    encode_png_data_url,
    image_metadata,
    run_pipeline_array,
)
from app.scanner import filename_timestamp_template
from app.schemas import (
    HeatmapRunCreate,
    HeatmapRunRead,
    PreprocessingGraph,
    RoiDefinitionCreate,
    RoiDefinitionRead,
    RoiPreviewRequest,
    RoiPreviewResponse,
    TestingRunCreate,
    TestingRunResultImageResponse,
    TestingRunRead,
    TestingRunResultRead,
    TestingRunResultsResponse,
)
from app.training.data import ResolvedDatasetImage, enumerate_training_dataset_image_records
from app.training.engine import _build_model, _to_nchw
from app.training.scheduler import scheduler


CURRENT_HEATMAP_RENDER_VERSION = 2


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _testing_run_dir(run_id: int) -> Path:
    path = data_dir() / "testing_runs" / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _roi_geometry(roi: models.RoiDefinition | None) -> dict | None:
    if roi is None:
        return None
    points = _roi_points(roi)
    return {
        "image_width": roi.image_width,
        "image_height": roi.image_height,
        "x": roi.x,
        "y": roi.y,
        "width": roi.width,
        "height": roi.height,
        "geometry_type": roi.geometry_type,
        "points": points,
        "tile_rows": roi.tile_rows,
        "tile_cols": roi.tile_cols,
    }


def _rectangle_points(x: int, y: int, width: int, height: int) -> list[dict]:
    return [
        {"x": float(x), "y": float(y)},
        {"x": float(x + width), "y": float(y)},
        {"x": float(x + width), "y": float(y + height)},
        {"x": float(x), "y": float(y + height)},
    ]


def _payload_points(payload: RoiDefinitionCreate) -> list[dict]:
    if payload.points:
        return [{"x": float(point.x), "y": float(point.y)} for point in payload.points]
    return _rectangle_points(payload.x, payload.y, payload.width, payload.height)


def _roi_points(roi: models.RoiDefinition) -> list[dict]:
    if roi.points and len(roi.points) == 4:
        return [{"x": float(point["x"]), "y": float(point["y"])} for point in roi.points]
    return _rectangle_points(roi.x, roi.y, roi.width, roi.height)


def _bounding_box(points: list[dict], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    min_x = max(0, min(point["x"] for point in points))
    max_x = min(image_width, max(point["x"] for point in points))
    min_y = max(0, min(point["y"] for point in points))
    max_y = min(image_height, max(point["y"] for point in points))
    x = int(np.floor(min_x))
    y = int(np.floor(min_y))
    width = max(1, int(np.ceil(max_x)) - x)
    height = max(1, int(np.ceil(max_y)) - y)
    return x, y, width, height


def _validate_roi_payload(payload: RoiDefinitionCreate) -> None:
    points = _payload_points(payload)
    if len(points) != 4:
        raise ValueError("ROI must define exactly four corner points.")
    for index, point in enumerate(points, start=1):
        if point["x"] < 0 or point["x"] > payload.image_width or point["y"] < 0 or point["y"] > payload.image_height:
            raise ValueError(f"ROI point {index} is outside the image bounds.")
    if payload.x + payload.width > payload.image_width:
        raise ValueError("ROI width extends beyond image width.")
    if payload.y + payload.height > payload.image_height:
        raise ValueError("ROI height extends beyond image height.")


def list_rois(db: Session) -> list[RoiDefinitionRead]:
    rois = db.scalars(select(models.RoiDefinition).order_by(models.RoiDefinition.created_at.desc())).all()
    return [RoiDefinitionRead.model_validate(roi) for roi in rois]


def create_roi(db: Session, payload: RoiDefinitionCreate) -> RoiDefinitionRead:
    _validate_roi_payload(payload)
    data = payload.model_dump()
    points = _payload_points(payload)
    x, y, width, height = _bounding_box(points, payload.image_width, payload.image_height)
    data.update(
        {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "geometry_type": payload.geometry_type or "polygon",
            "points": points,
        }
    )
    roi = models.RoiDefinition(**data)
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
        image_data_url=encode_absolute_image_data_url(image),
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


def _mse_masked(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Cannot score arrays with different shapes: {left.shape} vs {right.shape}.")
    if mask.shape != left.shape[:2]:
        raise ValueError(f"ROI mask shape {mask.shape} does not match image shape {left.shape[:2]}.")
    delta = left.astype(np.float64) - right.astype(np.float64)
    values = delta[mask]
    if values.size == 0:
        raise ValueError("ROI mask contains no pixels.")
    return float(np.mean(values * values))


def _validate_roi_size(image: np.ndarray, roi: models.RoiDefinition) -> tuple[int, int]:
    height, width = image.shape[:2]
    if roi.image_width != width or roi.image_height != height:
        raise ValueError(
            f"ROI '{roi.name}' is tuned for {roi.image_width}x{roi.image_height}, "
            f"but testing image is {width}x{height}."
        )
    return width, height


def _polygon_mask(width: int, height: int, points: list[dict]) -> np.ndarray:
    mask_image = Image.new("L", (width, height), 0)
    polygon = [(float(point["x"]), float(point["y"])) for point in points]
    ImageDraw.Draw(mask_image).polygon(polygon, fill=1)
    return np.asarray(mask_image, dtype=bool)


def _interpolate_quad(points: list[dict], u: float, v: float) -> dict:
    # Points are ordered top-left, top-right, bottom-right, bottom-left. This
    # bilinear interpolation keeps tile cells aligned with skewed quadrilaterals.
    tl, tr, br, bl = points
    top_x = tl["x"] + (tr["x"] - tl["x"]) * u
    top_y = tl["y"] + (tr["y"] - tl["y"]) * u
    bottom_x = bl["x"] + (br["x"] - bl["x"]) * u
    bottom_y = bl["y"] + (br["y"] - bl["y"]) * u
    return {
        "x": top_x + (bottom_x - top_x) * v,
        "y": top_y + (bottom_y - top_y) * v,
    }


def _tile_polygons(points: list[dict], rows: int, cols: int) -> list[dict]:
    tiles: list[dict] = []
    for row in range(rows):
        for col in range(cols):
            u0 = col / cols
            u1 = (col + 1) / cols
            v0 = row / rows
            v1 = (row + 1) / rows
            tiles.append(
                {
                    "row": row + 1,
                    "col": col + 1,
                    "points": [
                        _interpolate_quad(points, u0, v0),
                        _interpolate_quad(points, u1, v0),
                        _interpolate_quad(points, u1, v1),
                        _interpolate_quad(points, u0, v1),
                    ],
                }
            )
    return tiles


def _roi_scores(
    source: np.ndarray, reconstruction: np.ndarray, roi: models.RoiDefinition
) -> tuple[float, list[dict]]:
    width, height = _validate_roi_size(source, roi)
    points = _roi_points(roi)
    roi_mask = _polygon_mask(width, height, points)
    roi_mse = _mse_masked(source, reconstruction, roi_mask)
    tile_scores: list[dict] = []
    rows = max(1, int(roi.tile_rows or 1))
    cols = max(1, int(roi.tile_cols or 1))
    for tile in _tile_polygons(points, rows, cols):
        mask = _polygon_mask(width, height, tile["points"])
        pixel_count = int(mask.sum())
        tile_scores.append(
            {
                "row": tile["row"],
                "col": tile["col"],
                "mse": None if pixel_count == 0 else _mse_masked(source, reconstruction, mask),
                "pixels": pixel_count,
            }
        )
    return roi_mse, tile_scores


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

    def reconstruct_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Reconstruct several images in one forward pass. Uses CUDA when
        available (the scheduler pins one GPU via CUDA_VISIBLE_DEVICES); falls
        back to CPU. The model is moved to the chosen device once and reused."""
        if not images:
            return []
        if self.mean_image is not None:
            for image in images:
                if image.shape != self.mean_image.shape:
                    raise ValueError(
                        f"Mean image shape {self.mean_image.shape} does not match test image shape {image.shape}."
                    )
            return [self.mean_image for _ in images]

        self._ensure_torch_model(images[0])
        device = self._batch_device()
        batch = np.stack([_to_nchw(image) for image in images], axis=0)
        x = self.torch.from_numpy(batch).to(device)
        with self.torch.no_grad():
            reconstruction, _ = self.model(x)
        arr = reconstruction.cpu().numpy()
        return [_as_image(arr[index]) for index in range(arr.shape[0])]

    def _batch_device(self):
        cached = getattr(self, "_device", None)
        if cached is not None:
            return cached
        device = "cpu"
        try:
            if self.torch.cuda.is_available():
                device = "cuda"
                self.model.to(device)
        except Exception:  # noqa: BLE001 - GPU optional; CPU fallback is always valid
            device = "cpu"
        self._device = device
        return device

    def score(
        self, image: np.ndarray, roi: models.RoiDefinition | None
    ) -> tuple[float, float | None, int, int, list[dict] | None]:
        reconstruction = self.reconstruct(image)
        return self._score_pair(image, reconstruction, roi)

    def score_batch(
        self, images: list[np.ndarray], roi: models.RoiDefinition | None
    ) -> list[tuple[float, float | None, int, int, list[dict] | None]]:
        """Score several images with one (GPU-)batched reconstruction pass."""
        reconstructions = self.reconstruct_batch(images)
        return [self._score_pair(image, rec, roi) for image, rec in zip(images, reconstructions)]

    def _score_pair(
        self, image: np.ndarray, reconstruction: np.ndarray, roi: models.RoiDefinition | None
    ) -> tuple[float, float | None, int, int, list[dict] | None]:
        source = image if self.mean_image is not None else _as_image(_to_nchw(image))
        width, height, _, _, _, _ = image_metadata(source)
        full_mse = _mse(source, reconstruction)
        roi_mse = None
        tile_scores = None
        if roi is not None:
            roi_mse, tile_scores = _roi_scores(source, reconstruction, roi)
        return full_mse, roi_mse, width, height, tile_scores


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
        enqueued_at=run.enqueued_at,
        started_at=run.started_at,
        ended_at=run.ended_at,
        duration_seconds=run.duration_seconds,
        gpu_index=run.gpu_index,
        device=run.device,
        error_message=run.error_message,
        image_count=run.image_count,
        expected_image_count=run.expected_image_count,
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
        tile_scores=result.tile_scores,
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


def get_testing_run_result_image(
    db: Session, run_id: int, result_id: int
) -> TestingRunResultImageResponse | None:
    """Return the preprocessed source image used for one testing result.

    Testing results store local filesystem paths. The browser cannot display
    those paths directly, so Analysis asks the backend to re-run the saved
    preprocessing graph for the specific result row and return a PNG data URL.
    """

    result = db.scalar(
        select(models.TestingRunResult)
        .where(
            models.TestingRunResult.id == result_id,
            models.TestingRunResult.testing_run_id == run_id,
        )
        .options(
            selectinload(models.TestingRunResult.testing_run)
            .selectinload(models.TestingRun.training_run)
            .selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.preprocessing_pipeline)
        )
    )
    if result is None:
        return None

    preprocessing = result.testing_run.training_run.training_pipeline.preprocessing_pipeline
    graph = PreprocessingGraph.model_validate(preprocessing.graph)
    image = run_pipeline_array(graph, result.image_path)
    width, height, channels, dtype, _, _ = image_metadata(image)
    return TestingRunResultImageResponse(
        testing_run_id=run_id,
        result_id=result_id,
        image_path=result.image_path,
        timestamp=result.timestamp,
        width=width,
        height=height,
        channels=channels,
        dtype=dtype,
        image_data_url=encode_absolute_image_data_url(image),
    )


def _load_testing_result_for_heatmap(
    db: Session, run_id: int, result_id: int
) -> models.TestingRunResult | None:
    return db.scalar(
        select(models.TestingRunResult)
        .where(
            models.TestingRunResult.id == result_id,
            models.TestingRunResult.testing_run_id == run_id,
        )
        .options(
            selectinload(models.TestingRunResult.testing_run)
            .selectinload(models.TestingRun.training_run)
            .selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.preprocessing_pipeline),
            selectinload(models.TestingRunResult.testing_run)
            .selectinload(models.TestingRun.training_run)
            .selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.method_configuration),
        )
    )


def _load_testing_run_for_heatmap(db: Session, run_id: int) -> models.TestingRun | None:
    return db.scalar(
        select(models.TestingRun)
        .where(models.TestingRun.id == run_id)
        .options(
            selectinload(models.TestingRun.training_run)
            .selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.preprocessing_pipeline),
            selectinload(models.TestingRun.training_run)
            .selectinload(models.TrainingRun.training_pipeline)
            .selectinload(models.TrainingPipeline.method_configuration),
        )
    )


def _folder_filename_template(db: Session, folder: models.DatasetFolder) -> dict:
    if folder.filename_template:
        return folder.filename_template

    image = db.scalar(
        select(models.DatasetImage)
        .where(models.DatasetImage.folder_id == folder.id)
        .order_by(models.DatasetImage.timestamp_parsed.asc(), models.DatasetImage.id.asc())
    )
    if image is None:
        raise ValueError(
            f"Dataset folder '{folder.relative_path}' has no filename template. Rescan the source dataset before computing direct heatmaps."
        )
    dataset = folder.dataset
    if not dataset.timestamp_regex or not dataset.timestamp_format:
        raise ValueError(f"Dataset '{dataset.name}' has no confirmed timestamp parser.")

    template = filename_timestamp_template(image.file_name, dataset.timestamp_regex, dataset.timestamp_format)
    folder.filename_template = template
    db.flush()
    return template


def _direct_heatmap_image_path(db: Session, testing_run: models.TestingRun, timestamp: datetime) -> str:
    training_dataset = _load_training_dataset(db, testing_run.training_dataset_id)
    if training_dataset is None:
        raise ValueError(f"Train/test dataset does not exist: {testing_run.training_dataset_id}")

    attempted_filenames: list[str] = []
    valid_rules = [
        rule
        for rule in training_dataset.rules
        if rule.folder is not None and rule.folder.dataset is not None
    ]
    for rule in sorted(
        valid_rules,
        key=lambda item: (
            item.start_timestamp,
            item.end_timestamp,
            item.folder.dataset.name,
            item.folder.relative_path,
            item.id,
        ),
    ):
        if timestamp < rule.start_timestamp or timestamp > rule.end_timestamp:
            continue
        dataset = rule.folder.dataset
        if not dataset.timestamp_format:
            raise ValueError(f"Dataset '{dataset.name}' has no confirmed timestamp format.")
        template = _folder_filename_template(db, rule.folder)
        timestamp_raw = timestamp.strftime(template.get("timestamp_format") or dataset.timestamp_format)
        filename = f"{template.get('prefix', '')}{timestamp_raw}{template.get('suffix', '')}"
        attempted_filenames.append(filename)
        folder_path = Path(dataset.root_path).expanduser()
        if rule.folder.relative_path != ".":
            folder_path = folder_path / rule.folder.relative_path
        image_path = folder_path / filename
        if image_path.exists() and image_path.is_file():
            return str(image_path)

    if attempted_filenames:
        raise ValueError(f"Datei mit Filename {attempted_filenames[0]} existiert nicht.")
    raise ValueError(f"Timestamp {timestamp.isoformat()} is outside the selected inference dataset ranges.")


def _pixel_error_map(source: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    if source.shape != reconstruction.shape:
        raise ValueError(f"Cannot build heatmap for different shapes: {source.shape} vs {reconstruction.shape}.")
    delta = source.astype(np.float64) - reconstruction.astype(np.float64)
    squared = delta * delta
    if squared.ndim == 3:
        return np.mean(squared, axis=2)
    return squared


def _heatmap_overlay(error_map: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Render a transparent Jet-style error overlay. ``vmax`` fixes the
    normalization ceiling (shared scale across a video); when None each frame is
    normalized to its own maximum."""
    ceiling = float(vmax) if vmax is not None else (float(np.max(error_map)) if error_map.size else 0.0)
    if ceiling <= 0:
        normalized = np.zeros_like(error_map, dtype=np.float64)
    else:
        normalized = np.clip(error_map / ceiling, 0.0, 1.0)

    # Jet-style transparent error map: low error stays blue and subtle, high error
    # becomes yellow/red and more opaque for anomaly-style inspection overlays.
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    # Zero error is invisible. Maximum error remains translucent so the source
    # image is still inspectable beneath the anomaly colors.
    alpha = np.clip(normalized * 140.0, 0.0, 140.0)

    overlay = np.zeros((*error_map.shape, 4), dtype=np.uint8)
    overlay[..., 0] = np.asarray(red * 255.0, dtype=np.uint8)
    overlay[..., 1] = np.asarray(green * 255.0, dtype=np.uint8)
    overlay[..., 2] = np.asarray(blue * 255.0, dtype=np.uint8)
    overlay[..., 3] = np.asarray(alpha, dtype=np.uint8)
    return overlay


def _error_matrix(error_map: np.ndarray) -> list[list[float]]:
    """Per-pixel error grid for the frontend Plotly heatmap (z-matrix), at the
    image's native resolution (pixel-exact, no downsampling). Rounded to bound
    the JSON payload while preserving color/structure."""
    grid = error_map.astype(np.float64)
    if grid.ndim != 2:
        grid = grid.reshape(grid.shape[0], -1)
    return np.round(grid, 6).tolist()


def compute_heatmap_run(db: Session, payload: HeatmapRunCreate) -> HeatmapRunRead:
    """Compute or return a cached CPU pixel-level reconstruction-error heatmap."""
    result_id = payload.testing_result_id
    timestamp = payload.timestamp
    direct_image_path: str | None = None

    if timestamp is not None:
        existing_by_timestamp = db.scalar(
            select(models.HeatmapRun).where(
                models.HeatmapRun.testing_run_id == payload.testing_run_id,
                models.HeatmapRun.timestamp == timestamp,
                models.HeatmapRun.status == "finished",
            )
        )
        if (
            existing_by_timestamp is not None
            and existing_by_timestamp.render_version == CURRENT_HEATMAP_RENDER_VERSION
            and not payload.force_recompute
        ):
            return HeatmapRunRead.model_validate(existing_by_timestamp)

    if result_id is None and timestamp is not None:
        result = db.scalar(
            select(models.TestingRunResult)
            .where(
                models.TestingRunResult.testing_run_id == payload.testing_run_id,
                models.TestingRunResult.timestamp == timestamp,
            )
            .options(
                selectinload(models.TestingRunResult.testing_run)
                .selectinload(models.TestingRun.training_run)
                .selectinload(models.TrainingRun.training_pipeline)
                .selectinload(models.TrainingPipeline.preprocessing_pipeline),
                selectinload(models.TestingRunResult.testing_run)
                .selectinload(models.TestingRun.training_run)
                .selectinload(models.TrainingRun.training_pipeline)
                .selectinload(models.TrainingPipeline.method_configuration),
            )
        )
        if result is None:
            testing_run = _load_testing_run_for_heatmap(db, payload.testing_run_id)
            if testing_run is None:
                raise ValueError(f"Testing run does not exist: {payload.testing_run_id}")
            direct_image_path = _direct_heatmap_image_path(db, testing_run, timestamp)
        else:
            result_id = result.id
    elif result_id is None:
        raise ValueError("Either testing_result_id or timestamp is required.")

    if result_id is not None:
        existing = db.scalar(
            select(models.HeatmapRun).where(
                models.HeatmapRun.testing_run_id == payload.testing_run_id,
                models.HeatmapRun.testing_result_id == result_id,
                models.HeatmapRun.status == "finished",
            )
        )
        if (
            existing is not None
            and existing.render_version == CURRENT_HEATMAP_RENDER_VERSION
            and not payload.force_recompute
        ):
            return HeatmapRunRead.model_validate(existing)

    if result_id is not None:
        result = _load_testing_result_for_heatmap(db, payload.testing_run_id, result_id)
        if result is None:
            raise ValueError("Testing result does not exist.")
        testing_run = result.testing_run
        training_run = testing_run.training_run
        image_path = result.image_path
        timestamp = result.timestamp
        pending_width = result.width
        pending_height = result.height
    else:
        result = None
        testing_run = _load_testing_run_for_heatmap(db, payload.testing_run_id)
        if testing_run is None:
            raise ValueError(f"Testing run does not exist: {payload.testing_run_id}")
        training_run = testing_run.training_run
        image_path = direct_image_path
        pending_width = 1
        pending_height = 1

    if testing_run.status != "finished":
        raise ValueError("Heatmaps can only be computed for finished testing runs.")
    if training_run is None:
        raise ValueError("Training run no longer exists.")
    if image_path is None or timestamp is None:
        raise ValueError("Heatmap source image could not be resolved.")

    row_query = select(models.HeatmapRun).where(
        models.HeatmapRun.testing_run_id == payload.testing_run_id,
        models.HeatmapRun.timestamp == timestamp,
    )
    row_query = (
        row_query.where(models.HeatmapRun.testing_result_id == result_id)
        if result_id is not None
        else row_query.where(models.HeatmapRun.testing_result_id.is_(None))
    )
    row = db.scalar(row_query)
    if row is None:
        row = models.HeatmapRun(
            testing_run_id=payload.testing_run_id,
            testing_result_id=result_id,
        )
        db.add(row)
    row.image_path = image_path
    row.timestamp = timestamp
    row.status = "running"
    row.error_message = None
    row.width = pending_width
    row.height = pending_height
    row.channels = 1
    row.dtype = "pending"
    row.max_error = 0.0
    row.mean_error = 0.0
    row.max_x = 0
    row.max_y = 0
    row.source_image_data_url = ""
    row.reconstruction_image_data_url = ""
    row.heatmap_image_data_url = ""
    row.error_matrix = None
    row.render_version = CURRENT_HEATMAP_RENDER_VERSION
    row.updated_at = _utcnow()
    db.commit()

    try:
        preprocessing = training_run.training_pipeline.preprocessing_pipeline
        image = run_pipeline_array(PreprocessingGraph.model_validate(preprocessing.graph), image_path)
        evaluator = ArtifactEvaluator(training_run)
        reconstruction = evaluator.reconstruct(image)
        source = image if evaluator.mean_image is not None else _as_image(_to_nchw(image))
        error_map = _pixel_error_map(source, reconstruction)
        max_y, max_x = np.unravel_index(int(np.argmax(error_map)), error_map.shape)
        width, height, channels, dtype, _, _ = image_metadata(source)

        row.status = "finished"
        row.error_message = None
        row.width = width
        row.height = height
        row.channels = channels
        row.dtype = dtype
        row.max_error = float(np.max(error_map))
        row.mean_error = float(np.mean(error_map))
        row.max_x = int(max_x)
        row.max_y = int(max_y)
        row.source_image_data_url = encode_absolute_image_data_url(source)
        row.reconstruction_image_data_url = encode_absolute_image_data_url(reconstruction)
        row.heatmap_image_data_url = encode_png_data_url(_heatmap_overlay(error_map))
        row.error_matrix = _error_matrix(error_map)
        row.updated_at = _utcnow()
        db.commit()
        db.refresh(row)
        return HeatmapRunRead.model_validate(row)
    except Exception as exc:
        db.rollback()
        row = db.scalar(
            select(models.HeatmapRun).where(
                models.HeatmapRun.testing_run_id == payload.testing_run_id,
                models.HeatmapRun.timestamp == timestamp,
            )
        )
        if row is not None:
            row.status = "failed"
            row.error_message = str(exc)
            row.updated_at = _utcnow()
            db.commit()
        raise


def list_heatmap_runs(db: Session) -> list[HeatmapRunRead]:
    rows = db.scalars(select(models.HeatmapRun).order_by(models.HeatmapRun.created_at.desc())).all()
    return [HeatmapRunRead.model_validate(row) for row in rows]


def clear_heatmap_runs(db: Session) -> int:
    count = len(db.scalars(select(models.HeatmapRun.id)).all())
    db.execute(delete(models.HeatmapRun))
    db.commit()
    return count


def delete_testing_run(db: Session, run_id: int) -> bool:
    run = db.get(models.TestingRun, run_id)
    if run is None:
        return False
    if run.status == "running":
        raise ValueError("Abort the testing run before removing it.")
    shutil.rmtree(_testing_run_dir(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return True


def _write_results_csv(testing_run: models.TestingRun, rows: list[models.TestingRunResult]) -> Path:
    path = _testing_run_dir(testing_run.id) / "reconstruction_errors.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "position",
            "timestamp",
            "image_path",
            "score",
            "full_mse",
            "roi_mse",
            "tile_scores_json",
            "width",
            "height",
        ])
        for row in rows:
            writer.writerow([
                row.position,
                row.timestamp.isoformat(),
                row.image_path,
                row.score,
                row.full_mse,
                "" if row.roi_mse is None else row.roi_mse,
                "" if row.tile_scores is None else json.dumps(row.tile_scores, sort_keys=True),
                row.width,
                row.height,
            ])
    return path


class TestingConflict(Exception):
    """Raised when an identical testing configuration already exists (HTTP 409)."""

    def __init__(self, existing: models.TestingRun) -> None:
        self.existing = existing
        super().__init__(
            f"A testing run for this model + dataset + ROI already exists as '{existing.name}'."
        )


def _find_duplicate_testing_run(
    db: Session, training_run_id: int, training_dataset_id: int, roi_id: int | None
) -> models.TestingRun | None:
    query = select(models.TestingRun).where(
        models.TestingRun.training_run_id == training_run_id,
        models.TestingRun.training_dataset_id == training_dataset_id,
    )
    query = query.where(
        models.TestingRun.roi_id.is_(None) if roi_id is None else models.TestingRun.roi_id == roi_id
    )
    return db.scalar(query)


def _reset_testing_run_for_queue(run: models.TestingRun) -> None:
    run.status = "queued"
    run.enqueued_at = _utcnow()
    run.started_at = None
    run.ended_at = None
    run.duration_seconds = None
    run.gpu_index = None
    run.device = None
    run.pid = None
    run.log_path = None
    run.error_message = None
    run.image_count = None
    run.score_mean = None
    run.score_min = None
    run.score_max = None
    run.full_mse_mean = None
    run.roi_mse_mean = None
    run.results_path = None
    run.results_size_bytes = None


def enqueue_testing_run(db: Session, payload: TestingRunCreate, *, wake_scheduler: bool = True) -> TestingRunRead:
    """Validate refs, dedup, and queue a testing run for the scheduler."""
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
    expected_image_count = len(enumerate_training_dataset_image_records(training_dataset))

    roi = db.get(models.RoiDefinition, payload.roi_id) if payload.roi_id is not None else None
    if payload.roi_id is not None and roi is None:
        raise ValueError(f"ROI does not exist: {payload.roi_id}")

    existing = _find_duplicate_testing_run(db, training_run.id, training_dataset.id, roi.id if roi else None)
    if existing is not None:
        raise TestingConflict(existing)

    pipeline = training_run.training_pipeline
    configuration = pipeline.method_configuration
    name = payload.name or f"{training_dataset.name} on {_snapshot_name(training_run)}"
    testing_run = models.TestingRun(
        name=name,
        training_run_id=training_run.id,
        training_dataset_id=training_dataset.id,
        roi_id=roi.id if roi else None,
        status="queued",
        enqueued_at=_utcnow(),
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
        expected_image_count=expected_image_count,
    )
    db.add(testing_run)
    db.commit()
    db.refresh(testing_run)
    if wake_scheduler:
        scheduler.wake()
    return _serialize_testing_run(testing_run)


def create_testing_run(db: Session, payload: TestingRunCreate) -> TestingRunRead:
    """Synchronous testing helper retained for unit tests.

    The public API queues testing runs via ``enqueue_testing_run``. Tests and
    low-level service callers sometimes need deterministic in-process execution
    against their own session, so this helper runs the same scoring path without
    spawning the scheduler worker.
    """

    queued = enqueue_testing_run(db, payload, wake_scheduler=False)
    run = db.get(models.TestingRun, queued.id)
    if run is None:
        raise ValueError("Queued testing run disappeared before execution.")

    started = time.perf_counter()
    run.status = "running"
    run.started_at = _utcnow()
    run.device = "CPU"
    db.commit()

    training_run = _load_training_run(db, run.training_run_id)
    training_dataset = _load_training_dataset(db, run.training_dataset_id)
    if training_run is None or training_dataset is None:
        raise ValueError("Training run or train/test dataset no longer exists.")
    roi = db.get(models.RoiDefinition, run.roi_id) if run.roi_id is not None else None
    graph = PreprocessingGraph.model_validate(training_run.training_pipeline.preprocessing_pipeline.graph)
    records = enumerate_training_dataset_image_records(training_dataset)
    run.expected_image_count = len(records)
    evaluator = ArtifactEvaluator(training_run)
    rows: list[models.TestingRunResult] = []
    scores: list[float] = []
    full_scores: list[float] = []
    roi_scores: list[float] = []
    for position, record in enumerate(records):
        image = run_pipeline_array(graph, record.file_path)
        full_mse, roi_mse, width, height, tile_scores = evaluator.score(image, roi)
        score = roi_mse if roi_mse is not None else full_mse
        row = models.TestingRunResult(
            testing_run_id=run.id,
            position=position,
            image_path=record.file_path,
            timestamp=record.timestamp_parsed,
            score=score,
            full_mse=full_mse,
            roi_mse=roi_mse,
            tile_scores=tile_scores,
            width=width,
            height=height,
        )
        db.add(row)
        rows.append(row)
        scores.append(score)
        full_scores.append(full_mse)
        if roi_mse is not None:
            roi_scores.append(roi_mse)

    db.flush()
    results_path = _write_results_csv(run, rows)
    run.status = "finished"
    run.ended_at = _utcnow()
    run.duration_seconds = round(time.perf_counter() - started, 3)
    run.image_count = len(rows)
    run.score_mean = float(np.mean(scores)) if scores else None
    run.score_min = float(np.min(scores)) if scores else None
    run.score_max = float(np.max(scores)) if scores else None
    run.full_mse_mean = float(np.mean(full_scores)) if full_scores else None
    run.roi_mse_mean = float(np.mean(roi_scores)) if roi_scores else None
    run.results_path = str(results_path)
    run.results_size_bytes = results_path.stat().st_size
    db.commit()
    db.refresh(run)
    return _serialize_testing_run(run)


def abort_testing_run(db: Session, run_id: int) -> TestingRunRead | None:
    run = db.get(models.TestingRun, run_id)
    if run is None:
        return None
    if run.status == "queued":
        run.status = "aborted"
        run.ended_at = _utcnow()
        run.error_message = "Aborted before it started."
        db.commit()
        db.refresh(run)
    elif run.status == "running":
        scheduler.request_abort("test", run.id, run.pid)
    else:
        raise ValueError("Only queued or running runs can be aborted.")
    return _serialize_testing_run(run)


def restart_testing_run(db: Session, run_id: int) -> TestingRunRead | None:
    run = db.get(models.TestingRun, run_id)
    if run is None:
        return None
    if run.status in ("queued", "running"):
        raise ValueError("Run is already queued or running.")
    # Clear prior results/CSV and re-queue the same row (one history per config).
    db.execute(delete(models.TestingRunResult).where(models.TestingRunResult.testing_run_id == run.id))
    shutil.rmtree(_testing_run_dir(run.id), ignore_errors=True)
    _reset_testing_run_for_queue(run)
    db.commit()
    db.refresh(run)
    scheduler.wake()
    return _serialize_testing_run(run)


def read_testing_log(db: Session, run_id: int, max_lines: int = 400) -> str | None:
    run = db.get(models.TestingRun, run_id)
    if run is None:
        return None
    if not run.log_path:
        return ""
    try:
        with open(run.log_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return ""
    return "".join(lines[-max_lines:])


def clear_testing_rows_for_training_run(db: Session, training_run_id: int) -> None:
    testing_ids = list(db.scalars(select(models.TestingRun.id).where(models.TestingRun.training_run_id == training_run_id)))
    if not testing_ids:
        return
    db.execute(delete(models.TestingRunResult).where(models.TestingRunResult.testing_run_id.in_(testing_ids)))
    db.execute(delete(models.TestingRun).where(models.TestingRun.id.in_(testing_ids)))
