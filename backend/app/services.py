from __future__ import annotations

import logging
import time
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app import models
import app.modeling  # noqa: F401 ensures core model architectures are registered
from app.modeling.architectures.common import validate_sequential_model_graph
from app.modeling.base import validate_schema_values
from app.modeling.forward import run_image_forward_pass
from app.modeling.layers import list_layer_definitions
from app.modeling.registry import registry as model_registry
from app.modeling.validation import run_cnn_torch_dummy_forward, validate_cnn_tensor_contract
from app.preprocessing.pipeline import (
    encode_png_data_url,
    execute_preview,
    execute_with_previews,
    image_metadata,
    validate_linear_graph,
)
from app.preprocessing.registry import registry
from app.scanner import detect_timestamp_pattern, probe_first_direct_tiff, scan_dataset_files
from app.training.data import FolderTimestampCache, count_folder_range_images
from app.schemas import (
    DatasetConnectionTestResponse,
    DatasetRead,
    MethodConfigurationCreate,
    MethodConfigurationParameterRead,
    MethodConfigurationPayload,
    MethodConfigurationRead,
    MethodTorchCheckResponse,
    MethodConfigurationValidationResponse,
    MethodDefinitionRead,
    ModelLayerRead,
    PreprocessingGraph,
    PreprocessingPipelineCreate,
    PreprocessingPipelineRead,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResponse,
    PreprocessingStepRead,
    TrainingDatasetCreate,
    TrainingDatasetPreviewRequest,
    TrainingDatasetPreviewResponse,
    TrainingDatasetRead,
    TrainingDatasetRulePreview,
    TrainingDatasetRuleRead,
    TrainingPipelineCreate,
    TrainingPipelineDatasetRead,
    TrainingPipelineDryRunRequest,
    TrainingPipelineDryRunResponse,
    TrainingPipelineModelOutput,
    TrainingPipelinePayload,
    TrainingPipelineRead,
)

logger = logging.getLogger("mltrace.services")


def dataset_update_lock_reasons(db: Session, dataset_id: int) -> list[str]:
    referenced_rules = db.scalar(
        select(func.count(models.TrainingDatasetRule.id))
        .join(models.DatasetFolder, models.TrainingDatasetRule.folder_id == models.DatasetFolder.id)
        .where(models.DatasetFolder.dataset_id == dataset_id)
    ) or 0
    if referenced_rules:
        return ["Dataset parser/rescan is locked because its folders are used by Train/Test Datasets."]
    return []


def training_dataset_update_lock_reasons(db: Session, training_dataset_id: int) -> list[str]:
    reasons: list[str] = []
    referencing_pipelines = db.scalar(
        select(func.count(models.TrainingPipelineDataset.id)).where(
            models.TrainingPipelineDataset.training_dataset_id == training_dataset_id
        )
    ) or 0
    if referencing_pipelines:
        reasons.append("Train/Test Dataset not editable, because already used in other Pipelines.")
    referencing_testing_runs = db.scalar(
        select(func.count(models.TestingRun.id)).where(
            models.TestingRun.training_dataset_id == training_dataset_id
        )
    ) or 0
    if referencing_testing_runs:
        reasons.append("Train/Test Dataset not editable, because already used in Testing Runs.")
    return reasons


def preprocessing_pipeline_update_lock_reasons(db: Session, pipeline_id: int) -> list[str]:
    referencing_pipelines = db.scalar(
        select(func.count(models.TrainingPipeline.id)).where(
            models.TrainingPipeline.preprocessing_pipeline_id == pipeline_id
        )
    ) or 0
    if referencing_pipelines:
        return ["Preprocessing Pipeline not editable, because already used in Training Pipelines."]
    return []


def method_configuration_update_lock_reasons(db: Session, configuration_id: int) -> list[str]:
    referencing_pipelines = db.scalar(
        select(func.count(models.TrainingPipeline.id)).where(
            models.TrainingPipeline.method_configuration_id == configuration_id
        )
    ) or 0
    if referencing_pipelines:
        return ["Method/Architecture not editable, because already used in Training Pipelines."]
    return []


def training_pipeline_update_lock_reasons(db: Session, pipeline_id: int) -> list[str]:
    referencing_runs = db.scalar(
        select(func.count(models.TrainingRun.id)).where(models.TrainingRun.training_pipeline_id == pipeline_id)
    ) or 0
    if referencing_runs:
        return ["Training Pipeline not editable, because it already has Training Runs."]
    return []


def _raise_if_locked(reasons: list[str]) -> None:
    if reasons:
        raise ValueError(reasons[0])


def create_dataset(db: Session, name: str, root_path: str) -> models.Dataset:
    started_at = time.perf_counter()
    logger.warning("create_dataset started name=%s root_path=%s", name, root_path)
    root = Path(root_path).expanduser().resolve()
    logger.warning("create_dataset resolved root=%s elapsed=%.3fs", root, time.perf_counter() - started_at)
    if not root.exists() or not root.is_dir():
        logger.warning("create_dataset path invalid root=%s elapsed=%.3fs", root, time.perf_counter() - started_at)
        raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {root}")

    resolved_root = str(root)
    pattern = detect_timestamp_pattern(resolved_root)
    logger.warning(
        "create_dataset detect_timestamp_pattern finished root=%s detected=%s example=%s elapsed=%.3fs",
        resolved_root,
        pattern.regex if pattern else None,
        pattern.example if pattern else None,
        time.perf_counter() - started_at,
    )
    dataset = models.Dataset(
        name=name,
        root_path=resolved_root,
        status="awaiting_confirmation",
        timestamp_regex=pattern.regex if pattern else None,
        timestamp_format=pattern.timestamp_format if pattern else None,
        timestamp_example=pattern.example if pattern else None,
        scan_summary={"detected_matches": pattern.matches} if pattern else {"detected_matches": 0},
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    logger.warning("create_dataset saved dataset_id=%s elapsed=%.3fs", dataset.id, time.perf_counter() - started_at)
    return dataset


def test_dataset_connection(root_path: str) -> DatasetConnectionTestResponse:
    started_at = time.perf_counter()
    logger.warning("test_dataset_connection started root_path=%s", root_path)
    root = Path(root_path).expanduser().resolve(strict=False)
    logger.warning("test_dataset_connection resolved root=%s elapsed=%.3fs", root, time.perf_counter() - started_at)
    if not root.exists():
        logger.warning("test_dataset_connection missing root=%s elapsed=%.3fs", root, time.perf_counter() - started_at)
        return DatasetConnectionTestResponse(
            root_path=str(root),
            exists=False,
            is_directory=False,
            supported_file_found=False,
            sample_file_path=None,
            message=f"Dataset path does not exist: {root}",
        )
    if not root.is_dir():
        logger.warning("test_dataset_connection not directory root=%s elapsed=%.3fs", root, time.perf_counter() - started_at)
        return DatasetConnectionTestResponse(
            root_path=str(root),
            exists=True,
            is_directory=False,
            supported_file_found=False,
            sample_file_path=None,
            message=f"Dataset path is not a directory: {root}",
        )

    probe = probe_first_direct_tiff(root)
    if probe.path is not None:
        logger.warning(
            "test_dataset_connection found sample_file=%s elapsed=%.3fs",
            probe.path,
            time.perf_counter() - started_at,
        )
        return DatasetConnectionTestResponse(
            root_path=str(root),
            exists=True,
            is_directory=True,
            supported_file_found=True,
            sample_file_path=str(probe.path),
            message=f"Path is reachable. Found supported image: {probe.path}",
        )

    if probe.reached_limit:
        message = (
            "Path is reachable, but no supported TIFF file was found within the fast probe limit "
            f"({probe.checked_root_entries} direct entries). Point MLTrace directly at the folder "
            "that contains the TIFF files."
        )
    else:
        message = f"Path is reachable, but no supported TIFF files were found directly in this folder: {root}"
    logger.warning("test_dataset_connection no supported file root=%s elapsed=%.3fs", root, time.perf_counter() - started_at)
    return DatasetConnectionTestResponse(
        root_path=str(root),
        exists=True,
        is_directory=True,
        supported_file_found=False,
        sample_file_path=None,
        message=message,
    )


def get_dataset_or_404(db: Session, dataset_id: int) -> models.Dataset | None:
    return db.scalar(
        select(models.Dataset)
        .where(models.Dataset.id == dataset_id)
        .options(selectinload(models.Dataset.folders))
    )


def scan_dataset(db: Session, dataset: models.Dataset, timestamp_regex: str, timestamp_format: str) -> models.Dataset:
    _raise_if_locked(dataset_update_lock_reasons(db, dataset.id))
    started_at = time.perf_counter()
    logger.warning("scan_dataset started dataset_id=%s root=%s", dataset.id, dataset.root_path)
    dataset.timestamp_regex = timestamp_regex
    dataset.timestamp_format = timestamp_format
    dataset.status = "scanning"
    dataset.scan_error = None
    db.commit()

    try:
        scanned_images, folder_summaries, scan_summary = scan_dataset_files(
            dataset.root_path, timestamp_regex, timestamp_format
        )
        logger.warning(
            "scan_dataset scan_dataset_files finished dataset_id=%s folders=%s representative_images=%s elapsed=%.3fs",
            dataset.id,
            len(folder_summaries),
            len(scanned_images),
            time.perf_counter() - started_at,
        )

        db.execute(delete(models.DatasetImage).where(models.DatasetImage.dataset_id == dataset.id))
        db.execute(delete(models.DatasetFolder).where(models.DatasetFolder.dataset_id == dataset.id))
        db.flush()

        folder_records: dict[str, models.DatasetFolder] = {}
        for relative_path, summary in folder_summaries.items():
            folder = models.DatasetFolder(
                dataset_id=dataset.id,
                relative_path=relative_path,
                image_count=summary["image_count"],
                first_timestamp=summary["first_timestamp"],
                last_timestamp=summary["last_timestamp"],
                extension_summary=summary["extension_summary"],
                resolution_summary=summary["resolution_summary"],
                image_metadata=summary["image_metadata"],
                cadence_summary=summary["cadence_summary"],
            )
            db.add(folder)
            folder_records[relative_path] = folder

        db.flush()

        for image in scanned_images:
            db.add(
                models.DatasetImage(
                    dataset_id=dataset.id,
                    folder_id=folder_records[image.folder_relative_path].id,
                    file_path=str(image.file_path),
                    relative_path=image.relative_path,
                    file_name=image.file_name,
                    extension=image.extension,
                    width=image.width,
                    height=image.height,
                    timestamp_raw=image.timestamp_raw,
                    timestamp_parsed=image.timestamp_parsed,
                    file_size_bytes=image.file_size_bytes,
                    modified_time=image.modified_time,
                )
            )

        dataset.status = "ready"
        dataset.scan_summary = scan_summary
        dataset.scan_error = None
    except Exception as exc:
        dataset.status = "failed"
        dataset.scan_error = str(exc)
        logger.exception("scan_dataset failed dataset_id=%s elapsed=%.3fs", dataset.id, time.perf_counter() - started_at)

    db.commit()
    db.refresh(dataset)
    logger.warning(
        "scan_dataset saved dataset_id=%s status=%s elapsed=%.3fs",
        dataset.id,
        dataset.status,
        time.perf_counter() - started_at,
    )
    return get_dataset_or_404(db, dataset.id) or dataset


def serialize_dataset(db: Session, dataset: models.Dataset) -> DatasetRead:
    reasons = dataset_update_lock_reasons(db, dataset.id)
    return DatasetRead.model_validate(dataset).model_copy(
        update={"is_update_locked": bool(reasons), "update_lock_reasons": reasons}
    )


def list_datasets(db: Session) -> list[DatasetRead]:
    datasets = list(
        db.scalars(
            select(models.Dataset)
            .order_by(models.Dataset.created_at.desc())
            .options(selectinload(models.Dataset.folders))
        )
    )
    return [serialize_dataset(db, dataset) for dataset in datasets]


def delete_dataset(db: Session, dataset_id: int) -> bool:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        return False

    referenced_training_rules = db.scalar(
        select(func.count(models.TrainingDatasetRule.id))
        .join(models.DatasetFolder, models.TrainingDatasetRule.folder_id == models.DatasetFolder.id)
        .where(models.DatasetFolder.dataset_id == dataset_id)
    ) or 0
    if referenced_training_rules:
        raise ValueError(
            "Dataset is used by saved train/test datasets. Delete those train/test datasets before deleting the dataset."
        )

    folder_ids = list(
        db.scalars(select(models.DatasetFolder.id).where(models.DatasetFolder.dataset_id == dataset_id))
    )
    if folder_ids:
        db.query(models.PreprocessingPipeline).filter(
            models.PreprocessingPipeline.preview_folder_id.in_(folder_ids)
        ).update({models.PreprocessingPipeline.preview_folder_id: None}, synchronize_session=False)

    db.delete(dataset)
    db.commit()
    return True


def preview_training_dataset(
    db: Session, request: TrainingDatasetPreviewRequest
) -> TrainingDatasetPreviewResponse:
    validate_rules(db, request.rules)

    previews: list[TrainingDatasetRulePreview] = []
    total_matching = 0
    total_selected = 0
    count_cache: FolderTimestampCache = {}

    for rule in request.rules:
        folder = db.get(models.DatasetFolder, rule.folder_id)
        if folder is None:
            raise ValueError(f"Folder does not exist: {rule.folder_id}")
        counts = count_folder_range_images(folder, rule.start_timestamp, rule.end_timestamp, rule.stride, count_cache)
        previews.append(
            TrainingDatasetRulePreview(
                folder_id=rule.folder_id,
                start_timestamp=rule.start_timestamp,
                end_timestamp=rule.end_timestamp,
                stride=rule.stride,
                matching_images=counts.matching_images,
                selected_images=counts.selected_images,
            )
        )
        total_matching += counts.matching_images
        total_selected += counts.selected_images

    return TrainingDatasetPreviewResponse(
        total_matching_images=total_matching,
        total_selected_images=total_selected,
        rules=previews,
    )


def create_training_dataset(db: Session, payload: TrainingDatasetCreate) -> TrainingDatasetRead:
    validate_rules(db, payload.rules)

    training_dataset = models.TrainingDataset(
        name=payload.name,
        usage_label=payload.usage_label,
        notes=payload.notes,
    )
    db.add(training_dataset)
    db.flush()

    for rule in payload.rules:
        db.add(
            models.TrainingDatasetRule(
                training_dataset_id=training_dataset.id,
                folder_id=rule.folder_id,
                start_timestamp=rule.start_timestamp,
                end_timestamp=rule.end_timestamp,
                stride=rule.stride,
            )
        )

    db.commit()
    saved = db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset.id)
        .options(
            selectinload(models.TrainingDataset.rules)
            .selectinload(models.TrainingDatasetRule.folder)
            .selectinload(models.DatasetFolder.dataset)
        )
    ) or training_dataset
    return serialize_training_dataset(db, saved)


def update_training_dataset(
    db: Session, training_dataset_id: int, payload: TrainingDatasetCreate
) -> TrainingDatasetRead | None:
    training_dataset = db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset_id)
        .options(selectinload(models.TrainingDataset.rules))
    )
    if training_dataset is None:
        return None
    _raise_if_locked(training_dataset_update_lock_reasons(db, training_dataset_id))
    validate_rules(db, payload.rules)

    training_dataset.name = payload.name
    training_dataset.usage_label = payload.usage_label
    training_dataset.notes = payload.notes
    db.execute(
        delete(models.TrainingDatasetRule).where(
            models.TrainingDatasetRule.training_dataset_id == training_dataset_id
        )
    )
    db.flush()
    for rule in payload.rules:
        db.add(
            models.TrainingDatasetRule(
                training_dataset_id=training_dataset_id,
                folder_id=rule.folder_id,
                start_timestamp=rule.start_timestamp,
                end_timestamp=rule.end_timestamp,
                stride=rule.stride,
            )
        )
    db.commit()
    return get_training_dataset(db, training_dataset_id)


def list_training_datasets(db: Session) -> list[TrainingDatasetRead]:
    training_datasets = list(
        db.scalars(
            select(models.TrainingDataset)
            .order_by(models.TrainingDataset.created_at.desc())
            .options(
                selectinload(models.TrainingDataset.rules)
                .selectinload(models.TrainingDatasetRule.folder)
                .selectinload(models.DatasetFolder.dataset)
            )
        )
    )
    count_cache: FolderTimestampCache = {}
    return [serialize_training_dataset(db, training_dataset, count_cache) for training_dataset in training_datasets]


def get_training_dataset(db: Session, training_dataset_id: int) -> TrainingDatasetRead | None:
    training_dataset = db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset_id)
        .options(
            selectinload(models.TrainingDataset.rules)
            .selectinload(models.TrainingDatasetRule.folder)
            .selectinload(models.DatasetFolder.dataset)
        )
    )
    if training_dataset is None:
        return None
    return serialize_training_dataset(db, training_dataset, {})


def delete_training_dataset(db: Session, training_dataset_id: int) -> bool:
    training_dataset = db.get(models.TrainingDataset, training_dataset_id)
    if training_dataset is None:
        return False
    referencing_pipelines = db.scalar(
        select(func.count(models.TrainingPipelineDataset.id)).where(
            models.TrainingPipelineDataset.training_dataset_id == training_dataset_id
        )
    ) or 0
    if referencing_pipelines:
        raise ValueError(
            "Train/test dataset is used by saved training pipelines. Delete those training pipelines first."
        )
    db.delete(training_dataset)
    db.commit()
    return True


def cleanup_invalid_training_dataset_rules(db: Session, training_dataset_id: int) -> TrainingDatasetRead | None:
    training_dataset = db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == training_dataset_id)
        .options(
            selectinload(models.TrainingDataset.rules)
            .selectinload(models.TrainingDatasetRule.folder)
            .selectinload(models.DatasetFolder.dataset)
        )
    )
    if training_dataset is None:
        return None
    _raise_if_locked(training_dataset_update_lock_reasons(db, training_dataset_id))
    invalid_rule_ids = [
        rule.id
        for rule in training_dataset.rules
        if rule.folder is None or rule.folder.dataset is None
    ]
    if invalid_rule_ids:
        db.execute(delete(models.TrainingDatasetRule).where(models.TrainingDatasetRule.id.in_(invalid_rule_ids)))
        db.commit()
    return get_training_dataset(db, training_dataset_id)


def validate_rules(db: Session, rules) -> None:
    if not rules:
        raise ValueError("At least one rule is required.")

    folder_ids = [rule.folder_id for rule in rules]
    folders = {
        folder.id: folder
        for folder in db.scalars(
            select(models.DatasetFolder).where(models.DatasetFolder.id.in_(folder_ids))
        )
    }
    missing = sorted(set(folder_ids) - set(folders))
    if missing:
        raise ValueError(f"Folders do not exist: {missing}")

    signatures: set[str] = set()
    for rule in rules:
        folder = folders[rule.folder_id]
        if folder.first_timestamp is None or folder.last_timestamp is None:
            raise ValueError(f"Folder {folder.id} has no timestamp bounds.")
        if rule.start_timestamp < folder.first_timestamp or rule.end_timestamp > folder.last_timestamp:
            raise ValueError(
                f"Rule for folder {folder.relative_path} must stay within "
                f"{folder.first_timestamp.isoformat()} and {folder.last_timestamp.isoformat()}."
            )
        signature = folder_image_signature(folder)
        if signature is None:
            raise ValueError(
                f"Folder {folder.relative_path} is missing image metadata. Rescan the source dataset before using it."
            )
        signatures.add(signature)

    if len(signatures) > 1:
        raise ValueError(
            "All ranges in one train/test dataset must use the same image data signature. "
            f"Found: {', '.join(sorted(signatures))}."
        )


def count_rule_images(
    db: Session, rule: models.TrainingDatasetRule, count_cache: FolderTimestampCache | None = None
) -> tuple[int, int]:
    counts = count_folder_range_images(rule.folder, rule.start_timestamp, rule.end_timestamp, rule.stride, count_cache)
    return counts.matching_images, counts.selected_images


def count_images_in_folder_range(
    folder: models.DatasetFolder,
    start_timestamp,
    end_timestamp,
    count_cache: FolderTimestampCache | None = None,
) -> int:
    return count_folder_range_images(folder, start_timestamp, end_timestamp, 1, count_cache).matching_images


def folder_image_signature(folder: models.DatasetFolder) -> str | None:
    """Stable compatibility key for combining folders into one train/test set."""
    if not folder.resolution_summary or not folder.extension_summary or not folder.image_metadata:
        return None
    resolutions = ",".join(sorted(str(key) for key in folder.resolution_summary.keys()))
    extensions = ",".join(sorted(str(key) for key in folder.extension_summary.keys()))
    dtype = folder.image_metadata.get("dtype") or "unknown-dtype"
    channels = folder.image_metadata.get("channels")
    mode = folder.image_metadata.get("mode") or "unknown-mode"
    if channels is None:
        channels_text = "unknown-ch"
    else:
        channels_text = f"{channels}ch"
    return f"{resolutions} | {extensions} | {dtype} | {channels_text} | {mode}"


def serialize_training_dataset(
    db: Session,
    training_dataset: models.TrainingDataset,
    count_cache: FolderTimestampCache | None = None,
) -> TrainingDatasetRead:
    rule_reads: list[TrainingDatasetRuleRead] = []
    dataset_names: set[str] = set()
    resolutions: set[str] = set()
    signatures: set[str] = set()
    total_matching = 0
    total_selected = 0
    invalid_rules = [
        rule
        for rule in training_dataset.rules
        if rule.folder is None or rule.folder.dataset is None
    ]
    integrity_warnings = []
    if invalid_rules:
        integrity_warnings.append(
            f"{len(invalid_rules)} invalid Train/Test rule(s) reference a missing dataset folder and were skipped."
        )
    lock_reasons = training_dataset_update_lock_reasons(db, training_dataset.id)

    for rule in sorted(
        [
            rule
            for rule in training_dataset.rules
            if rule.folder is not None and rule.folder.dataset is not None
        ],
        key=lambda item: (
            item.start_timestamp,
            item.end_timestamp,
            item.folder.dataset.name,
            item.folder.relative_path,
            item.id,
        ),
    ):
        folder = rule.folder
        dataset = folder.dataset
        matching, selected = count_rule_images(db, rule, count_cache)
        total_matching += matching
        total_selected += selected
        dataset_names.add(dataset.name)
        if folder.resolution_summary:
            resolutions.update(folder.resolution_summary.keys())
        signature = folder_image_signature(folder)
        if signature:
            signatures.add(signature)
        rule_reads.append(
            TrainingDatasetRuleRead(
                id=rule.id,
                folder_id=folder.id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                dataset_root_path=dataset.root_path,
                folder_relative_path=folder.relative_path,
                folder_first_timestamp=folder.first_timestamp,
                folder_last_timestamp=folder.last_timestamp,
                folder_extension_summary=folder.extension_summary,
                folder_resolution_summary=folder.resolution_summary,
                folder_image_metadata=folder.image_metadata,
                folder_image_signature=signature,
                start_timestamp=rule.start_timestamp,
                end_timestamp=rule.end_timestamp,
                stride=rule.stride,
                matching_images=matching,
                selected_images=selected,
            )
        )

    return TrainingDatasetRead(
        id=training_dataset.id,
        name=training_dataset.name,
        usage_label=training_dataset.usage_label,
        notes=training_dataset.notes,
        created_at=training_dataset.created_at,
        dataset_names=sorted(dataset_names),
        image_resolutions=sorted(resolutions),
        image_signatures=sorted(signatures),
        total_matching_images=total_matching,
        total_selected_images=total_selected,
        rules=rule_reads,
        is_update_locked=bool(lock_reasons),
        update_lock_reasons=lock_reasons,
        invalid_rule_count=len(invalid_rules),
        integrity_warnings=integrity_warnings,
    )


def list_preprocessing_steps() -> list[PreprocessingStepRead]:
    return [
        PreprocessingStepRead(
            type=definition.type,
            label=definition.label,
            category=definition.category,
            input_kind=definition.input_kind,
            output_kind=definition.output_kind,
            config_schema=definition.config_schema,
            default_config=definition.default_config,
        )
        for definition in registry.list_definitions()
    ]


def _assert_unique_pipeline_name(db: Session, name: str, exclude_id: int | None = None) -> None:
    query = select(models.PreprocessingPipeline).where(
        func.lower(models.PreprocessingPipeline.name) == name.lower()
    )
    if exclude_id is not None:
        query = query.where(models.PreprocessingPipeline.id != exclude_id)
    if db.scalar(query) is not None:
        raise ValueError(f"A preprocessing pipeline named '{name}' already exists.")


def create_preprocessing_pipeline(db: Session, payload: PreprocessingPipelineCreate) -> PreprocessingPipelineRead:
    validate_linear_graph(payload.graph)
    _assert_unique_pipeline_name(db, payload.name)
    pipeline = models.PreprocessingPipeline(
        name=payload.name,
        description=payload.description,
        graph=payload.graph.model_dump(mode="json"),
        preview_folder_id=payload.preview_folder_id,
        input_width=payload.input_width,
        input_height=payload.input_height,
        output_width=payload.output_width,
        output_height=payload.output_height,
    )
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    return serialize_preprocessing_pipeline(db, pipeline)


def update_preprocessing_pipeline(
    db: Session, pipeline_id: int, payload: PreprocessingPipelineCreate
) -> PreprocessingPipelineRead | None:
    pipeline = db.get(models.PreprocessingPipeline, pipeline_id)
    if pipeline is None:
        return None
    _raise_if_locked(preprocessing_pipeline_update_lock_reasons(db, pipeline_id))
    validate_linear_graph(payload.graph)
    _assert_unique_pipeline_name(db, payload.name, exclude_id=pipeline_id)
    pipeline.name = payload.name
    pipeline.description = payload.description
    pipeline.graph = payload.graph.model_dump(mode="json")
    pipeline.preview_folder_id = payload.preview_folder_id
    # Only overwrite the stored resolution when the client provides one (a save without a
    # preview should not wipe the previously recorded sizes).
    if payload.input_width is not None:
        pipeline.input_width = payload.input_width
    if payload.input_height is not None:
        pipeline.input_height = payload.input_height
    if payload.output_width is not None:
        pipeline.output_width = payload.output_width
    if payload.output_height is not None:
        pipeline.output_height = payload.output_height
    db.commit()
    db.refresh(pipeline)
    return serialize_preprocessing_pipeline(db, pipeline)


def serialize_preprocessing_pipeline(
    db: Session, pipeline: models.PreprocessingPipeline
) -> PreprocessingPipelineRead:
    reasons = preprocessing_pipeline_update_lock_reasons(db, pipeline.id)
    return PreprocessingPipelineRead.model_validate(pipeline).model_copy(
        update={"is_update_locked": bool(reasons), "update_lock_reasons": reasons}
    )


def list_preprocessing_pipelines(db: Session) -> list[PreprocessingPipelineRead]:
    pipelines = list(db.scalars(select(models.PreprocessingPipeline).order_by(models.PreprocessingPipeline.created_at.desc())))
    return [serialize_preprocessing_pipeline(db, pipeline) for pipeline in pipelines]


def get_preprocessing_pipeline(db: Session, pipeline_id: int) -> PreprocessingPipelineRead | None:
    pipeline = db.get(models.PreprocessingPipeline, pipeline_id)
    if pipeline is None:
        return None
    return serialize_preprocessing_pipeline(db, pipeline)


def delete_preprocessing_pipeline(db: Session, pipeline_id: int) -> bool:
    pipeline = db.get(models.PreprocessingPipeline, pipeline_id)
    if pipeline is None:
        return False
    referencing_pipelines = db.scalar(
        select(func.count(models.TrainingPipeline.id)).where(
            models.TrainingPipeline.preprocessing_pipeline_id == pipeline_id
        )
    ) or 0
    if referencing_pipelines:
        raise ValueError(
            "Preprocessing pipeline is used by saved training pipelines. Delete those training pipelines first."
        )
    db.delete(pipeline)
    db.commit()
    return True


def preview_preprocessing_pipeline(
    db: Session, payload: PreprocessingPreviewRequest
) -> PreprocessingPreviewResponse:
    validate_linear_graph(payload.graph)
    source_image = db.scalar(
        select(models.DatasetImage)
        .where(models.DatasetImage.folder_id == payload.folder_id)
        .order_by(models.DatasetImage.timestamp_parsed.asc(), models.DatasetImage.id.asc())
        .limit(1)
    )
    if source_image is None:
        raise ValueError("Selected folder does not contain indexed images.")
    previews = execute_preview(payload.graph, source_image.file_path)
    return PreprocessingPreviewResponse(
        source_image_id=source_image.id,
        source_image_path=source_image.file_path,
        source_timestamp=source_image.timestamp_parsed,
        previews=previews,
    )


def list_method_definitions() -> list[MethodDefinitionRead]:
    return [
        MethodDefinitionRead(
            type=definition.type,
            label=definition.label,
            category=definition.category,
            description=definition.description,
            framework=definition.framework,
            method_family=definition.method_family,
            method_version=definition.method_version,
            training_mode=definition.training_mode,
            architecture_version=definition.architecture_version,
            requires_training=definition.requires_training,
            supports_training_pipeline=definition.supports_training_pipeline,
            artifact_kind=definition.artifact_kind,
            builder_kind=definition.builder_kind,
            capabilities=definition.capabilities,
            method_schema=definition.method_schema,
            model_schema=definition.model_schema,
            training_schema=definition.training_schema,
            inference_schema=definition.inference_schema,
            default_method_config=definition.default_method_config,
            default_model_config=definition.default_model_config,
            default_training_config=definition.default_training_config,
            default_inference_config=definition.default_inference_config,
        )
        for definition in model_registry.list_definitions()
    ]


def get_method_definition(method_type: str) -> MethodDefinitionRead:
    method = model_registry.get(method_type)
    return MethodDefinitionRead(
        type=method.type,
        label=method.label,
        category=method.category,
        description=method.description,
        framework=method.framework,
        method_family=method.method_family,
        method_version=method.method_version,
        training_mode=method.training_mode,
        architecture_version=method.architecture_version,
        requires_training=method.requires_training,
        supports_training_pipeline=method.supports_training_pipeline,
        artifact_kind=method.artifact_kind,
        builder_kind=method.builder_kind,
        capabilities=method.capabilities,
        method_schema=method.method_schema,
        model_schema=method.model_schema,
        training_schema=method.training_schema,
        inference_schema=method.inference_schema,
        default_method_config=method.default_method_config,
        default_model_config=method.default_model_config,
        default_training_config=method.default_training_config,
        default_inference_config=method.default_inference_config,
    )


def list_method_layers() -> list[ModelLayerRead]:
    return [
        ModelLayerRead(
            type=definition.type,
            label=definition.label,
            category=definition.category,
            config_schema=definition.config_schema,
            default_config=definition.default_config,
            input_rank=definition.input_rank,
            output_rank=definition.output_rank,
            shape_notes=definition.shape_notes,
        )
        for definition in list_layer_definitions()
    ]


def _assert_unique_method_name(db: Session, name: str, exclude_id: int | None = None) -> None:
    query = select(models.MethodConfiguration).where(func.lower(models.MethodConfiguration.name) == name.lower())
    if exclude_id is not None:
        query = query.where(models.MethodConfiguration.id != exclude_id)
    if db.scalar(query) is not None:
        raise ValueError(f"A method named '{name}' already exists.")


def _payload_method_type(payload: MethodConfigurationPayload) -> str:
    method_type = payload.method_type or payload.architecture_type
    if not method_type:
        raise ValueError("method_type is required.")
    return method_type


def _payload_method_graph(payload: MethodConfigurationPayload) -> dict:
    return payload.method_graph or payload.model_graph or {}


def _payload_method_config(payload: MethodConfigurationPayload) -> dict:
    return {**(payload.model_params or {}), **(payload.method_config or {})}


def _normalize_method_payload(payload: MethodConfigurationPayload, reject_invalid: bool = True) -> tuple:
    method = model_registry.get(_payload_method_type(payload))
    method_config = method.merged_method_config(_payload_method_config(payload))
    training_config = method.merged_training_config(payload.training_config)
    inference_config = method.merged_inference_config(payload.inference_config)
    method_graph = _payload_method_graph(payload)
    method.validate_config(method_graph, method_config, training_config, inference_config)

    validation = {"valid": True, "errors": [], "warnings": [], "layer_specs": [], "torch_check": None}
    if method.builder_kind != "form":
        method_graph = validate_sequential_model_graph(method_graph, method.builder_kind)
        validation = validate_cnn_tensor_contract(method_graph, method_config)
        if validation["errors"] and reject_invalid:
            raise ValueError("; ".join(validation["errors"]))
    else:
        method_graph = {}

    diagram = build_method_diagram(
        method_type=method.type,
        builder_kind=method.builder_kind,
        method_graph=method_graph,
        method_config=method_config,
        validation=validation,
    )
    return method, method_graph, method_config, training_config, inference_config, diagram, validation


def build_method_diagram(
    method_type: str,
    builder_kind: str,
    method_graph: dict,
    method_config: dict,
    validation: dict | None = None,
) -> dict:
    if builder_kind == "form":
        nodes = [
            {"id": "input", "label": "Input images", "section": "input", "detail": "Training dataset images"},
            {
                "id": "aggregate",
                "label": "Aggregate mean image",
                "section": "method",
                "detail": f"{method_config.get('aggregation', 'mean')} / {method_config.get('accumulator_dtype', 'float32')}",
            },
            {
                "id": "reference",
                "label": "Reconstruction/error reference",
                "section": "output",
                "detail": f"output {method_config.get('output_dtype_policy', 'source')}",
            },
        ]
        return {
            "method_type": method_type,
            "architecture_type": method_type,
            "builder_kind": builder_kind,
            "nodes": nodes,
            "edges": [{"source": "input", "target": "aggregate"}, {"source": "aggregate", "target": "reference"}],
        }

    shape_by_node = {
        item["layer_id"]: item["output_label"]
        for item in (validation or {}).get("layer_specs", [])
        if item.get("layer_id")
    }
    nodes: list[dict] = [
        {
            "id": "input",
            "label": "Input",
            "section": "input",
            "detail": (
                f"{method_config.get('input_channels', '?')}x"
                f"{method_config.get('input_height', '?')}x{method_config.get('input_width', '?')}"
            ),
        }
    ]
    edges: list[dict] = []
    previous_id = "input"

    def add_node(node_id: str, label: str, section: str, detail: str) -> None:
        nonlocal previous_id
        nodes.append({"id": node_id, "label": label, "section": section, "detail": detail})
        edges.append({"source": previous_id, "target": node_id})
        previous_id = node_id

    for index, layer in enumerate(method_graph.get("encoder", []), start=1):
        detail = _layer_summary(layer)
        if shape_by_node.get(layer.get("id")):
            detail = f"{detail} -> {shape_by_node[layer.get('id')]}"
        add_node(f"encoder-{index}", layer["type"], "encoder", detail)

    if builder_kind == "sequential_variational_autoencoder":
        latent_dim = method_config.get("latent_dim", "?")
        add_node(
            "vae-mu-logvar",
            "Mu/logvar projection",
            "latent",
            f"encoder output -> mu/logvar dim {latent_dim}",
        )
        add_node(
            "latent",
            "Sample z",
            "latent",
            f"dim {latent_dim}, KL {method_config.get('kl_weight', '?')}, reparameterization",
        )
        add_node("vae-seed", "Decoder seed projection", "latent", "z -> encoder output shape")
    else:
        add_node("latent", "Latent", "latent", f"dim {method_config.get('latent_dim', '?')}")

    for index, layer in enumerate(method_graph.get("decoder", []), start=1):
        detail = _layer_summary(layer)
        if shape_by_node.get(layer.get("id")):
            detail = f"{detail} -> {shape_by_node[layer.get('id')]}"
        add_node(f"decoder-{index}", layer["type"], "decoder", detail)

    add_node(
        "output",
        "Output reconstruction",
        "output",
        f"activation {method_config.get('output_activation', 'none')}",
    )

    return {
        "method_type": method_type,
        "architecture_type": method_type,
        "builder_kind": builder_kind,
        "nodes": nodes,
        "edges": edges,
    }


def _layer_summary(layer: dict) -> str:
    config = layer.get("config") or {}
    interesting_keys = [
        "out_channels",
        "out_features",
        "kernel_size",
        "stride",
        "padding",
        "scale_factor",
        "mode",
        "p",
        "channels",
        "height",
        "width",
    ]
    parts = [f"{key}={config[key]}" for key in interesting_keys if key in config]
    return ", ".join(parts) if parts else "default config"


def validate_method_configuration(payload: MethodConfigurationPayload) -> MethodConfigurationValidationResponse:
    try:
        _, _, _, _, _, diagram, validation = _normalize_method_payload(payload, reject_invalid=False)
    except ValueError as exc:
        return MethodConfigurationValidationResponse(valid=False, errors=[str(exc)], diagram={})
    return MethodConfigurationValidationResponse(
        valid=validation["valid"],
        errors=validation["errors"],
        warnings=validation["warnings"],
        layer_specs=validation["layer_specs"],
        torch_check=validation["torch_check"],
        diagram=diagram,
    )


def run_method_torch_check(payload: MethodConfigurationPayload) -> MethodTorchCheckResponse:
    try:
        method, method_graph, method_config, _, _, _, validation = _normalize_method_payload(payload, reject_invalid=False)
    except ValueError as exc:
        return MethodTorchCheckResponse(
            valid=False,
            status="failed",
            errors=[str(exc)],
            logs=[str(exc), "Failed"],
            torch_check={"status": "failed", "message": str(exc)},
        )

    if method.builder_kind == "form":
        message = "Torch dummy-forward check is only available for neural sequential methods."
        return MethodTorchCheckResponse(
            valid=False,
            status="not_applicable",
            warnings=[message],
            logs=[message],
            torch_check={"status": "not_applicable", "message": message},
        )

    if validation["errors"]:
        message = "Static shape validation failed; fix hard errors before running the Torch check."
        return MethodTorchCheckResponse(
            valid=False,
            status="failed",
            errors=[message, *validation["errors"]],
            warnings=validation["warnings"],
            logs=[message, "Failed"],
            torch_check={"status": "failed", "message": message},
        )

    result = run_cnn_torch_dummy_forward(method_graph, method_config)
    return MethodTorchCheckResponse(**result)


def create_method_configuration(db: Session, payload: MethodConfigurationCreate) -> MethodConfigurationRead:
    _assert_unique_method_name(db, payload.name)
    method, method_graph, method_config, training_config, inference_config, diagram, validation = _normalize_method_payload(payload)
    method_configuration = models.MethodConfiguration(
        name=payload.name,
        description=payload.description,
        method_type=method.type,
        method_family=method.method_family,
        method_version=method.method_version,
        training_mode=method.training_mode,
        requires_training=method.requires_training,
        supports_training_pipeline=method.supports_training_pipeline,
        artifact_kind=method.artifact_kind,
        builder_kind=method.builder_kind,
        method_graph=method_graph,
        method_config=method_config,
        training_config=training_config,
        inference_config=inference_config,
        diagram=diagram,
        validation=validation,
    )
    db.add(method_configuration)
    db.flush()
    _replace_method_parameter_index(db, method_configuration)
    db.commit()
    return get_method_configuration(db, method_configuration.id)  # type: ignore[return-value]


def list_method_configurations(db: Session) -> list[MethodConfigurationRead]:
    configurations = list(
        db.scalars(
            select(models.MethodConfiguration)
            .order_by(models.MethodConfiguration.created_at.desc())
            .options(selectinload(models.MethodConfiguration.parameters))
        )
    )
    return [serialize_method_configuration(configuration, db) for configuration in configurations]


def get_method_configuration(db: Session, configuration_id: int) -> MethodConfigurationRead | None:
    configuration = db.scalar(
        select(models.MethodConfiguration)
        .where(models.MethodConfiguration.id == configuration_id)
        .options(selectinload(models.MethodConfiguration.parameters))
    )
    if configuration is None:
        return None
    return serialize_method_configuration(configuration, db)


def update_method_configuration(
    db: Session, configuration_id: int, payload: MethodConfigurationCreate
) -> MethodConfigurationRead | None:
    configuration = db.get(models.MethodConfiguration, configuration_id)
    if configuration is None:
        return None
    _raise_if_locked(method_configuration_update_lock_reasons(db, configuration_id))
    _assert_unique_method_name(db, payload.name, exclude_id=configuration_id)
    method, method_graph, method_config, training_config, inference_config, diagram, validation = _normalize_method_payload(payload)
    configuration.name = payload.name
    configuration.description = payload.description
    configuration.method_type = method.type
    configuration.method_family = method.method_family
    configuration.method_version = method.method_version
    configuration.training_mode = method.training_mode
    configuration.requires_training = method.requires_training
    configuration.supports_training_pipeline = method.supports_training_pipeline
    configuration.artifact_kind = method.artifact_kind
    configuration.builder_kind = method.builder_kind
    configuration.method_graph = method_graph
    configuration.method_config = method_config
    configuration.training_config = training_config
    configuration.inference_config = inference_config
    configuration.diagram = diagram
    configuration.validation = validation
    _replace_method_parameter_index(db, configuration)
    db.commit()
    return get_method_configuration(db, configuration_id)


def delete_method_configuration(db: Session, configuration_id: int) -> bool:
    configuration = db.get(models.MethodConfiguration, configuration_id)
    if configuration is None:
        return False
    referencing_pipelines = db.scalar(
        select(func.count(models.TrainingPipeline.id)).where(
            models.TrainingPipeline.method_configuration_id == configuration_id
        )
    ) or 0
    if referencing_pipelines:
        raise ValueError(
            "Method configuration is used by saved training pipelines. Delete those training pipelines first."
        )
    db.delete(configuration)
    db.commit()
    return True


def serialize_method_configuration(
    configuration: models.MethodConfiguration, db: Session | None = None
) -> MethodConfigurationRead:
    parameters = [
        MethodConfigurationParameterRead(
            path=parameter.path,
            value_type=parameter.value_type,
            value_text=parameter.value_text,
            value_number=parameter.value_number,
            value_bool=parameter.value_bool,
        )
        for parameter in sorted(configuration.parameters, key=lambda item: item.path)
    ]
    lock_reasons = method_configuration_update_lock_reasons(db, configuration.id) if db is not None else []
    return MethodConfigurationRead(
        id=configuration.id,
        name=configuration.name,
        description=configuration.description,
        method_type=configuration.method_type,
        method_family=configuration.method_family,
        method_version=configuration.method_version,
        training_mode=configuration.training_mode,
        architecture_type=configuration.method_type,
        architecture_version=configuration.method_version,
        requires_training=configuration.requires_training,
        supports_training_pipeline=configuration.supports_training_pipeline,
        artifact_kind=configuration.artifact_kind,
        builder_kind=configuration.builder_kind,
        method_graph=configuration.method_graph,
        model_graph=configuration.method_graph,
        method_config=configuration.method_config,
        model_params=configuration.method_config,
        training_config=configuration.training_config,
        inference_config=configuration.inference_config,
        diagram=configuration.diagram,
        created_at=configuration.created_at,
        updated_at=configuration.updated_at,
        validation=configuration.validation,
        parameters=parameters,
        is_update_locked=bool(lock_reasons),
        update_lock_reasons=lock_reasons,
    )


def _replace_method_parameter_index(db: Session, configuration: models.MethodConfiguration) -> None:
    db.execute(
        delete(models.MethodConfigurationParameter).where(
            models.MethodConfigurationParameter.method_configuration_id == configuration.id
        )
    )
    for path, value in _flatten_scalar_values(
        {
            "method_type": configuration.method_type,
            "method_family": configuration.method_family,
            "method_version": configuration.method_version,
            "training_mode": configuration.training_mode,
            "requires_training": configuration.requires_training,
            "supports_training_pipeline": configuration.supports_training_pipeline,
            "artifact_kind": configuration.artifact_kind,
            "builder_kind": configuration.builder_kind,
            "method_graph": configuration.method_graph,
            "method_config": configuration.method_config,
            "training_config": configuration.training_config,
            "inference_config": configuration.inference_config,
        }
    ):
        parameter = models.MethodConfigurationParameter(
            method_configuration_id=configuration.id,
            path=path,
            value_type=_scalar_value_type(value),
            value_text=str(value) if isinstance(value, str) else None,
            value_number=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
            value_bool=value if isinstance(value, bool) else None,
        )
        db.add(parameter)


list_model_architectures = list_method_definitions
get_model_architecture = get_method_definition
list_model_layers = list_method_layers
validate_model_configuration = validate_method_configuration
create_model_configuration = create_method_configuration
list_model_configurations = list_method_configurations
get_model_configuration = get_method_configuration
update_model_configuration = update_method_configuration
delete_model_configuration = delete_method_configuration


def _flatten_scalar_values(value, prefix: str = "") -> list[tuple[str, object]]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [(prefix, value)] if prefix else []
    if isinstance(value, dict):
        items: list[tuple[str, object]] = []
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_scalar_values(child, next_prefix))
        return items
    if isinstance(value, list):
        items = []
        for index, child in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            items.extend(_flatten_scalar_values(child, next_prefix))
        return items
    return []


def _scalar_value_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "number"
    if isinstance(value, float):
        return "number"
    return "text"


# --- Training pipelines -----------------------------------------------------


class DuplicatePipelineError(ValueError):
    """Raised when a training pipeline with an identical configuration exists."""

    def __init__(self, existing: "models.TrainingPipeline") -> None:
        self.existing = existing
        super().__init__(
            f"An identical training pipeline configuration already exists as '{existing.name}'."
        )


def training_pipeline_signature(
    dataset_ids,
    preprocessing_pipeline_id: int,
    method_configuration_id: int,
    shuffle: bool,
    training_parameters: dict | None,
) -> str:
    """Stable hash of the full configuration, independent of name/order.

    Mirrored by alembic migration 0016 for backfilling existing rows.
    """
    import hashlib
    import json

    canonical = {
        "datasets": sorted({int(value) for value in dataset_ids}),
        "preprocessing": int(preprocessing_pipeline_id),
        "method": int(method_configuration_id),
        "shuffle": bool(shuffle),
        "params": training_parameters or {},
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def find_training_pipeline_by_signature(
    db: Session, payload: TrainingPipelinePayload
) -> models.TrainingPipeline | None:
    """Return an existing pipeline whose full configuration matches the payload."""
    _, _, configuration = _resolve_training_pipeline_refs(db, payload)
    training_parameters = _merged_training_parameters(configuration, payload.training_parameters)
    signature = training_pipeline_signature(
        payload.training_dataset_ids,
        payload.preprocessing_pipeline_id,
        payload.method_configuration_id,
        payload.shuffle,
        training_parameters,
    )
    return db.scalar(
        select(models.TrainingPipeline).where(models.TrainingPipeline.config_signature == signature)
    )


def _assert_unique_training_pipeline_name(db: Session, name: str, exclude_id: int | None = None) -> None:
    query = select(models.TrainingPipeline).where(func.lower(models.TrainingPipeline.name) == name.lower())
    if exclude_id is not None:
        query = query.where(models.TrainingPipeline.id != exclude_id)
    if db.scalar(query) is not None:
        raise ValueError(f"A training pipeline named '{name}' already exists.")


def _resolve_training_pipeline_refs(
    db: Session, payload: TrainingPipelinePayload
) -> tuple[list[models.TrainingDataset], models.PreprocessingPipeline, models.MethodConfiguration]:
    """Load and validate all building blocks referenced by a pipeline payload.

    Training datasets are returned in payload order (which becomes the stored
    position). Save, update and dry-run all funnel through this guard.
    """
    training_datasets_by_id = {
        training_dataset.id: training_dataset
        for training_dataset in db.scalars(
            select(models.TrainingDataset)
            .where(models.TrainingDataset.id.in_(payload.training_dataset_ids))
            .options(
                selectinload(models.TrainingDataset.rules)
                .selectinload(models.TrainingDatasetRule.folder)
                .selectinload(models.DatasetFolder.dataset)
            )
        )
    }
    missing = sorted(set(payload.training_dataset_ids) - set(training_datasets_by_id))
    if missing:
        raise ValueError(f"Training datasets do not exist: {missing}")
    training_datasets = [training_datasets_by_id[item] for item in payload.training_dataset_ids]

    preprocessing_pipeline = db.get(models.PreprocessingPipeline, payload.preprocessing_pipeline_id)
    if preprocessing_pipeline is None:
        raise ValueError(f"Preprocessing pipeline does not exist: {payload.preprocessing_pipeline_id}")

    configuration = db.get(models.MethodConfiguration, payload.method_configuration_id)
    if configuration is None:
        raise ValueError(f"Method configuration does not exist: {payload.method_configuration_id}")
    if not configuration.supports_training_pipeline:
        raise ValueError(
            f"Method '{configuration.name}' ({configuration.method_type}) does not support training pipelines."
        )

    return training_datasets, preprocessing_pipeline, configuration


def _merged_training_parameters(configuration: models.MethodConfiguration, overrides: dict | None) -> dict:
    """Merge user overrides onto the saved method's training config and validate.

    The saved configuration's training_config (already merged from the method
    definition's defaults at method save time) acts as the baseline.
    """
    definition = model_registry.get(configuration.method_type)
    merged = {**(configuration.training_config or {}), **(overrides or {})}
    validate_schema_values(definition.training_schema, merged, f"{definition.type}.training_parameters")
    return merged


def create_training_pipeline(db: Session, payload: TrainingPipelineCreate) -> TrainingPipelineRead:
    _assert_unique_training_pipeline_name(db, payload.name)
    training_datasets, _, configuration = _resolve_training_pipeline_refs(db, payload)
    training_parameters = _merged_training_parameters(configuration, payload.training_parameters)
    signature = training_pipeline_signature(
        payload.training_dataset_ids,
        payload.preprocessing_pipeline_id,
        payload.method_configuration_id,
        payload.shuffle,
        training_parameters,
    )
    existing = db.scalar(
        select(models.TrainingPipeline).where(models.TrainingPipeline.config_signature == signature)
    )
    if existing is not None:
        raise DuplicatePipelineError(existing)

    pipeline = models.TrainingPipeline(
        name=payload.name,
        description=payload.description,
        preprocessing_pipeline_id=payload.preprocessing_pipeline_id,
        method_configuration_id=payload.method_configuration_id,
        shuffle=payload.shuffle,
        training_parameters=training_parameters,
        config_signature=signature,
    )
    db.add(pipeline)
    db.flush()
    for position, training_dataset in enumerate(training_datasets):
        db.add(
            models.TrainingPipelineDataset(
                training_pipeline_id=pipeline.id,
                training_dataset_id=training_dataset.id,
                position=position,
            )
        )
    db.commit()
    return get_training_pipeline(db, pipeline.id)  # type: ignore[return-value]


def update_training_pipeline(
    db: Session, pipeline_id: int, payload: TrainingPipelineCreate
) -> TrainingPipelineRead | None:
    pipeline = db.get(models.TrainingPipeline, pipeline_id)
    if pipeline is None:
        return None
    _raise_if_locked(training_pipeline_update_lock_reasons(db, pipeline_id))
    _assert_unique_training_pipeline_name(db, payload.name, exclude_id=pipeline_id)
    training_datasets, _, configuration = _resolve_training_pipeline_refs(db, payload)
    training_parameters = _merged_training_parameters(configuration, payload.training_parameters)
    signature = training_pipeline_signature(
        payload.training_dataset_ids,
        payload.preprocessing_pipeline_id,
        payload.method_configuration_id,
        payload.shuffle,
        training_parameters,
    )
    existing = db.scalar(
        select(models.TrainingPipeline).where(
            models.TrainingPipeline.config_signature == signature,
            models.TrainingPipeline.id != pipeline_id,
        )
    )
    if existing is not None:
        raise DuplicatePipelineError(existing)

    pipeline.name = payload.name
    pipeline.description = payload.description
    pipeline.preprocessing_pipeline_id = payload.preprocessing_pipeline_id
    pipeline.method_configuration_id = payload.method_configuration_id
    pipeline.shuffle = payload.shuffle
    pipeline.training_parameters = training_parameters
    pipeline.config_signature = signature
    # Replace entries wholesale; position always reflects the payload order.
    db.execute(
        delete(models.TrainingPipelineDataset).where(
            models.TrainingPipelineDataset.training_pipeline_id == pipeline_id
        )
    )
    db.flush()
    for position, training_dataset in enumerate(training_datasets):
        db.add(
            models.TrainingPipelineDataset(
                training_pipeline_id=pipeline_id,
                training_dataset_id=training_dataset.id,
                position=position,
            )
        )
    db.commit()
    return get_training_pipeline(db, pipeline_id)


def _training_pipeline_query():
    return select(models.TrainingPipeline).options(
        selectinload(models.TrainingPipeline.entries)
        .selectinload(models.TrainingPipelineDataset.training_dataset)
        .selectinload(models.TrainingDataset.rules)
        .selectinload(models.TrainingDatasetRule.folder)
        .selectinload(models.DatasetFolder.dataset),
        selectinload(models.TrainingPipeline.preprocessing_pipeline),
        selectinload(models.TrainingPipeline.method_configuration),
    )


def list_training_pipelines(db: Session) -> list[TrainingPipelineRead]:
    pipelines = list(
        db.scalars(_training_pipeline_query().order_by(models.TrainingPipeline.created_at.desc()))
    )
    count_cache: FolderTimestampCache = {}
    return [serialize_training_pipeline(db, pipeline, count_cache) for pipeline in pipelines]


def get_training_pipeline(db: Session, pipeline_id: int) -> TrainingPipelineRead | None:
    pipeline = db.scalar(_training_pipeline_query().where(models.TrainingPipeline.id == pipeline_id))
    if pipeline is None:
        return None
    return serialize_training_pipeline(db, pipeline, {})


def delete_training_pipeline(db: Session, pipeline_id: int) -> bool:
    pipeline = db.get(models.TrainingPipeline, pipeline_id)
    if pipeline is None:
        return False
    active_run = db.scalar(
        select(func.count(models.TrainingRun.id)).where(
            models.TrainingRun.training_pipeline_id == pipeline_id,
            models.TrainingRun.status.in_(["queued", "running"]),
        )
    ) or 0
    if active_run:
        raise ValueError(
            "This training pipeline has a queued or running training run. Abort it before deleting the pipeline."
        )
    db.delete(pipeline)
    db.commit()
    return True


def serialize_training_pipeline(
    db: Session,
    pipeline: models.TrainingPipeline,
    count_cache: FolderTimestampCache | None = None,
) -> TrainingPipelineRead:
    entries: list[TrainingPipelineDatasetRead] = []
    total_selected = 0
    for entry in pipeline.entries:
        summary = serialize_training_dataset(db, entry.training_dataset, count_cache)
        total_selected += summary.total_selected_images
        entries.append(
            TrainingPipelineDatasetRead(
                training_dataset_id=entry.training_dataset_id,
                position=entry.position,
                name=summary.name,
                total_selected_images=summary.total_selected_images,
                dataset_names=summary.dataset_names,
            )
        )

    lock_reasons = training_pipeline_update_lock_reasons(db, pipeline.id)
    return TrainingPipelineRead(
        id=pipeline.id,
        name=pipeline.name,
        description=pipeline.description,
        shuffle=pipeline.shuffle,
        training_parameters=pipeline.training_parameters,
        preprocessing_pipeline_id=pipeline.preprocessing_pipeline_id,
        preprocessing_pipeline_name=pipeline.preprocessing_pipeline.name,
        preprocessing_input_width=pipeline.preprocessing_pipeline.input_width,
        preprocessing_input_height=pipeline.preprocessing_pipeline.input_height,
        preprocessing_output_width=pipeline.preprocessing_pipeline.output_width,
        preprocessing_output_height=pipeline.preprocessing_pipeline.output_height,
        method_configuration_id=pipeline.method_configuration_id,
        method_configuration_name=pipeline.method_configuration.name,
        method_type=pipeline.method_configuration.method_type,
        training_mode=pipeline.method_configuration.training_mode,
        builder_kind=pipeline.method_configuration.builder_kind,
        total_selected_images=total_selected,
        training_datasets=entries,
        created_at=pipeline.created_at,
        updated_at=pipeline.updated_at,
        is_update_locked=bool(lock_reasons),
        update_lock_reasons=lock_reasons,
    )


def resolve_first_training_image(_db: Session, training_dataset: models.TrainingDataset):
    """Return the first concrete image selected by a train/test dataset."""
    from app.training.data import enumerate_training_dataset_image_records

    images = enumerate_training_dataset_image_records(training_dataset)
    if not images:
        raise ValueError(f"Training dataset '{training_dataset.name}' contains no matching image files.")
    return images[0], []


def dry_run_training_pipeline(db: Session, payload: TrainingPipelineDryRunRequest) -> TrainingPipelineDryRunResponse:
    """Push the first training image through preprocessing and the model architecture.

    The forward pass uses randomly initialized weights: it validates the
    composition (shapes, configs) end-to-end, not the model quality.
    """
    errors: list[str] = []
    warnings: list[str] = []
    logs: list[str] = []

    try:
        training_datasets, preprocessing_pipeline, configuration = _resolve_training_pipeline_refs(db, payload)
        training_parameters = _merged_training_parameters(configuration, payload.training_parameters)
    except ValueError as exc:
        return TrainingPipelineDryRunResponse(valid=False, mode="failed", errors=[str(exc)])

    logs.append(f"Training parameters: {training_parameters}")

    first_dataset = training_datasets[0]
    try:
        source_image, resolve_warnings = resolve_first_training_image(db, first_dataset)
    except ValueError as exc:
        return TrainingPipelineDryRunResponse(
            valid=False,
            mode="failed",
            errors=[str(exc)],
            logs=logs,
            training_dataset_name=first_dataset.name,
        )
    warnings.extend(resolve_warnings)
    logs.append(
        f"Using first image of training dataset '{first_dataset.name}': "
        f"{source_image.file_path} @ {source_image.timestamp_parsed.isoformat()}"
    )

    try:
        graph = PreprocessingGraph.model_validate(preprocessing_pipeline.graph)
        previews, final_image = execute_with_previews(graph, source_image.file_path)
    except (ValueError, FileNotFoundError) as exc:
        return TrainingPipelineDryRunResponse(
            valid=False,
            mode="failed",
            errors=[f"Preprocessing failed: {exc}"],
            warnings=warnings,
            logs=logs,
            training_dataset_name=first_dataset.name,
            source_image_path=source_image.file_path,
            source_timestamp=source_image.timestamp_parsed,
        )
    logs.append(f"Preprocessing produced {len(previews)} step outputs, final {previews[-1].width}x{previews[-1].height}")

    base_response = dict(
        warnings=warnings,
        logs=logs,
        training_dataset_name=first_dataset.name,
        source_image_path=source_image.file_path,
        source_timestamp=source_image.timestamp_parsed,
        preprocessing_previews=previews,
    )

    if configuration.builder_kind == "form":
        # Fit-style methods (e.g. mean image) have no forward pass to demo; the
        # preprocessed image is exactly what training would accumulate.
        logs.append("Method is fitted directly; skipping the forward pass.")
        return TrainingPipelineDryRunResponse(
            valid=True,
            mode="fit_contribution",
            errors=errors,
            note=(
                f"'{configuration.name}' is a fit-style method without a neural forward pass. "
                "The preprocessed image shown above is the first contribution that would be "
                "accumulated into the artifact during training."
            ),
            **base_response,
        )

    try:
        result = run_image_forward_pass(configuration.method_graph, configuration.method_config, final_image)
    except ValueError as exc:
        last = previews[-1]
        method_config = configuration.method_config or {}
        errors.append(str(exc))
        errors.append(
            f"Preprocessing output: {last.width}x{last.height} ({last.channels} channel(s)); method input: "
            f"{method_config.get('input_width', '?')}x{method_config.get('input_height', '?')} "
            f"({method_config.get('input_channels', '?')} channel(s))."
        )
        return TrainingPipelineDryRunResponse(valid=False, mode="failed", errors=errors, **base_response)

    warnings.extend(result["warnings"])
    logs.extend(result["logs"])
    output_image = result["output"]
    width, height, channels, dtype, value_min, value_max = image_metadata(output_image)
    return TrainingPipelineDryRunResponse(
        valid=True,
        mode="forward_pass",
        errors=errors,
        model_output=TrainingPipelineModelOutput(
            input_shape=result["input_shape"],
            output_shape=result["output_shape"],
            width=width,
            height=height,
            channels=channels,
            dtype=dtype,
            value_min=value_min,
            value_max=value_max,
            image_data_url=encode_png_data_url(output_image),
            elapsed_ms=result["elapsed_ms"],
        ),
        **base_response,
    )
