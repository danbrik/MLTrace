from __future__ import annotations

import math
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app import models
from app.preprocessing.pipeline import execute_preview, validate_linear_graph
from app.preprocessing.registry import registry
from app.scanner import detect_timestamp_pattern, iter_tiff_files, scan_dataset_files
from app.schemas import (
    DatasetConnectionTestResponse,
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
)


def create_dataset(db: Session, name: str, root_path: str) -> models.Dataset:
    root = Path(root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {root}")

    resolved_root = str(root)
    pattern = detect_timestamp_pattern(resolved_root)
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
    return dataset


def test_dataset_connection(root_path: str) -> DatasetConnectionTestResponse:
    root = Path(root_path).expanduser().resolve(strict=False)
    if not root.exists():
        return DatasetConnectionTestResponse(
            root_path=str(root),
            exists=False,
            is_directory=False,
            supported_file_found=False,
            sample_file_path=None,
            message=f"Dataset path does not exist: {root}",
        )
    if not root.is_dir():
        return DatasetConnectionTestResponse(
            root_path=str(root),
            exists=True,
            is_directory=False,
            supported_file_found=False,
            sample_file_path=None,
            message=f"Dataset path is not a directory: {root}",
        )

    files = iter_tiff_files(root)
    if files:
        sample_file = files[0]
        return DatasetConnectionTestResponse(
            root_path=str(root),
            exists=True,
            is_directory=True,
            supported_file_found=True,
            sample_file_path=str(sample_file),
            message=f"Path is reachable. Found supported image: {sample_file}",
        )

    message = f"Path is reachable, but no supported TIFF files were found directly under the root or its first-level folders: {root}"
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
    dataset.timestamp_regex = timestamp_regex
    dataset.timestamp_format = timestamp_format
    dataset.status = "scanning"
    dataset.scan_error = None
    db.commit()

    try:
        scanned_images, folder_summaries, scan_summary = scan_dataset_files(
            dataset.root_path, timestamp_regex, timestamp_format
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

    db.commit()
    db.refresh(dataset)
    return get_dataset_or_404(db, dataset.id) or dataset


def list_datasets(db: Session) -> list[models.Dataset]:
    return list(
        db.scalars(
            select(models.Dataset)
            .order_by(models.Dataset.created_at.desc())
            .options(selectinload(models.Dataset.folders))
        )
    )


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
            "Dataset is used by saved training datasets. Delete those training datasets before deleting the dataset."
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

    for rule in request.rules:
        folder = db.get(models.DatasetFolder, rule.folder_id)
        if folder is None:
            raise ValueError(f"Folder does not exist: {rule.folder_id}")
        matching = count_images_in_folder_range(folder, rule.start_timestamp, rule.end_timestamp)
        selected = math.ceil(matching / rule.stride) if matching else 0
        previews.append(
            TrainingDatasetRulePreview(
                folder_id=rule.folder_id,
                start_timestamp=rule.start_timestamp,
                end_timestamp=rule.end_timestamp,
                stride=rule.stride,
                matching_images=matching,
                selected_images=selected,
            )
        )
        total_matching += matching
        total_selected += selected

    return TrainingDatasetPreviewResponse(
        total_matching_images=total_matching,
        total_selected_images=total_selected,
        rules=previews,
    )


def create_training_dataset(db: Session, payload: TrainingDatasetCreate) -> TrainingDatasetRead:
    validate_rules(db, payload.rules)

    training_dataset = models.TrainingDataset(
        name=payload.name,
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
    return [serialize_training_dataset(db, training_dataset) for training_dataset in training_datasets]


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
    return serialize_training_dataset(db, training_dataset)


def delete_training_dataset(db: Session, training_dataset_id: int) -> bool:
    training_dataset = db.get(models.TrainingDataset, training_dataset_id)
    if training_dataset is None:
        return False
    db.delete(training_dataset)
    db.commit()
    return True


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

    for rule in rules:
        folder = folders[rule.folder_id]
        if folder.first_timestamp is None or folder.last_timestamp is None:
            raise ValueError(f"Folder {folder.id} has no timestamp bounds.")
        if rule.start_timestamp < folder.first_timestamp or rule.end_timestamp > folder.last_timestamp:
            raise ValueError(
                f"Rule for folder {folder.relative_path} must stay within "
                f"{folder.first_timestamp.isoformat()} and {folder.last_timestamp.isoformat()}."
            )


def count_rule_images(db: Session, rule: models.TrainingDatasetRule) -> tuple[int, int]:
    matching = count_images_in_folder_range(rule.folder, rule.start_timestamp, rule.end_timestamp)
    selected = math.ceil(matching / rule.stride) if matching else 0
    return matching, selected


def count_images_in_folder_range(folder: models.DatasetFolder, start_timestamp, end_timestamp) -> int:
    if folder.first_timestamp is None or folder.last_timestamp is None:
        return 0

    range_start = max(start_timestamp, folder.first_timestamp)
    range_end = min(end_timestamp, folder.last_timestamp)
    if range_end < range_start:
        return 0

    image_count = folder.image_count
    if image_count <= 1:
        return image_count if folder.first_timestamp >= range_start and folder.first_timestamp <= range_end else 0

    cadence_seconds = None
    if folder.cadence_summary:
        cadence_seconds = folder.cadence_summary.get("median_seconds")
    if not cadence_seconds or cadence_seconds <= 0:
        if range_start <= folder.first_timestamp and range_end >= folder.last_timestamp:
            return image_count
        return 0

    start_offset = max(0, math.ceil((range_start - folder.first_timestamp).total_seconds() / cadence_seconds))
    end_offset = min(
        image_count - 1,
        math.floor((range_end - folder.first_timestamp).total_seconds() / cadence_seconds),
    )
    if end_offset < start_offset:
        return 0
    return end_offset - start_offset + 1


def serialize_training_dataset(db: Session, training_dataset: models.TrainingDataset) -> TrainingDatasetRead:
    rule_reads: list[TrainingDatasetRuleRead] = []
    dataset_names: set[str] = set()
    total_matching = 0
    total_selected = 0

    for rule in sorted(
        training_dataset.rules,
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
        matching, selected = count_rule_images(db, rule)
        total_matching += matching
        total_selected += selected
        dataset_names.add(dataset.name)
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
        notes=training_dataset.notes,
        created_at=training_dataset.created_at,
        dataset_names=sorted(dataset_names),
        total_matching_images=total_matching,
        total_selected_images=total_selected,
        rules=rule_reads,
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
    return PreprocessingPipelineRead.model_validate(pipeline)


def update_preprocessing_pipeline(
    db: Session, pipeline_id: int, payload: PreprocessingPipelineCreate
) -> PreprocessingPipelineRead | None:
    pipeline = db.get(models.PreprocessingPipeline, pipeline_id)
    if pipeline is None:
        return None
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
    return PreprocessingPipelineRead.model_validate(pipeline)


def list_preprocessing_pipelines(db: Session) -> list[models.PreprocessingPipeline]:
    return list(db.scalars(select(models.PreprocessingPipeline).order_by(models.PreprocessingPipeline.created_at.desc())))


def get_preprocessing_pipeline(db: Session, pipeline_id: int) -> models.PreprocessingPipeline | None:
    return db.get(models.PreprocessingPipeline, pipeline_id)


def delete_preprocessing_pipeline(db: Session, pipeline_id: int) -> bool:
    pipeline = db.get(models.PreprocessingPipeline, pipeline_id)
    if pipeline is None:
        return False
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
