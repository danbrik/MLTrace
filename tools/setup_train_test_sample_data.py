from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from generate_sample_datasets import main as generate_sample_datasets


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
SAMPLE_ROOT = REPO_ROOT / "sample_datasets"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import models, services  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.schemas import TrainingDatasetCreate, TrainingDatasetRuleInput  # noqa: E402


TIMESTAMP_REGEX = r"(?P<timestamp>\d{8}_\d{6})"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


DATASETS = (
    ("Sample Line A 96x96", "sample_line_a_96x96", "train", "Sample train set: 96x96 grayscale TIFF."),
    ("Sample Line B 96x96", "sample_line_b_96x96", "test", "Sample test set: 96x96 grayscale TIFF."),
    ("Sample Line C 128x96", "sample_line_c_128x96", "test", "Incompatible sample set: 128x96 grayscale TIFF."),
)


def _dataset_ids_under_sample_root(db) -> list[int]:
    sample_prefix = f"{SAMPLE_ROOT.resolve()}%"
    return list(
        db.scalars(
            select(models.Dataset.id).where(models.Dataset.root_path.like(sample_prefix))
        )
    )


def _training_dataset_ids_for_datasets(db, dataset_ids: list[int]) -> list[int]:
    if not dataset_ids:
        return []
    folder_ids = list(
        db.scalars(
            select(models.DatasetFolder.id).where(models.DatasetFolder.dataset_id.in_(dataset_ids))
        )
    )
    if not folder_ids:
        return []
    return sorted(
        set(
            db.scalars(
                select(models.TrainingDatasetRule.training_dataset_id).where(
                    models.TrainingDatasetRule.folder_id.in_(folder_ids)
                )
            )
        )
    )


def _training_pipeline_ids_for_training_datasets(db, training_dataset_ids: list[int]) -> list[int]:
    if not training_dataset_ids:
        return []
    return sorted(
        set(
            db.scalars(
                select(models.TrainingPipelineDataset.training_pipeline_id).where(
                    models.TrainingPipelineDataset.training_dataset_id.in_(training_dataset_ids)
                )
            )
        )
    )


def clear_old_sample_records(db) -> None:
    """Remove only database rows that point into the local sample_datasets tree."""
    dataset_ids = _dataset_ids_under_sample_root(db)
    training_dataset_ids = _training_dataset_ids_for_datasets(db, dataset_ids)
    training_pipeline_ids = _training_pipeline_ids_for_training_datasets(db, training_dataset_ids)

    if training_pipeline_ids:
        db.execute(delete(models.TrainingRun).where(models.TrainingRun.training_pipeline_id.in_(training_pipeline_ids)))
        db.execute(
            delete(models.TrainingPipelineDataset).where(
                models.TrainingPipelineDataset.training_pipeline_id.in_(training_pipeline_ids)
            )
        )
        db.execute(delete(models.TrainingPipeline).where(models.TrainingPipeline.id.in_(training_pipeline_ids)))

    if training_dataset_ids:
        db.execute(
            delete(models.TrainingDatasetRule).where(
                models.TrainingDatasetRule.training_dataset_id.in_(training_dataset_ids)
            )
        )
        db.execute(delete(models.TrainingDataset).where(models.TrainingDataset.id.in_(training_dataset_ids)))

    if dataset_ids:
        db.execute(delete(models.DatasetImage).where(models.DatasetImage.dataset_id.in_(dataset_ids)))
        db.execute(delete(models.DatasetFolder).where(models.DatasetFolder.dataset_id.in_(dataset_ids)))
        db.execute(delete(models.Dataset).where(models.Dataset.id.in_(dataset_ids)))

    db.commit()


def register_dataset(db, name: str, folder_name: str) -> models.Dataset:
    dataset = services.create_dataset(db, name, str(SAMPLE_ROOT / folder_name))
    services.scan_dataset(db, dataset, TIMESTAMP_REGEX, TIMESTAMP_FORMAT)
    return db.scalar(
        select(models.Dataset)
        .where(models.Dataset.id == dataset.id)
        .options(selectinload(models.Dataset.folders))
    )


def create_train_test_dataset(db, dataset: models.Dataset, usage_label: str, notes: str):
    if len(dataset.folders) != 1:
        raise RuntimeError(f"Expected one direct folder for sample dataset {dataset.name}.")
    folder = dataset.folders[0]
    return services.create_training_dataset(
        db,
        TrainingDatasetCreate(
            name=dataset.name.replace("Sample Line", "TTD Line"),
            usage_label=usage_label,
            notes=notes,
            rules=[
                TrainingDatasetRuleInput(
                    folder_id=folder.id,
                    start_timestamp=folder.first_timestamp,
                    end_timestamp=folder.last_timestamp,
                    stride=1,
                )
            ],
        ),
    )


def main() -> None:
    generate_sample_datasets()
    with SessionLocal() as db:
        clear_old_sample_records(db)
        created = []
        for name, folder_name, usage_label, notes in DATASETS:
            dataset = register_dataset(db, name, folder_name)
            train_test_dataset = create_train_test_dataset(db, dataset, usage_label, notes)
            signature = train_test_dataset.image_signatures[0] if train_test_dataset.image_signatures else "n/a"
            created.append((dataset.root_path, train_test_dataset.name, usage_label, signature))

    print("Prepared train/test sample data:")
    for root_path, train_test_name, usage_label, signature in created:
        print(f"- {train_test_name} [{usage_label}]")
        print(f"  path: {root_path}")
        print(f"  image data: {signature}")


if __name__ == "__main__":
    main()
