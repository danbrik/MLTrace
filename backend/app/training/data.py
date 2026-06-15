"""Resolving the concrete image file list of a training pipeline.

This generalizes the single-image query used by the dummy test
(``resolve_first_training_image``) to enumerate every indexed image selected by
a pipeline's training datasets, in a deterministic order, honoring each rule's
time window and stride.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models


def _sorted_rules(training_dataset: models.TrainingDataset) -> list[models.TrainingDatasetRule]:
    # Same deterministic ordering used by serialize_training_dataset.
    return sorted(
        training_dataset.rules,
        key=lambda rule: (
            rule.start_timestamp,
            rule.end_timestamp,
            rule.folder.dataset.name,
            rule.folder.relative_path,
            rule.id,
        ),
    )


def enumerate_training_pipeline_images(db: Session, pipeline: models.TrainingPipeline) -> list[str]:
    """Return the ordered, de-duplicated list of image file paths for a pipeline.

    Order: pipeline entries by position → each training dataset's rules in their
    deterministic order → images within the rule's time window ascending. Each
    rule's stride keeps every Nth indexed image.
    """
    paths: list[str] = []
    seen: set[str] = set()

    for entry in sorted(pipeline.entries, key=lambda item: item.position):
        training_dataset = entry.training_dataset
        for rule in _sorted_rules(training_dataset):
            rows = list(
                db.scalars(
                    select(models.DatasetImage)
                    .where(
                        models.DatasetImage.folder_id == rule.folder_id,
                        models.DatasetImage.timestamp_parsed >= rule.start_timestamp,
                        models.DatasetImage.timestamp_parsed <= rule.end_timestamp,
                    )
                    .order_by(
                        models.DatasetImage.timestamp_parsed.asc(),
                        models.DatasetImage.id.asc(),
                    )
                )
            )
            stride = max(1, rule.stride)
            for image in rows[::stride]:
                if image.file_path in seen:
                    continue
                seen.add(image.file_path)
                paths.append(image.file_path)

    return paths
