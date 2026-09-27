"""Shared selection and frozen source-file checks for image comparisons."""
from pathlib import Path
from sqlalchemy.orm import Session
from app import models
from app.preprocessing.pipeline import compile_pipeline
from app.schemas import PreprocessingGraph
from app.training.data import enumerate_training_dataset_image_records
from app.reference_image.engine import select_records


def prepare(db: Session, config):
    dataset = db.get(models.TrainingDataset, config.training_dataset_id)
    pipeline = db.get(models.PreprocessingPipeline, config.preprocessing_pipeline_id)
    if dataset is None or pipeline is None:
        raise ValueError("Datensatz oder Preprocessing-Pipeline nicht gefunden.")
    if not dataset.rules or any(rule.folder is None for rule in dataset.rules):
        raise ValueError("Der Datensatz enthält keine gültige vollständige Regelauswahl.")
    first = min(rule.start_timestamp for rule in dataset.rules)
    last = max(rule.end_timestamp for rule in dataset.rules)
    for interval in (config.reference, config.anomaly):
        if interval.start < first or interval.end > last:
            raise ValueError("Die Zeiträume müssen innerhalb der Datensatzgrenzen liegen.")
    compile_pipeline(PreprocessingGraph.model_validate(pipeline.graph))
    records = enumerate_training_dataset_image_records(dataset)
    samples, preview = select_records(records, config)
    return dataset, pipeline, samples, preview



def freeze_samples(samples):
    for group in samples.values():
        for sample in group:
            info = Path(sample["file_path"]).stat()
            sample["size_bytes"] = info.st_size
            sample["mtime_ns"] = info.st_mtime_ns


def read_frozen_image(sample, pipeline):
    path = Path(sample["file_path"])
    try:
        before = path.stat()
        if before.st_size != sample["size_bytes"] or before.st_mtime_ns != sample["mtime_ns"]:
            raise ValueError(f"Quelldatei seit Erstellung verändert: {path.name}")
        raw = pipeline.run(str(path))
        after = path.stat()
    except FileNotFoundError as exc:
        raise ValueError(f"Quelldatei nicht gefunden: {path.name}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Quelldatei während des Lesens verändert: {path.name}")
    return raw
