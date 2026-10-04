"""Resolve independent training/validation selections using existing dataset rules."""
from pathlib import Path
from types import SimpleNamespace

from app import models
from app.modeling.registry import registry
from app.modeling.training_ui import supports_validation
from app.training.data import enumerate_training_dataset_image_records, enumerate_training_pipeline_clip_samples


def datasets_by_ids(db, ids):
    datasets = [db.get(models.TrainingDataset, id) for id in ids]
    if any(dataset is None for dataset in datasets):
        raise ValueError('Ein ausgewählter Validierungsdatensatz existiert in diesem Projekt nicht.')
    return datasets


def canonical(path):
    return str(Path(path).resolve())


def sample_paths(samples, clips=False):
    return [frame.file_path for clip in samples for frame in (*clip.input_frames, *clip.future_frames)] if clips else list(samples)


def resolve_group(datasets, configuration):
    if configuration.builder_kind == 'spatiotemporal_autoencoder':
        proxy = SimpleNamespace(entries=[SimpleNamespace(position=i, training_dataset=d) for i, d in enumerate(datasets)])
        summary = enumerate_training_pipeline_clip_samples(proxy, configuration.method_config)
        unique = {}
        for clip in summary.clips:
            key = tuple(canonical(f.file_path) for f in (*clip.input_frames, *clip.future_frames))
            unique.setdefault(key, clip)
        return list(unique.values())
    unique = {}
    cache = {}
    for dataset in datasets:
        for image in enumerate_training_dataset_image_records(dataset, cache):
            unique.setdefault(canonical(image.file_path), image)
    return [image.file_path for image in sorted(unique.values(), key=lambda r: (r.timestamp_parsed, canonical(r.file_path)))]


def resolve_selections(db, training_datasets, configuration, mode, validation_ids):
    if mode == 'external' and not supports_validation(registry.get(configuration.method_type)):
        raise ValueError('Dieses Modell unterstützt keine separate Validierung.')
    training = resolve_group(training_datasets, configuration)
    validation = resolve_group(datasets_by_ids(db, validation_ids), configuration) if mode == 'external' else []
    if not training:
        raise ValueError('Die Trainingsauswahl enthält keine verwendbaren Bilder beziehungsweise Clips.')
    if mode == 'external' and not validation:
        raise ValueError('Die Validierungsauswahl enthält keine verwendbaren Bilder beziehungsweise Clips.')
    clips = configuration.builder_kind == 'spatiotemporal_autoencoder'
    shared = set(map(canonical, sample_paths(training, clips))) & set(map(canonical, sample_paths(validation, clips)))
    if shared:
        raise ValueError(f'Training und Validierung überschneiden sich in {len(shared)} Quelldateien, beispielsweise {next(iter(sorted(shared)))}.')
    return training, validation


def validation_signature(run, validation_sources=()):
    result = {'shuffle': run.shuffle}
    mode = getattr(run, 'validation_mode', None) or 'legacy_fraction'
    if mode != 'legacy_fraction':
        result.update(validation_mode=mode, validation_shuffle=bool(run.validation_shuffle),
                      validation_dataset_ids=list(run.validation_dataset_ids or []),
                      validation_sources=[canonical(path) for path in validation_sources])
    return result


def check_sample_shapes(training, validation, configuration, graph):
    """Dry run: check every distinct source, including clip targets."""
    from app.preprocessing.pipeline import compile_pipeline
    pipeline = compile_pipeline(graph)
    clips = configuration.builder_kind == 'spatiotemporal_autoencoder'
    expected = configuration.method_config or {}
    shape = None
    seen = set()
    for role, samples in [('Training', training), ('Validierung', validation)]:
        for path in sample_paths(samples, clips):
            source = canonical(path)
            if source in seen:
                continue
            seen.add(source)
            image = pipeline.run(path)
            actual = image.shape
            if shape is not None and actual != shape:
                raise ValueError(f'{role}: Preprocessing liefert unterschiedliche Bildgrößen oder Kanäle.')
            shape = actual
            h, w = actual[:2]
            channels = actual[2] if len(actual) == 3 else 1
            for key, value in [('input_width', w), ('input_height', h), ('input_channels', channels)]:
                if expected.get(key) is not None and int(expected[key]) != value:
                    raise ValueError(f'{role}: {key} ist {value}, das Modell erwartet {expected[key]}.')
