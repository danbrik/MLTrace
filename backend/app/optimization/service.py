from __future__ import annotations

import copy
import logging
import math
import random
import threading
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app import models, services
from app.database import SessionLocal, project_context
from app.projects import list_projects
from app.schemas import (
    MethodConfigurationCreate,
    OptimizationPromoteRequest,
    OptimizationSplitCreate,
    OptimizationSplitRead,
    OptimizationStudyCreate,
    OptimizationStudyRead,
    OptimizationStudyUpdate,
    OptimizationTrialRead,
    TrainingDatasetCreate,
    TrainingDatasetRuleInput,
    TrainingPipelineCreate,
    TestingRunCreate,
)
from app.testing.service import TestingConflict, enqueue_testing_run
from app.training import service as training_service
from app.training.service import RunConflict

logger = logging.getLogger("mltrace.optimization")

TERMINAL_RUN_STATUSES = {"finished", "failed", "aborted"}
ACTIVE_TRIAL_STATUSES = {"materializing", "training", "testing"}
POLL_INTERVAL_SECONDS = 2.0


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _study_query():
    return select(models.OptimizationStudy).options(
        selectinload(models.OptimizationStudy.trials),
        selectinload(models.OptimizationStudy.preprocessing_pipeline),
        selectinload(models.OptimizationStudy.normal_train_dataset),
        selectinload(models.OptimizationStudy.normal_validation_dataset),
        selectinload(models.OptimizationStudy.anomaly_validation_dataset),
        selectinload(models.OptimizationStudy.normal_holdout_dataset),
        selectinload(models.OptimizationStudy.anomaly_holdout_dataset),
    )


def _assert_unique_study_name(db: Session, name: str, exclude_id: int | None = None) -> None:
    query = select(models.OptimizationStudy).where(func.lower(models.OptimizationStudy.name) == name.lower())
    if exclude_id is not None:
        query = query.where(models.OptimizationStudy.id != exclude_id)
    if db.scalar(query) is not None:
        raise ValueError(f"An optimization study named '{name}' already exists.")


def _validate_study_refs(db: Session, payload: OptimizationStudyCreate | OptimizationStudyUpdate) -> None:
    if db.get(models.PreprocessingPipeline, payload.preprocessing_pipeline_id) is None:
        raise ValueError(f"Preprocessing pipeline does not exist: {payload.preprocessing_pipeline_id}")
    method_ids = set(payload.method_configuration_ids)
    found_methods = set(db.scalars(select(models.MethodConfiguration.id).where(models.MethodConfiguration.id.in_(method_ids))).all())
    missing_methods = sorted(method_ids - found_methods)
    if missing_methods:
        raise ValueError(f"Method configurations do not exist: {missing_methods}")
    dataset_ids = {
        payload.normal_train_dataset_id,
        payload.normal_validation_dataset_id,
        payload.anomaly_validation_dataset_id,
    }
    if payload.normal_holdout_dataset_id is not None:
        dataset_ids.add(payload.normal_holdout_dataset_id)
    if payload.anomaly_holdout_dataset_id is not None:
        dataset_ids.add(payload.anomaly_holdout_dataset_id)
    found_datasets = set(db.scalars(select(models.TrainingDataset.id).where(models.TrainingDataset.id.in_(dataset_ids))).all())
    missing_datasets = sorted(dataset_ids - found_datasets)
    if missing_datasets:
        raise ValueError(f"Train/Test datasets do not exist: {missing_datasets}")


def _serialize_trial(trial: models.OptimizationTrial) -> OptimizationTrialRead:
    return OptimizationTrialRead(
        id=trial.id,
        study_id=trial.study_id,
        number=trial.number,
        status=trial.status,
        phase=trial.phase,
        sampled_params=trial.sampled_params or {},
        method_configuration_id=trial.method_configuration_id,
        training_pipeline_id=trial.training_pipeline_id,
        training_run_id=trial.training_run_id,
        normal_testing_run_id=trial.normal_testing_run_id,
        anomaly_testing_run_id=trial.anomaly_testing_run_id,
        normal_holdout_testing_run_id=trial.normal_holdout_testing_run_id,
        anomaly_holdout_testing_run_id=trial.anomaly_holdout_testing_run_id,
        objective_value=trial.objective_value,
        metrics=trial.metrics,
        error_message=trial.error_message,
        created_at=trial.created_at,
        updated_at=trial.updated_at,
    )


def serialize_study(study: models.OptimizationStudy) -> OptimizationStudyRead:
    return OptimizationStudyRead(
        id=study.id,
        name=study.name,
        description=study.description,
        status=study.status,
        objective_name=study.objective_name,
        direction=study.direction,
        n_trials=study.n_trials,
        max_parallel_trials=study.max_parallel_trials,
        sampler=study.sampler,
        preprocessing_pipeline_id=study.preprocessing_pipeline_id,
        preprocessing_pipeline_name=study.preprocessing_pipeline.name if study.preprocessing_pipeline else "",
        method_configuration_ids=list(study.method_configuration_ids or []),
        normal_train_dataset_id=study.normal_train_dataset_id,
        normal_train_dataset_name=study.normal_train_dataset.name if study.normal_train_dataset else "",
        normal_validation_dataset_id=study.normal_validation_dataset_id,
        normal_validation_dataset_name=study.normal_validation_dataset.name if study.normal_validation_dataset else "",
        anomaly_validation_dataset_id=study.anomaly_validation_dataset_id,
        anomaly_validation_dataset_name=study.anomaly_validation_dataset.name if study.anomaly_validation_dataset else "",
        normal_holdout_dataset_id=study.normal_holdout_dataset_id,
        normal_holdout_dataset_name=study.normal_holdout_dataset.name if study.normal_holdout_dataset else None,
        anomaly_holdout_dataset_id=study.anomaly_holdout_dataset_id,
        anomaly_holdout_dataset_name=study.anomaly_holdout_dataset.name if study.anomaly_holdout_dataset else None,
        search_space=list(study.search_space or []),
        split_config=study.split_config or {},
        objective_config=study.objective_config or {},
        best_trial_id=study.best_trial_id,
        best_value=study.best_value,
        error_message=study.error_message,
        started_at=study.started_at,
        ended_at=study.ended_at,
        created_at=study.created_at,
        updated_at=study.updated_at,
        trials=[_serialize_trial(trial) for trial in sorted(study.trials, key=lambda item: item.number)],
    )


def list_studies(db: Session) -> list[OptimizationStudyRead]:
    studies = db.scalars(_study_query().order_by(models.OptimizationStudy.created_at.desc())).all()
    return [serialize_study(study) for study in studies]


def get_study(db: Session, study_id: int) -> OptimizationStudyRead | None:
    study = db.scalar(_study_query().where(models.OptimizationStudy.id == study_id))
    return serialize_study(study) if study else None


def create_study(db: Session, payload: OptimizationStudyCreate) -> OptimizationStudyRead:
    _assert_unique_study_name(db, payload.name)
    _validate_study_refs(db, payload)
    study = models.OptimizationStudy(
        name=payload.name,
        description=payload.description,
        preprocessing_pipeline_id=payload.preprocessing_pipeline_id,
        method_configuration_ids=list(payload.method_configuration_ids),
        normal_train_dataset_id=payload.normal_train_dataset_id,
        normal_validation_dataset_id=payload.normal_validation_dataset_id,
        anomaly_validation_dataset_id=payload.anomaly_validation_dataset_id,
        normal_holdout_dataset_id=payload.normal_holdout_dataset_id,
        anomaly_holdout_dataset_id=payload.anomaly_holdout_dataset_id,
        search_space=[item.model_dump() for item in payload.search_space],
        objective_name=payload.objective_name,
        direction=payload.direction,
        n_trials=payload.n_trials,
        max_parallel_trials=payload.max_parallel_trials,
        sampler=payload.sampler,
        split_config=payload.split_config,
        objective_config=payload.objective_config,
    )
    db.add(study)
    db.commit()
    return get_study(db, study.id)  # type: ignore[return-value]


def update_study(db: Session, study_id: int, payload: OptimizationStudyUpdate) -> OptimizationStudyRead | None:
    study = db.get(models.OptimizationStudy, study_id)
    if study is None:
        return None
    if study.status not in {"draft", "finished", "failed", "aborted"}:
        raise ValueError("Optimization study can only be edited while draft or terminal.")
    _assert_unique_study_name(db, payload.name, exclude_id=study_id)
    _validate_study_refs(db, payload)
    study.name = payload.name
    study.description = payload.description
    study.preprocessing_pipeline_id = payload.preprocessing_pipeline_id
    study.method_configuration_ids = list(payload.method_configuration_ids)
    study.normal_train_dataset_id = payload.normal_train_dataset_id
    study.normal_validation_dataset_id = payload.normal_validation_dataset_id
    study.anomaly_validation_dataset_id = payload.anomaly_validation_dataset_id
    study.normal_holdout_dataset_id = payload.normal_holdout_dataset_id
    study.anomaly_holdout_dataset_id = payload.anomaly_holdout_dataset_id
    study.search_space = [item.model_dump() for item in payload.search_space]
    study.objective_name = payload.objective_name
    study.direction = payload.direction
    study.n_trials = payload.n_trials
    study.max_parallel_trials = payload.max_parallel_trials
    study.sampler = payload.sampler
    study.split_config = payload.split_config
    study.objective_config = payload.objective_config
    if study.status in {"finished", "failed", "aborted"}:
        study.status = "draft"
        study.error_message = None
    db.commit()
    return get_study(db, study_id)


def delete_study(db: Session, study_id: int) -> bool:
    study = db.get(models.OptimizationStudy, study_id)
    if study is None:
        return False
    if study.status == "running":
        raise ValueError("Pause or abort the optimization study before deleting it.")
    db.delete(study)
    db.commit()
    return True


def _unique_name(db: Session, model, base: str) -> str:
    candidate = base[:255]
    index = 2
    while db.scalar(select(model.id).where(func.lower(model.name) == candidate.lower())) is not None:
        suffix = f" {index}"
        candidate = f"{base[:255 - len(suffix)]}{suffix}"
        index += 1
    return candidate


def _split_time(start: datetime, end: datetime, fraction: float) -> datetime:
    delta = end - start
    return start + timedelta(seconds=delta.total_seconds() * fraction)


def _split_rules_for_range(
    source: models.TrainingDataset,
    start_fraction: float,
    end_fraction: float,
) -> list[TrainingDatasetRuleInput]:
    rules: list[TrainingDatasetRuleInput] = []
    for rule in sorted(source.rules, key=lambda item: (item.start_timestamp, item.end_timestamp, item.id)):
        start = _split_time(rule.start_timestamp, rule.end_timestamp, start_fraction)
        end = _split_time(rule.start_timestamp, rule.end_timestamp, end_fraction)
        if end <= start:
            continue
        rules.append(
            TrainingDatasetRuleInput(
                folder_id=rule.folder_id,
                start_timestamp=start,
                end_timestamp=end,
                stride=rule.stride,
            )
        )
    if not rules:
        raise ValueError(f"Source Train/Test Dataset '{source.name}' produced no split rules.")
    return rules


def create_time_split(db: Session, payload: OptimizationSplitCreate) -> OptimizationSplitRead:
    normal = db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == payload.normal_source_dataset_id)
        .options(selectinload(models.TrainingDataset.rules))
    )
    anomaly = db.scalar(
        select(models.TrainingDataset)
        .where(models.TrainingDataset.id == payload.anomaly_source_dataset_id)
        .options(selectinload(models.TrainingDataset.rules))
    )
    if normal is None or anomaly is None:
        raise ValueError("Normal or anomaly source Train/Test Dataset does not exist.")
    train_end = payload.normal_train_fraction
    validation_end = payload.normal_train_fraction + payload.normal_validation_fraction
    anomaly_validation_end = payload.anomaly_validation_fraction

    def make_dataset(label: str, usage_label: str, rules: list[TrainingDatasetRuleInput]):
        return services.create_training_dataset(
            db,
            TrainingDatasetCreate(
                name=_unique_name(db, models.TrainingDataset, f"{payload.name_prefix} {label}"),
                usage_label=usage_label,
                notes=f"Optimization split from {normal.name if 'Normal' in label else anomaly.name}",
                rules=rules,
            ),
        )

    normal_train = make_dataset("Normal Train", "train", _split_rules_for_range(normal, 0.0, train_end))
    normal_validation = make_dataset("Normal Validation", "validation", _split_rules_for_range(normal, train_end, validation_end))
    normal_holdout = make_dataset("Normal Holdout", "test", _split_rules_for_range(normal, validation_end, 1.0))
    anomaly_validation = make_dataset("Anomaly Validation", "validation", _split_rules_for_range(anomaly, 0.0, anomaly_validation_end))
    anomaly_holdout = make_dataset("Anomaly Holdout", "test", _split_rules_for_range(anomaly, anomaly_validation_end, 1.0))
    return OptimizationSplitRead(
        normal_train_dataset=normal_train,
        normal_validation_dataset=normal_validation,
        normal_holdout_dataset=normal_holdout,
        anomaly_validation_dataset=anomaly_validation,
        anomaly_holdout_dataset=anomaly_holdout,
    )


def _parameter_distributions(search_space: list[dict], method_ids: list[int]) -> dict[str, Any]:
    distributions: dict[str, Any] = {"method_configuration_id": {"kind": "categorical", "choices": method_ids}}
    for item in search_space:
        path = str(item.get("path") or "").strip()
        if path:
            distributions[path] = item
    return distributions


def _random_sample(distributions: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    params: dict[str, Any] = {}
    for path, spec in distributions.items():
        kind = spec.get("kind")
        if kind == "categorical":
            choices = list(spec.get("choices") or [])
            if not choices:
                raise ValueError(f"Search parameter '{path}' has no choices.")
            params[path] = rng.choice(choices)
        elif kind == "int":
            low = int(spec.get("low"))
            high = int(spec.get("high"))
            step = int(spec.get("step") or 1)
            values = list(range(low, high + 1, step))
            params[path] = rng.choice(values)
        elif kind == "float":
            low = float(spec.get("low"))
            high = float(spec.get("high"))
            if bool(spec.get("log", False)):
                params[path] = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                params[path] = rng.uniform(low, high)
        else:
            raise ValueError(f"Unsupported search parameter kind for '{path}': {kind}")
    return params


def _optuna_sample(db: Session, study: models.OptimizationStudy, number: int) -> dict[str, Any]:
    distributions = _parameter_distributions(list(study.search_space or []), list(study.method_configuration_ids or []))
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    except Exception:
        return _random_sample(distributions, seed=study.id * 100_000 + number)

    optuna_distributions = {}
    for path, spec in distributions.items():
        kind = spec.get("kind")
        if kind == "categorical":
            optuna_distributions[path] = CategoricalDistribution(list(spec.get("choices") or []))
        elif kind == "int":
            optuna_distributions[path] = IntDistribution(
                int(spec.get("low")),
                int(spec.get("high")),
                step=int(spec.get("step") or 1),
                log=bool(spec.get("log", False)),
            )
        elif kind == "float":
            optuna_distributions[path] = FloatDistribution(
                float(spec.get("low")),
                float(spec.get("high")),
                step=spec.get("step"),
                log=bool(spec.get("log", False)),
            )
    sampler = optuna.samplers.RandomSampler(seed=study.id) if study.sampler == "random" else optuna.samplers.TPESampler(seed=study.id)
    optuna_study = optuna.create_study(direction=study.direction, sampler=sampler)
    for finished in db.scalars(
        select(models.OptimizationTrial).where(
            models.OptimizationTrial.study_id == study.id,
            models.OptimizationTrial.status == "finished",
            models.OptimizationTrial.objective_value.is_not(None),
        )
    ):
        try:
            optuna_study.add_trial(
                optuna.trial.create_trial(
                    params=finished.sampled_params or {},
                    distributions={key: optuna_distributions[key] for key in (finished.sampled_params or {}) if key in optuna_distributions},
                    value=float(finished.objective_value),
                )
            )
        except Exception:
            logger.debug("Could not replay Optuna trial %s", finished.id, exc_info=True)
    trial = optuna_study.ask(optuna_distributions)
    return dict(trial.params)


def _set_path(container: dict, path: str, value: Any) -> None:
    current = container
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _apply_sampled_params(
    base: models.MethodConfiguration,
    sampled_params: dict[str, Any],
) -> tuple[dict, dict, dict, dict]:
    method_graph = copy.deepcopy(base.method_graph or {})
    method_config = copy.deepcopy(base.method_config or {})
    training_config = copy.deepcopy(base.training_config or {})
    inference_config = copy.deepcopy(base.inference_config or {})
    for path, value in sampled_params.items():
        if path == "method_configuration_id":
            continue
        if path.startswith("method_config."):
            if path.removeprefix("method_config.").split(".")[0] not in method_config:
                continue
            _set_path(method_config, path.removeprefix("method_config."), value)
        elif path.startswith("training_config."):
            if path.removeprefix("training_config.").split(".")[0] not in training_config:
                continue
            _set_path(training_config, path.removeprefix("training_config."), value)
        elif path.startswith("training_parameters."):
            if path.removeprefix("training_parameters.").split(".")[0] not in training_config:
                continue
            _set_path(training_config, path.removeprefix("training_parameters."), value)
        elif path.startswith("inference_config."):
            if path.removeprefix("inference_config.").split(".")[0] not in inference_config:
                continue
            _set_path(inference_config, path.removeprefix("inference_config."), value)
        elif path.startswith("method_graph."):
            _set_path(method_graph, path.removeprefix("method_graph."), value)
        else:
            raise ValueError(f"Unsupported optimization parameter path: {path}")
    _sync_common_graph_params(method_graph, method_config)
    return method_graph, method_config, training_config, inference_config


def _sync_common_graph_params(method_graph: dict, method_config: dict) -> None:
    if "latent_dim" in method_config:
        latent_dim = int(method_config["latent_dim"])
        latent = method_graph.setdefault("latent", {})
        latent["latent_dim"] = latent_dim
        for layer in method_graph.get("encoder", []):
            if layer.get("type") == "Linear" and ("latent" in str(layer.get("id", "")) or layer is method_graph.get("encoder", [])[-1]):
                layer.setdefault("config", {})["out_features"] = latent_dim
    if "bottleneck_channels" in method_config:
        channels = int(method_config["bottleneck_channels"])
        latent = method_graph.setdefault("latent", {})
        latent["bottleneck_channels"] = channels
        for layer in method_graph.get("encoder", []):
            if layer.get("id") == "enc-spatial-bottleneck":
                layer.setdefault("config", {})["out_channels"] = channels


def _create_trial_objects(db: Session, study: models.OptimizationStudy, trial: models.OptimizationTrial) -> None:
    base_id = int((trial.sampled_params or {}).get("method_configuration_id"))
    base = db.get(models.MethodConfiguration, base_id)
    if base is None:
        raise ValueError(f"Base method configuration does not exist: {base_id}")
    method_graph, method_config, training_config, inference_config = _apply_sampled_params(base, trial.sampled_params or {})
    trial_label = f"{study.name} trial {trial.number:04d}"
    method = services.create_method_configuration(
        db,
        MethodConfigurationCreate(
            name=_unique_name(db, models.MethodConfiguration, f"{trial_label} method"),
            description=f"Generated by optimization study '{study.name}' from '{base.name}'.",
            method_type=base.method_type,
            method_graph=method_graph,
            method_config=method_config,
            training_config=training_config,
            inference_config=inference_config,
        ),
    )
    pipeline = services.create_training_pipeline(
        db,
        TrainingPipelineCreate(
            name=_unique_name(db, models.TrainingPipeline, f"{trial_label} pipeline"),
            description=f"Generated by optimization study '{study.name}'.",
            training_dataset_ids=[study.normal_train_dataset_id],
            preprocessing_pipeline_id=study.preprocessing_pipeline_id,
            method_configuration_id=method.id,
            shuffle=True,
            training_parameters={},
        ),
    )
    run = training_service.enqueue_training_run(db, pipeline.id)
    trial.method_configuration_id = method.id
    trial.training_pipeline_id = pipeline.id
    trial.training_run_id = run.id
    trial.status = "training"
    trial.phase = "training_queued"
    db.commit()


def _scores_for_testing_run(db: Session, testing_run_id: int) -> list[float]:
    return [
        float(value)
        for value in db.scalars(
            select(models.TestingRunResult.score)
            .where(models.TestingRunResult.testing_run_id == testing_run_id)
            .order_by(models.TestingRunResult.position)
        ).all()
    ]


def _roc_auc(normal_scores: list[float], anomaly_scores: list[float]) -> float | None:
    if not normal_scores or not anomaly_scores:
        return None
    pairs = [(score, 0) for score in normal_scores] + [(score, 1) for score in anomaly_scores]
    pairs.sort(key=lambda item: item[0])
    ranks = {id(item): index + 1 for index, item in enumerate(pairs)}
    pos_rank_sum = sum(ranks[id(item)] for item in pairs if item[1] == 1)
    n_pos = len(anomaly_scores)
    n_neg = len(normal_scores)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _pr_auc(normal_scores: list[float], anomaly_scores: list[float]) -> float | None:
    if not normal_scores or not anomaly_scores:
        return None
    pairs = sorted([(score, 0) for score in normal_scores] + [(score, 1) for score in anomaly_scores], reverse=True)
    total_pos = len(anomaly_scores)
    tp = 0
    fp = 0
    prev_recall = 0.0
    auc = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / max(1, tp + fp)
        auc += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return auc


def _evaluate_trial(db: Session, study: models.OptimizationStudy, trial: models.OptimizationTrial) -> None:
    if trial.normal_testing_run_id is None or trial.anomaly_testing_run_id is None:
        raise ValueError("Trial has no validation testing runs.")
    normal = _scores_for_testing_run(db, trial.normal_testing_run_id)
    anomaly = _scores_for_testing_run(db, trial.anomaly_testing_run_id)
    if not normal or not anomaly:
        raise ValueError("Validation testing runs produced no scores.")
    p95_normal = float(np.percentile(normal, 95))
    metrics = {
        "normal_count": len(normal),
        "anomaly_count": len(anomaly),
        "normal_mean": float(np.mean(normal)),
        "normal_median": float(np.median(normal)),
        "normal_p95": p95_normal,
        "anomaly_mean": float(np.mean(anomaly)),
        "anomaly_median": float(np.median(anomaly)),
        "anomaly_p05": float(np.percentile(anomaly, 5)),
        "mean_gap": float(np.mean(anomaly) - np.mean(normal)),
        "median_anomaly_minus_p95_normal": float(median(anomaly) - p95_normal),
        "roc_auc": _roc_auc(normal, anomaly),
        "pr_auc": _pr_auc(normal, anomaly),
    }
    objective_name = study.objective_name
    if objective_name == "normal_validation_loss":
        value = float(np.mean(normal))
    else:
        raw = metrics.get(objective_name)
        if raw is None:
            raise ValueError(f"Objective '{objective_name}' could not be computed.")
        value = float(raw)
    trial.objective_value = value
    trial.metrics = metrics
    trial.status = "finished"
    trial.phase = "finished"
    trial.error_message = None


def _update_best_trial(study: models.OptimizationStudy) -> None:
    finished = [trial for trial in study.trials if trial.status == "finished" and trial.objective_value is not None]
    if not finished:
        study.best_trial_id = None
        study.best_value = None
        return
    reverse = study.direction == "maximize"
    best = sorted(finished, key=lambda item: float(item.objective_value), reverse=reverse)[0]
    study.best_trial_id = best.id
    study.best_value = float(best.objective_value)


def _ensure_trials(db: Session, study: models.OptimizationStudy) -> None:
    existing_numbers = {trial.number for trial in study.trials}
    for number in range(study.n_trials):
        if number in existing_numbers:
            continue
        sampled = _optuna_sample(db, study, number)
        db.add(
            models.OptimizationTrial(
                study_id=study.id,
                number=number,
                sampled_params=sampled,
                status="waiting",
                phase="waiting",
            )
        )
    db.flush()


def start_study(db: Session, study_id: int) -> OptimizationStudyRead | None:
    study = db.scalar(_study_query().where(models.OptimizationStudy.id == study_id))
    if study is None:
        return None
    if study.status == "running":
        return serialize_study(study)
    if study.status not in {"draft", "paused", "finished", "failed", "aborted"}:
        raise ValueError(f"Cannot start study from status {study.status}.")
    if study.status in {"finished", "failed", "aborted"}:
        for trial in list(study.trials):
            db.delete(trial)
        db.flush()
        study.best_trial_id = None
        study.best_value = None
    _ensure_trials(db, study)
    study.status = "running"
    study.started_at = study.started_at or _utcnow()
    study.ended_at = None
    study.error_message = None
    db.commit()
    optimization_loop.wake()
    return get_study(db, study_id)


def pause_study(db: Session, study_id: int) -> OptimizationStudyRead | None:
    study = db.get(models.OptimizationStudy, study_id)
    if study is None:
        return None
    if study.status == "running":
        study.status = "paused"
        db.commit()
    return get_study(db, study_id)


def resume_study(db: Session, study_id: int) -> OptimizationStudyRead | None:
    return start_study(db, study_id)


def abort_study(db: Session, study_id: int) -> OptimizationStudyRead | None:
    study = db.scalar(_study_query().where(models.OptimizationStudy.id == study_id))
    if study is None:
        return None
    study.status = "aborted"
    study.ended_at = _utcnow()
    for trial in study.trials:
        if trial.status in {"waiting", "materializing"}:
            trial.status = "aborted"
            trial.phase = "aborted"
        if trial.training_run_id:
            run = db.get(models.TrainingRun, trial.training_run_id)
            if run is not None and run.status in {"queued", "running"}:
                try:
                    training_service.abort_training_run(db, run.id)
                except Exception:
                    logger.debug("Could not abort training run %s", run.id, exc_info=True)
        for testing_id in [trial.normal_testing_run_id, trial.anomaly_testing_run_id]:
            run = db.get(models.TestingRun, testing_id) if testing_id else None
            if run is not None and run.status in {"queued", "running"}:
                try:
                    from app.testing import service as testing_service

                    testing_service.abort_testing_run(db, run.id)
                except Exception:
                    logger.debug("Could not abort testing run %s", run.id, exc_info=True)
    db.commit()
    return get_study(db, study_id)


def promote_trial(db: Session, trial_id: int, payload: OptimizationPromoteRequest):
    trial = db.get(models.OptimizationTrial, trial_id)
    if trial is None:
        return None
    if trial.training_pipeline_id is None:
        raise ValueError("Trial has no generated training pipeline to promote.")
    original = services.get_training_pipeline(db, trial.training_pipeline_id)
    if original is None:
        raise ValueError("Trial training pipeline no longer exists.")
    return services.create_training_pipeline(
        db,
        TrainingPipelineCreate(
            name=payload.name,
            description=payload.description or f"Promoted from optimization trial {trial.number}.",
            training_dataset_ids=[entry.training_dataset_id for entry in original.training_datasets],
            preprocessing_pipeline_id=original.preprocessing_pipeline_id,
            method_configuration_id=original.method_configuration_id,
            shuffle=original.shuffle,
            training_parameters=original.training_parameters,
        ),
    )


def _enqueue_validation_runs(db: Session, study: models.OptimizationStudy, trial: models.OptimizationTrial, training_run: models.TrainingRun) -> None:
    for attr, dataset_id, label in [
        ("normal_testing_run_id", study.normal_validation_dataset_id, "normal validation"),
        ("anomaly_testing_run_id", study.anomaly_validation_dataset_id, "anomaly validation"),
    ]:
        if getattr(trial, attr):
            continue
        try:
            run = enqueue_testing_run(
                db,
                TestingRunCreate(
                    training_run_id=training_run.id,
                    training_dataset_id=dataset_id,
                    name=f"{study.name} trial {trial.number:04d} {label}",
                    inference_config=None,
                ),
            )
        except TestingConflict as conflict:
            run = conflict.existing
        setattr(trial, attr, run.id)
    trial.status = "testing"
    trial.phase = "validation_testing"
    db.commit()


def _process_trial(db: Session, study: models.OptimizationStudy, trial: models.OptimizationTrial) -> None:
    try:
        if trial.status == "waiting":
            trial.status = "materializing"
            trial.phase = "creating_pipeline"
            db.commit()
            _create_trial_objects(db, study, trial)
            return
        if trial.status == "training":
            run = db.get(models.TrainingRun, trial.training_run_id) if trial.training_run_id else None
            if run is None:
                raise ValueError("Training run disappeared.")
            if run.status in {"queued", "running"}:
                trial.phase = f"training_{run.status}"
                return
            if run.status != "finished":
                raise ValueError(run.error_message or f"Training run ended with status {run.status}.")
            _enqueue_validation_runs(db, study, trial, run)
            return
        if trial.status == "testing":
            runs = [
                db.get(models.TestingRun, trial.normal_testing_run_id) if trial.normal_testing_run_id else None,
                db.get(models.TestingRun, trial.anomaly_testing_run_id) if trial.anomaly_testing_run_id else None,
            ]
            if any(run is None for run in runs):
                raise ValueError("Validation testing run disappeared.")
            if any(run.status in {"queued", "running"} for run in runs if run):
                trial.phase = "validation_testing_running"
                return
            failed = [run for run in runs if run and run.status != "finished"]
            if failed:
                raise ValueError("; ".join(run.error_message or f"Testing run {run.id} ended with {run.status}" for run in failed))
            trial.phase = "evaluating"
            _evaluate_trial(db, study, trial)
            _update_best_trial(study)
            db.commit()
    except Exception as exc:
        trial.status = "failed"
        trial.phase = "failed"
        trial.error_message = str(exc)
        db.commit()


class OptimizationLoop:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="optimization-loop", daemon=True)
        self._thread.start()
        logger.info("Optimization loop started")

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Optimization loop tick failed")
            self._wake.wait(timeout=POLL_INTERVAL_SECONDS)
            self._wake.clear()

    def tick(self) -> None:
        for project in list_projects():
            with project_context(project.database_url, project.artifact_dir):
                db = SessionLocal()
                try:
                    for study in db.scalars(_study_query().where(models.OptimizationStudy.status == "running")).all():
                        _ensure_trials(db, study)
                        active = sum(1 for trial in study.trials if trial.status in ACTIVE_TRIAL_STATUSES)
                        for trial in sorted(study.trials, key=lambda item: item.number):
                            if trial.status == "waiting" and active >= study.max_parallel_trials:
                                break
                            previous = trial.status
                            if trial.status in {"waiting", "materializing", "training", "testing"}:
                                _process_trial(db, study, trial)
                            if previous == "waiting" and trial.status in ACTIVE_TRIAL_STATUSES:
                                active += 1
                        db.refresh(study)
                        _update_best_trial(study)
                        if study.trials and all(trial.status in {"finished", "failed", "aborted"} for trial in study.trials):
                            study.status = "finished" if any(trial.status == "finished" for trial in study.trials) else "failed"
                            study.ended_at = _utcnow()
                        db.commit()
                finally:
                    db.close()


optimization_loop = OptimizationLoop()
