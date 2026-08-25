"""Declarative entity registry for the Data Manager.

One :class:`EntitySpec` per user-facing object type describes how to list,
search, filter, inspect, and delete it — including its direct dependents (the
RESTRICT/usage edges) and its on-disk artifacts. The service layer walks these
specs generically, so adding a new entity type is a matter of adding a spec.

Deletion delegates to the existing ``delete_*`` service functions wherever they
exist (they already guard running jobs and clean their disk directories), so the
Data Manager never duplicates deletion semantics.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import models
from app.database import data_dir


@dataclass(frozen=True)
class Dependent:
    entity_type: str
    id: int
    name: str


@dataclass(frozen=True)
class FilterSpec:
    """A typed list filter. ``kind`` drives the frontend widget.

    - ``select``: options from static list or DISTINCT over ``column``
    - ``daterange``: two datetime bounds applied to ``column``
    - ``usage``: used/unused, resolved via the spec's dependents edges
    """

    key: str
    label: str
    kind: str  # "select" | "daterange" | "usage"
    column: str | None = None
    options: list[str] | None = None  # static options; None => distinct over column


@dataclass(frozen=True)
class EntitySpec:
    key: str
    label: str
    model: type
    name_of: Callable[[object], str]
    list_fields: list[str]
    search_fields: list[str]
    filters: list[FilterSpec] = field(default_factory=list)
    # Direct dependents (children that must be deleted first / block deletion).
    dependents: Callable[[Session, int], list[Dependent]] | None = None
    # Disk artifacts (dirs or files) owned by one row.
    artifacts: Callable[[Session, object], list[Path]] | None = None
    # Existing delete function; must clean its own disk artifacts.
    deleter: Callable[[Session, int], bool] | None = None
    # Reasons the row cannot be deleted right now (e.g. running job).
    blockers: Callable[[object], list[str]] | None = None
    # Heavy/legacy columns excluded from the full-detail payload.
    detail_exclude: frozenset[str] = frozenset()
    # Rough per-row disk size when cheaply known from columns.
    size_of: Callable[[object], int | None] | None = None


_ACTIVE = ("queued", "running")


def _job_blockers(row) -> list[str]:
    status = getattr(row, "status", None)
    if status in _ACTIVE:
        return [f"Job is {status}. Abort it before deleting."]
    return []


def _named(db: Session, entity_type: str, model, id_column, name_getter, where) -> list[Dependent]:
    rows = db.scalars(select(model).where(where)).all()
    return [Dependent(entity_type, int(getattr(r, "id")), name_getter(r)) for r in rows]


# -- dependents resolvers ------------------------------------------------------


def _dataset_dependents(db: Session, dataset_id: int) -> list[Dependent]:
    rows = db.scalars(
        select(models.TrainingDataset)
        .join(models.TrainingDatasetRule, models.TrainingDatasetRule.training_dataset_id == models.TrainingDataset.id)
        .join(models.DatasetFolder, models.TrainingDatasetRule.folder_id == models.DatasetFolder.id)
        .where(models.DatasetFolder.dataset_id == dataset_id)
        .distinct()
    ).all()
    return [Dependent("training_dataset", r.id, r.name) for r in rows]


def _training_dataset_dependents(db: Session, td_id: int) -> list[Dependent]:
    out: list[Dependent] = []
    pipelines = db.scalars(
        select(models.TrainingPipeline)
        .join(
            models.TrainingPipelineDataset,
            models.TrainingPipelineDataset.training_pipeline_id == models.TrainingPipeline.id,
        )
        .where(models.TrainingPipelineDataset.training_dataset_id == td_id)
        .distinct()
    ).all()
    out += [Dependent("training_pipeline", p.id, p.name) for p in pipelines]
    out += _named(
        db, "testing_run", models.TestingRun, models.TestingRun.id,
        lambda r: r.name, models.TestingRun.training_dataset_id == td_id,
    )
    out += _named(
        db, "inspect_run", models.InspectRun, models.InspectRun.id,
        lambda r: f"Inspect #{r.id} · {r.training_dataset_name}",
        models.InspectRun.training_dataset_id == td_id,
    )
    study_where = (
        (models.OptimizationStudy.normal_train_dataset_id == td_id)
        | (models.OptimizationStudy.normal_validation_dataset_id == td_id)
        | (models.OptimizationStudy.anomaly_validation_dataset_id == td_id)
        | (models.OptimizationStudy.normal_holdout_dataset_id == td_id)
        | (models.OptimizationStudy.anomaly_holdout_dataset_id == td_id)
    )
    out += _named(db, "optimization_study", models.OptimizationStudy, models.OptimizationStudy.id, lambda r: r.name, study_where)
    out += _named(
        db, "evaluation_label_set", models.EvaluationLabelSet, models.EvaluationLabelSet.id,
        lambda r: r.name, models.EvaluationLabelSet.training_dataset_id == td_id,
    )
    out += _named(db, "evaluation_separation_layout", models.EvaluationSeparationLayout,
                  models.EvaluationSeparationLayout.id, lambda r: r.name,
                  models.EvaluationSeparationLayout.training_dataset_id == td_id)
    out += _named(db, "evaluation_drift_layout", models.EvaluationDriftLayout,
                  models.EvaluationDriftLayout.id, lambda r: r.name,
                  models.EvaluationDriftLayout.training_dataset_id == td_id)
    return out


def _preprocessing_dependents(db: Session, pp_id: int) -> list[Dependent]:
    out = _named(
        db, "training_pipeline", models.TrainingPipeline, models.TrainingPipeline.id,
        lambda r: r.name, models.TrainingPipeline.preprocessing_pipeline_id == pp_id,
    )
    out += _named(
        db, "inspect_run", models.InspectRun, models.InspectRun.id,
        lambda r: f"Inspect #{r.id} · {r.preprocessing_pipeline_name}",
        models.InspectRun.preprocessing_pipeline_id == pp_id,
    )
    out += _named(
        db, "optimization_study", models.OptimizationStudy, models.OptimizationStudy.id,
        lambda r: r.name, models.OptimizationStudy.preprocessing_pipeline_id == pp_id,
    )
    return out


def _method_dependents(db: Session, mc_id: int) -> list[Dependent]:
    return _named(
        db, "training_pipeline", models.TrainingPipeline, models.TrainingPipeline.id,
        lambda r: r.name, models.TrainingPipeline.method_configuration_id == mc_id,
    )


def _training_pipeline_dependents(db: Session, tp_id: int) -> list[Dependent]:
    return _named(
        db, "training_run", models.TrainingRun, models.TrainingRun.id,
        lambda r: f"Run #{r.id} · {r.training_pipeline_name}",
        models.TrainingRun.training_pipeline_id == tp_id,
    )


def _training_run_dependents(db: Session, tr_id: int) -> list[Dependent]:
    out = _named(
        db, "testing_run", models.TestingRun, models.TestingRun.id,
        lambda r: r.name, models.TestingRun.training_run_id == tr_id,
    )
    out += _named(db, "evaluation_workspace", models.EvaluationModelWorkspace,
                  models.EvaluationModelWorkspace.id, lambda r: f"Evaluation workspace #{r.id}",
                  models.EvaluationModelWorkspace.training_run_id == tr_id)
    return out


def _testing_run_dependents(db: Session, ts_id: int) -> list[Dependent]:
    out = _named(
        db, "heatmap", models.HeatmapRun, models.HeatmapRun.id,
        lambda r: f"Heatmap #{r.id} · {r.timestamp:%Y-%m-%d %H:%M:%S}",
        models.HeatmapRun.testing_run_id == ts_id,
    )
    out += _named(
        db, "heatmap_range", models.HeatmapRangeRun, models.HeatmapRangeRun.id,
        lambda r: f"Heatmap video #{r.id} · {r.testing_run_name}",
        models.HeatmapRangeRun.testing_run_id == ts_id,
    )
    evaluations = db.scalars(
        select(models.ModelEvaluation).where(
            (models.ModelEvaluation.evaluation_testing_run_id == ts_id)
            | (models.ModelEvaluation.reference_testing_run_id == ts_id)
            | (models.ModelEvaluation.calibration_testing_run_id == ts_id)
        ).distinct()
    ).all()
    out += [Dependent("model_evaluation", row.id, row.name) for row in evaluations]
    out += _named(db, "evaluation_separation_calculation", models.EvaluationSeparationCalculation,
                  models.EvaluationSeparationCalculation.id, lambda r: f"A calculation #{r.id}",
                  models.EvaluationSeparationCalculation.testing_run_id == ts_id)
    out += _named(db, "evaluation_drift_calculation", models.EvaluationDriftCalculation,
                  models.EvaluationDriftCalculation.id, lambda r: f"B calculation #{r.id}",
                  models.EvaluationDriftCalculation.testing_run_id == ts_id)
    return out


def _workspace_dependents(db: Session, workspace_id: int) -> list[Dependent]:
    out = _named(db, "evaluation_separation_calculation", models.EvaluationSeparationCalculation,
                 models.EvaluationSeparationCalculation.id, lambda r: f"A calculation #{r.id}",
                 models.EvaluationSeparationCalculation.workspace_id == workspace_id)
    out += _named(db, "evaluation_drift_calculation", models.EvaluationDriftCalculation,
                  models.EvaluationDriftCalculation.id, lambda r: f"B calculation #{r.id}",
                  models.EvaluationDriftCalculation.workspace_id == workspace_id)
    return out


def _separation_layout_dependents(db: Session, layout_id: int) -> list[Dependent]:
    return _named(db, "evaluation_separation_calculation", models.EvaluationSeparationCalculation,
                  models.EvaluationSeparationCalculation.id, lambda r: f"A calculation #{r.id}",
                  models.EvaluationSeparationCalculation.layout_id == layout_id)


def _drift_layout_dependents(db: Session, layout_id: int) -> list[Dependent]:
    return _named(db, "evaluation_drift_calculation", models.EvaluationDriftCalculation,
                  models.EvaluationDriftCalculation.id, lambda r: f"B calculation #{r.id}",
                  models.EvaluationDriftCalculation.layout_id == layout_id)


def _evaluation_profile_dependents(db: Session, profile_id: int) -> list[Dependent]:
    return _named(
        db, "model_evaluation", models.ModelEvaluation, models.ModelEvaluation.id,
        lambda row: row.name, models.ModelEvaluation.profile_id == profile_id,
    )


def _evaluation_label_set_dependents(db: Session, label_set_id: int) -> list[Dependent]:
    return _named(
        db, "model_evaluation", models.ModelEvaluation, models.ModelEvaluation.id,
        lambda row: row.name, models.ModelEvaluation.label_set_id == label_set_id,
    )


def _redundancy_source_dependents(db: Session, source_id: int) -> list[Dependent]:
    return _named(
        db, "redundancy_analysis", models.RedundancyAnalysis, models.RedundancyAnalysis.id,
        lambda row: row.name, models.RedundancyAnalysis.source_id == source_id,
    )


# -- artifact resolvers --------------------------------------------------------


def _training_run_artifacts(_db: Session, row) -> list[Path]:
    return [data_dir() / "runs" / str(row.id)]


def _testing_run_artifacts(_db: Session, row) -> list[Path]:
    return [data_dir() / "testing_runs" / str(row.id)]


def _heatmap_artifacts(_db: Session, row) -> list[Path]:
    return [Path(row.artifacts_dir)] if row.artifacts_dir else []


def _heatmap_range_artifacts(_db: Session, row) -> list[Path]:
    if row.frames_dir:
        return [Path(row.frames_dir)]
    return [data_dir() / "heatmap_ranges" / str(row.id)]


def _inspect_artifacts(_db: Session, row) -> list[Path]:
    return [data_dir() / "inspect_runs" / str(row.id)]


def _redundancy_source_artifacts(_db: Session, row) -> list[Path]:
    return [Path(row.artifact_path)]


# -- deleters (wrappers around existing service functions) ----------------------


def _delete_dataset(db: Session, entity_id: int) -> bool:
    from app.services import delete_dataset

    return delete_dataset(db, entity_id)


def _delete_training_dataset(db: Session, entity_id: int) -> bool:
    from app.services import delete_training_dataset

    return delete_training_dataset(db, entity_id)


def _delete_preprocessing(db: Session, entity_id: int) -> bool:
    from app.services import delete_preprocessing_pipeline

    return delete_preprocessing_pipeline(db, entity_id)


def _delete_method(db: Session, entity_id: int) -> bool:
    from app.services import delete_method_configuration

    return delete_method_configuration(db, entity_id)


def _delete_training_pipeline(db: Session, entity_id: int) -> bool:
    from app.services import delete_training_pipeline

    return delete_training_pipeline(db, entity_id)


def _delete_training_run(db: Session, entity_id: int) -> bool:
    from app.training.service import delete_training_run

    return delete_training_run(db, entity_id)


def _delete_testing_run(db: Session, entity_id: int) -> bool:
    from app.testing.service import delete_testing_run

    return delete_testing_run(db, entity_id)


def _delete_roi(db: Session, entity_id: int) -> bool:
    from app.testing.service import delete_roi

    return delete_roi(db, entity_id)


def _delete_heatmap(db: Session, entity_id: int) -> bool:
    row = db.get(models.HeatmapRun, entity_id)
    if row is None:
        return False
    if row.artifacts_dir:
        shutil.rmtree(row.artifacts_dir, ignore_errors=True)
    db.delete(row)
    db.commit()
    return True


def _delete_heatmap_range(db: Session, entity_id: int) -> bool:
    from app.heatmap.service import delete_heatmap_range

    return delete_heatmap_range(db, entity_id)


def _delete_inspect_run(db: Session, entity_id: int) -> bool:
    from app.inspect.service import delete_inspect_run

    return delete_inspect_run(db, entity_id)


def _delete_analysis_layout(db: Session, entity_id: int) -> bool:
    from app.analysis.service import delete_analysis_layout

    return delete_analysis_layout(db, entity_id)


def _delete_optimization_study(db: Session, entity_id: int) -> bool:
    from app.optimization.service import delete_study

    return delete_study(db, entity_id)


def _delete_evaluation_profile(db: Session, entity_id: int) -> bool:
    row = db.get(models.EvaluationProfile, entity_id)
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def _delete_evaluation_label_set(db: Session, entity_id: int) -> bool:
    row = db.get(models.EvaluationLabelSet, entity_id)
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def _delete_model_evaluation(db: Session, entity_id: int) -> bool:
    row = db.get(models.ModelEvaluation, entity_id)
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def _delete_row(model):
    def delete_one(db: Session, entity_id: int) -> bool:
        row = db.get(model, entity_id)
        if row is None:
            return False
        db.delete(row)
        db.commit()
        return True
    return delete_one


def _delete_separation_calculation(db: Session, entity_id: int) -> bool:
    from app.evaluation import workspace_service
    row = db.get(models.EvaluationSeparationCalculation, entity_id)
    if row is None:
        return False
    workspace = db.get(models.EvaluationModelWorkspace, row.workspace_id)
    db.delete(row)
    db.flush()
    if workspace is not None:
        workspace_service._reaggregate(db, workspace)
    db.commit()
    return True


def _delete_drift_calculation(db: Session, entity_id: int) -> bool:
    from app.evaluation import workspace_service
    row = db.get(models.EvaluationDriftCalculation, entity_id)
    if row is None:
        return False
    workspace = db.get(models.EvaluationModelWorkspace, row.workspace_id)
    if workspace is None:
        db.delete(row); db.commit(); return True
    workspace_service.delete_drift_calculation(db, workspace.training_run_id, entity_id)
    return True


def _delete_redundancy_source(db: Session, entity_id: int) -> bool:
    from app.redundancy.service import delete_source
    return delete_source(db, entity_id)


def _delete_redundancy_analysis(db: Session, entity_id: int) -> bool:
    from app.redundancy.service import delete_analysis
    return delete_analysis(db, entity_id)


# -- specs -----------------------------------------------------------------------

_STATUS_FILTER = FilterSpec(key="status", label="Status", kind="select", column="status")
_CREATED_FILTER = FilterSpec(key="created", label="Created", kind="daterange", column="created_at")
_USAGE_FILTER = FilterSpec(key="usage", label="Usage", kind="usage", options=["used", "unused"])


ENTITY_SPECS: dict[str, EntitySpec] = {
    "dataset": EntitySpec(
        key="dataset",
        label="Datasets",
        model=models.Dataset,
        name_of=lambda r: r.name,
        list_fields=["id", "name", "root_path", "status", "created_at", "updated_at"],
        search_fields=["name", "root_path"],
        filters=[_STATUS_FILTER, _USAGE_FILTER, _CREATED_FILTER],
        dependents=_dataset_dependents,
        deleter=_delete_dataset,
    ),
    "training_dataset": EntitySpec(
        key="training_dataset",
        label="Train/Test Datasets",
        model=models.TrainingDataset,
        name_of=lambda r: r.name,
        list_fields=["id", "name", "usage_label", "notes", "created_at"],
        search_fields=["name", "notes"],
        filters=[
            FilterSpec(key="usage_label", label="Label", kind="select", column="usage_label"),
            _USAGE_FILTER,
            _CREATED_FILTER,
        ],
        dependents=_training_dataset_dependents,
        deleter=_delete_training_dataset,
    ),
    "preprocessing_pipeline": EntitySpec(
        key="preprocessing_pipeline",
        label="Preprocessing Pipelines",
        model=models.PreprocessingPipeline,
        name_of=lambda r: r.name,
        list_fields=[
            "id", "name", "description", "input_width", "input_height",
            "output_width", "output_height", "created_at", "updated_at",
        ],
        search_fields=["name", "description"],
        filters=[
            FilterSpec(key="output_width", label="Output width", kind="select", column="output_width"),
            _USAGE_FILTER,
            _CREATED_FILTER,
        ],
        dependents=_preprocessing_dependents,
        deleter=_delete_preprocessing,
    ),
    "method_configuration": EntitySpec(
        key="method_configuration",
        label="Methods",
        model=models.MethodConfiguration,
        name_of=lambda r: r.name,
        list_fields=[
            "id", "name", "description", "method_type", "method_family",
            "builder_kind", "training_mode", "created_at", "updated_at",
        ],
        search_fields=["name", "description", "method_type"],
        filters=[
            FilterSpec(key="method_type", label="Method type", kind="select", column="method_type"),
            FilterSpec(key="builder_kind", label="Builder", kind="select", column="builder_kind"),
            FilterSpec(key="training_mode", label="Training mode", kind="select", column="training_mode"),
            _USAGE_FILTER,
            _CREATED_FILTER,
        ],
        dependents=_method_dependents,
        deleter=_delete_method,
    ),
    "training_pipeline": EntitySpec(
        key="training_pipeline",
        label="Training Pipelines",
        model=models.TrainingPipeline,
        name_of=lambda r: r.name,
        list_fields=[
            "id", "name", "description", "preprocessing_pipeline_id",
            "method_configuration_id", "shuffle", "created_at", "updated_at",
        ],
        search_fields=["name", "description"],
        filters=[_USAGE_FILTER, _CREATED_FILTER],
        dependents=_training_pipeline_dependents,
        deleter=_delete_training_pipeline,
    ),
    "training_run": EntitySpec(
        key="training_run",
        label="Training Runs",
        model=models.TrainingRun,
        name_of=lambda r: f"Run #{r.id} · {r.training_pipeline_name}",
        list_fields=[
            "id", "training_pipeline_name", "method_type", "status", "device",
            "epochs_completed", "epochs_total", "train_loss", "val_loss",
            "image_count", "artifact_size_bytes", "duration_seconds", "created_at",
        ],
        search_fields=["training_pipeline_name", "method_type", "dataset_names_text"],
        filters=[
            _STATUS_FILTER,
            FilterSpec(key="method_type", label="Method type", kind="select", column="method_type"),
            _USAGE_FILTER,
            _CREATED_FILTER,
        ],
        dependents=_training_run_dependents,
        artifacts=_training_run_artifacts,
        deleter=_delete_training_run,
        blockers=_job_blockers,
        size_of=lambda r: r.artifact_size_bytes,
    ),
    "roi": EntitySpec(
        key="roi",
        label="ROIs",
        model=models.RoiDefinition,
        name_of=lambda r: r.name,
        list_fields=[
            "id", "name", "description", "image_width", "image_height",
            "geometry_type", "tile_rows", "tile_cols", "created_at",
        ],
        search_fields=["name", "description"],
        filters=[_CREATED_FILTER],
        # TestingRun/InspectRun reference ROIs with SET NULL — deleting only
        # detaches them, so ROIs have no blocking dependents.
        deleter=_delete_roi,
    ),
    "testing_run": EntitySpec(
        key="testing_run",
        label="Inference Runs",
        model=models.TestingRun,
        name_of=lambda r: r.name,
        list_fields=[
            "id", "name", "status", "method_type", "training_pipeline_name",
            "training_dataset_name", "image_count", "score_mean", "score_max",
            "results_size_bytes", "duration_seconds", "created_at",
        ],
        search_fields=["name", "training_pipeline_name", "training_dataset_name", "method_type"],
        filters=[
            _STATUS_FILTER,
            FilterSpec(key="method_type", label="Method type", kind="select", column="method_type"),
            _USAGE_FILTER,
            _CREATED_FILTER,
        ],
        dependents=_testing_run_dependents,
        artifacts=_testing_run_artifacts,
        deleter=_delete_testing_run,
        blockers=_job_blockers,
        size_of=lambda r: r.results_size_bytes,
    ),
    "heatmap": EntitySpec(
        key="heatmap",
        label="Heatmaps",
        model=models.HeatmapRun,
        name_of=lambda r: f"Heatmap #{r.id} · {r.timestamp:%Y-%m-%d %H:%M:%S}",
        list_fields=[
            "id", "testing_run_id", "status", "timestamp", "width", "height",
            "max_error", "mean_error", "render_version", "created_at",
        ],
        search_fields=["image_path"],
        filters=[_STATUS_FILTER, _CREATED_FILTER],
        artifacts=_heatmap_artifacts,
        deleter=_delete_heatmap,
        detail_exclude=frozenset({
            "source_image_data_url", "reconstruction_image_data_url",
            "heatmap_image_data_url", "error_matrix",
        }),
    ),
    "heatmap_range": EntitySpec(
        key="heatmap_range",
        label="Heatmap Videos",
        model=models.HeatmapRangeRun,
        name_of=lambda r: f"Heatmap video #{r.id} · {r.testing_run_name}",
        list_fields=[
            "id", "testing_run_name", "status", "start_timestamp", "end_timestamp",
            "stride", "scale_mode", "frame_count", "done_count", "created_at",
        ],
        search_fields=["testing_run_name"],
        filters=[
            _STATUS_FILTER,
            FilterSpec(key="scale_mode", label="Scale mode", kind="select", column="scale_mode"),
            _CREATED_FILTER,
        ],
        artifacts=_heatmap_range_artifacts,
        deleter=_delete_heatmap_range,
        blockers=_job_blockers,
    ),
    "inspect_run": EntitySpec(
        key="inspect_run",
        label="Inspect Runs",
        model=models.InspectRun,
        name_of=lambda r: f"Inspect #{r.id} · {r.training_dataset_name}",
        list_fields=[
            "id", "training_dataset_name", "preprocessing_pipeline_name", "status",
            "analysis_mode", "contrast_enabled", "start_timestamp", "end_timestamp",
            "frame_count", "done_count", "created_at",
        ],
        search_fields=["training_dataset_name", "preprocessing_pipeline_name"],
        filters=[
            _STATUS_FILTER,
            FilterSpec(key="analysis_mode", label="Mode", kind="select", column="analysis_mode"),
            _CREATED_FILTER,
        ],
        artifacts=_inspect_artifacts,
        deleter=_delete_inspect_run,
        blockers=_job_blockers,
    ),
    "analysis_layout": EntitySpec(
        key="analysis_layout",
        label="Analysis Layouts",
        model=models.AnalysisLayout,
        name_of=lambda r: r.name,
        list_fields=["id", "name", "description", "created_at", "updated_at"],
        search_fields=["name", "description"],
        filters=[_CREATED_FILTER],
        deleter=_delete_analysis_layout,
    ),
    "redundancy_csv_source": EntitySpec(
        key="redundancy_csv_source", label="Redundancy CSV Sources",
        model=models.RedundancyCsvSource, name_of=lambda row: row.name,
        list_fields=["id", "name", "original_filename", "row_count", "byte_size", "created_at"],
        search_fields=["name", "original_filename", "sha256"], filters=[_USAGE_FILTER, _CREATED_FILTER],
        dependents=_redundancy_source_dependents, artifacts=_redundancy_source_artifacts,
        deleter=_delete_redundancy_source, size_of=lambda row: row.byte_size,
        detail_exclude=frozenset({"preview_rows"}),
    ),
    "redundancy_analysis": EntitySpec(
        key="redundancy_analysis", label="Redundancy Analyses",
        model=models.RedundancyAnalysis, name_of=lambda row: row.name,
        list_fields=["id", "name", "source_id", "status", "job_status", "progress", "active_cutoff", "created_at", "finalized_at"],
        search_fields=["name", "description", "time_column"], filters=[_STATUS_FILTER, _CREATED_FILTER],
        deleter=_delete_redundancy_analysis, blockers=_job_blockers,
        detail_exclude=frozenset({"result"}),
    ),
    "evaluation_profile": EntitySpec(
        key="evaluation_profile",
        label="Evaluation Profiles",
        model=models.EvaluationProfile,
        name_of=lambda row: row.name,
        list_fields=[
            "id", "name", "normal_window_duration_seconds", "drift_window_seconds",
            "false_alarm_horizon_seconds", "anticipation_seconds", "created_at", "updated_at",
        ],
        search_fields=["name", "description"],
        filters=[_USAGE_FILTER, _CREATED_FILTER],
        dependents=_evaluation_profile_dependents,
        deleter=_delete_evaluation_profile,
    ),
    "evaluation_label_set": EntitySpec(
        key="evaluation_label_set",
        label="Evaluation Label Sets",
        model=models.EvaluationLabelSet,
        name_of=lambda row: row.name,
        list_fields=["id", "name", "training_dataset_id", "version", "created_at", "updated_at"],
        search_fields=["name", "description"],
        filters=[_USAGE_FILTER, _CREATED_FILTER],
        dependents=_evaluation_label_set_dependents,
        deleter=_delete_evaluation_label_set,
    ),
    "model_evaluation": EntitySpec(
        key="model_evaluation",
        label="Model Evaluations",
        model=models.ModelEvaluation,
        name_of=lambda row: row.name,
        list_fields=[
            "id", "name", "status", "score_series", "sep_median", "sep_min",
            "drift_mean", "drift_max", "event_recall", "median_delay_seconds",
            "frame_fpr", "false_alarm_rate_t0", "created_at", "finalized_at",
        ],
        search_fields=["name", "score_series"],
        filters=[_STATUS_FILTER, _CREATED_FILTER],
        deleter=_delete_model_evaluation,
        detail_exclude=frozenset({
            "separation_result", "drift_result", "detection_result", "label_snapshot",
            "source_snapshot",
        }),
    ),
    "evaluation_workspace": EntitySpec(
        key="evaluation_workspace", label="Evaluation Workspaces",
        model=models.EvaluationModelWorkspace,
        name_of=lambda row: f"Model workspace #{row.id}",
        list_fields=["id", "training_run_id", "artifact_signature", "sep_median", "sep_min", "drift_mean", "drift_max", "updated_at"],
        search_fields=["artifact_signature"], filters=[_USAGE_FILTER, _CREATED_FILTER],
        dependents=_workspace_dependents, deleter=_delete_row(models.EvaluationModelWorkspace),
    ),
    "evaluation_separation_layout": EntitySpec(
        key="evaluation_separation_layout", label="Event Separation Layouts",
        model=models.EvaluationSeparationLayout, name_of=lambda row: row.name,
        list_fields=["id", "name", "training_dataset_id", "version", "created_at", "updated_at"],
        search_fields=["name", "description"], filters=[_USAGE_FILTER, _CREATED_FILTER],
        dependents=_separation_layout_dependents, deleter=_delete_row(models.EvaluationSeparationLayout),
    ),
    "evaluation_drift_layout": EntitySpec(
        key="evaluation_drift_layout", label="Score Stability Layouts",
        model=models.EvaluationDriftLayout, name_of=lambda row: row.name,
        list_fields=["id", "name", "training_dataset_id", "version", "bucket_seconds", "created_at", "updated_at"],
        search_fields=["name", "description"], filters=[_USAGE_FILTER, _CREATED_FILTER],
        dependents=_drift_layout_dependents, deleter=_delete_row(models.EvaluationDriftLayout),
    ),
    "evaluation_separation_calculation": EntitySpec(
        key="evaluation_separation_calculation", label="Event Separation Calculations",
        model=models.EvaluationSeparationCalculation, name_of=lambda row: f"A calculation #{row.id}",
        list_fields=["id", "workspace_id", "testing_run_id", "layout_version", "score_series", "source_result_revision", "stale", "created_at"],
        search_fields=["score_series", "artifact_signature"], filters=[_CREATED_FILTER],
        deleter=_delete_separation_calculation, detail_exclude=frozenset({"layout_snapshot"}),
    ),
    "evaluation_drift_calculation": EntitySpec(
        key="evaluation_drift_calculation", label="Score Stability Calculations",
        model=models.EvaluationDriftCalculation, name_of=lambda row: f"B calculation #{row.id}",
        list_fields=["id", "workspace_id", "testing_run_id", "layout_version", "score_series", "drift_mean", "drift_max", "active", "stale", "created_at"],
        search_fields=["score_series", "artifact_signature"], filters=[_CREATED_FILTER],
        deleter=_delete_drift_calculation, detail_exclude=frozenset({"layout_snapshot"}),
    ),
    "optimization_study": EntitySpec(
        key="optimization_study",
        label="Optimization Studies",
        model=models.OptimizationStudy,
        name_of=lambda r: r.name,
        list_fields=[
            "id", "name", "status", "objective_name", "direction", "n_trials",
            "sampler", "best_value", "created_at",
        ],
        search_fields=["name", "description", "objective_name"],
        filters=[_STATUS_FILTER, _CREATED_FILTER],
        deleter=_delete_optimization_study,
        blockers=_job_blockers,
    ),
}

# Bottom-up deletion order for cascades: children before their parents.
DELETE_ORDER: list[str] = [
    "redundancy_analysis",
    "evaluation_separation_calculation",
    "evaluation_drift_calculation",
    "model_evaluation",
    "heatmap",
    "heatmap_range",
    "testing_run",
    "inspect_run",
    "optimization_study",
    "evaluation_workspace",
    "training_run",
    "training_pipeline",
    "analysis_layout",
    "evaluation_separation_layout",
    "evaluation_drift_layout",
    "evaluation_label_set",
    "evaluation_profile",
    "roi",
    "method_configuration",
    "preprocessing_pipeline",
    "training_dataset",
    "dataset",
    "redundancy_csv_source",
]
