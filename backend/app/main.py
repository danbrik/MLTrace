from contextlib import asynccontextmanager
import logging
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal, get_db, project_context, vacuum_database
from app.projects import (
    create_project,
    get_project,
    initialize_catalog,
    list_projects,
    mark_project_opened,
    migrate_all_projects,
    serialize_project,
    list_queue_entries,
)
from app.schemas import (
    DatasetConnectionTestRequest,
    DatasetConnectionTestResponse,
    DatasetCreate,
    DatasetRead,
    CacheRevisionsRead,
    AnalysisLayoutCreate,
    AnalysisLayoutRead,
    HeatmapRangeRunCreate,
    HeatmapRangeRunRead,
    HeatmapRunCreate,
    HeatmapRunRead,
    HeatmapRunSummary,
    InspectPreviewRequest,
    InspectPreviewResponse,
    InspectArtifactRunPage,
    InspectCsvData,
    InspectRunCreate,
    InspectRunRead,
    MethodConfigurationCreate,
    MethodConfigurationPayload,
    MethodConfigurationRead,
    MethodConfigurationSummaryRead,
    MethodTorchCheckResponse,
    MethodConfigurationValidationResponse,
    MethodDefinitionRead,
    ModelLayerRead,
    OptimizationPromoteRequest,
    OptimizationSplitCreate,
    OptimizationSplitRead,
    OptimizationStudyCreate,
    OptimizationStudyRead,
    OptimizationStudyUpdate,
    PreprocessingPipelineCreate,
    PreprocessingPipelineRead,
    PreprocessingPipelineSummaryRead,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResponse,
    PreprocessingStepRead,
    RegistryDeleteRequest,
    RoiDefinitionCreate,
    RoiDefinitionRead,
    RoiPreviewRequest,
    RoiPreviewResponse,
    SchedulerSettingsRead,
    SchedulerSettingsUpdate,
    ProjectCreate,
    ProjectRead,
    GpuSnapshotRead,
    SchedulerJobWithProjectRead,
    SchedulerJobMoveRequest,
    SchedulerJobMoveResponse,
    TestingRunBulkCreate,
    TestingRunBulkResponse,
    TestingRunCreate,
    TestingRunResultImageResponse,
    TestingRunRead,
    TestingRunResultsResponse,
    TimestampFormatConfirm,
    TrainingDatasetCreate,
    TrainingDatasetPreviewRequest,
    TrainingDatasetPreviewResponse,
    TrainingDatasetRead,
    TrainingDatasetSummaryRead,
    TrainingPipelineCreate,
    TrainingPipelineDryRunRequest,
    TrainingPipelineDryRunResponse,
    TrainingPipelineDuplicateResponse,
    TrainingPipelinePayload,
    TrainingPipelineRead,
    TrainingPipelineSummaryRead,
    TrainingRunEnqueueRequest,
    TrainingRunLogResponse,
    TrainingRunRead,
)
from app.analysis import service as analysis_service
from app.heatmap import service as heatmap_service
from app.registry import service as registry_service
from app.inspect import service as inspect_service
from app.optimization import service as optimization_service
from app.testing import service as testing_service
from app.testing.service import TestingConflict
from app.training import service as training_service
from app.training.scheduler import get_scheduler_settings, move_queued_job, scheduler, update_scheduler_settings
from app.training.service import RunConflict
from app.services import (
    DuplicatePipelineError,
    create_dataset,
    cache_revisions,
    create_method_configuration,
    create_preprocessing_pipeline,
    create_training_dataset,
    create_training_pipeline,
    delete_dataset,
    delete_method_configuration,
    delete_preprocessing_pipeline,
    delete_training_dataset,
    delete_training_pipeline,
    dry_run_training_pipeline,
    find_training_pipeline_by_signature,
    cleanup_invalid_training_dataset_rules,
    get_dataset_or_404,
    get_method_configuration,
    get_method_definition,
    get_preprocessing_pipeline,
    get_training_dataset,
    get_training_pipeline,
    list_datasets,
    list_method_configurations,
    list_method_definitions,
    list_method_layers,
    list_preprocessing_pipelines,
    list_preprocessing_steps,
    list_training_datasets,
    list_training_pipelines,
    preview_preprocessing_pipeline,
    preview_training_dataset,
    refresh_training_dataset_counts,
    run_method_torch_check,
    scan_dataset,
    serialize_dataset,
    test_dataset_connection,
    update_method_configuration,
    update_preprocessing_pipeline,
    update_training_dataset,
    update_training_pipeline,
    validate_method_configuration,
)


logger = logging.getLogger("mltrace.api")


def _value_error_status(exc: ValueError) -> int:
    message = str(exc).lower()
    if "not editable" in message or "locked" in message or "already used" in message:
        return 409
    return 400


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background training scheduler with the API process. Detached
    # worker subprocesses survive an API restart and are reconciled on startup.
    from app.modeling.defaults import ensure_default_method_configurations

    initialize_catalog()
    migrate_all_projects()
    for project in list_projects():
        with project_context(project.database_url, project.artifact_dir):
            db = SessionLocal()
            try:
                ensure_default_method_configurations(db)
            finally:
                db.close()
    scheduler.start()
    optimization_service.optimization_loop.start()
    inspect_service.inspect_queue.start()
    try:
        yield
    finally:
        inspect_service.inspect_queue.stop()
        optimization_service.optimization_loop.stop()
        scheduler.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="MLTrace API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def select_project_database(request: Request, call_next):
        path = request.url.path
        is_global = (
            path == "/api/health"
            or path == "/api/projects"
            or path.startswith("/api/projects/")
            or path.startswith("/api/system/")
            or path == "/api/scheduler/settings"
            or path == "/api/scheduler/jobs"
        )
        needs_project = path.startswith("/api/") and not is_global
        # Existing unit tests override the database dependency with an isolated
        # in-memory database and intentionally do not participate in the catalog.
        if get_db in app.dependency_overrides:
            return await call_next(request)
        if not needs_project:
            return await call_next(request)
        project_id = request.headers.get("X-MLTrace-Project-ID") or request.query_params.get("project_id")
        if not project_id:
            return JSONResponse(status_code=400, content={"detail": "X-MLTrace-Project-ID header is required."})
        project = get_project(project_id)
        if project is None:
            return JSONResponse(status_code=404, content={"detail": "Project not found."})
        request.state.project = project
        with project_context(project.database_url, project.artifact_dir):
            return await call_next(request)

    @app.middleware("http")
    async def log_slow_api_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if request.url.path.startswith("/api/") and elapsed_ms > 500:
            logger.info("Slow API request %.0fms %s %s", elapsed_ms, request.method, request.url.path)
        return response

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/projects", response_model=list[ProjectRead])
    def api_list_projects():
        initialize_catalog()
        return [serialize_project(project) for project in list_projects()]

    @app.post("/api/projects", response_model=ProjectRead)
    def api_create_project(payload: ProjectCreate):
        initialize_catalog()
        try:
            return serialize_project(create_project(payload.name, payload.description))
        except ValueError as exc:
            raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}", response_model=ProjectRead)
    def api_get_project(project_id: str):
        project = get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return serialize_project(project)

    @app.post("/api/projects/{project_id}/opened", response_model=ProjectRead)
    def api_mark_project_opened(project_id: str):
        project = mark_project_opened(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return serialize_project(project)

    @app.get("/api/system/gpu-usage", response_model=GpuSnapshotRead)
    def api_gpu_usage(refresh: bool = False):
        from app.gpu_monitor import gpu_snapshot
        initialize_catalog()
        return gpu_snapshot(force=refresh)

    @app.get("/api/cache/revisions", response_model=CacheRevisionsRead)
    def api_cache_revisions(db: Session = Depends(get_db)):
        return cache_revisions(db)

    @app.get("/api/analysis/layouts", response_model=list[AnalysisLayoutRead])
    def api_list_analysis_layouts(db: Session = Depends(get_db)):
        return analysis_service.list_analysis_layouts(db)

    @app.post("/api/analysis/layouts", response_model=AnalysisLayoutRead)
    def api_create_analysis_layout(payload: AnalysisLayoutCreate, db: Session = Depends(get_db)):
        try:
            return analysis_service.create_analysis_layout(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc)) from exc

    @app.get("/api/analysis/layouts/{layout_id}", response_model=AnalysisLayoutRead)
    def api_get_analysis_layout(layout_id: int, db: Session = Depends(get_db)):
        layout = analysis_service.get_analysis_layout(db, layout_id)
        if layout is None:
            raise HTTPException(status_code=404, detail="Analysis layout not found.")
        return layout

    @app.put("/api/analysis/layouts/{layout_id}", response_model=AnalysisLayoutRead)
    def api_update_analysis_layout(layout_id: int, payload: AnalysisLayoutCreate, db: Session = Depends(get_db)):
        try:
            layout = analysis_service.update_analysis_layout(db, layout_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc)) from exc
        if layout is None:
            raise HTTPException(status_code=404, detail="Analysis layout not found.")
        return layout

    @app.delete("/api/analysis/layouts/{layout_id}", status_code=204)
    def api_delete_analysis_layout(layout_id: int, db: Session = Depends(get_db)):
        if not analysis_service.delete_analysis_layout(db, layout_id):
            raise HTTPException(status_code=404, detail="Analysis layout not found.")
        return None

    @app.get("/api/optimization/studies", response_model=list[OptimizationStudyRead])
    def api_list_optimization_studies(db: Session = Depends(get_db)):
        return optimization_service.list_studies(db)

    @app.post("/api/optimization/splits", response_model=OptimizationSplitRead)
    def api_create_optimization_split(payload: OptimizationSplitCreate, db: Session = Depends(get_db)):
        try:
            return optimization_service.create_time_split(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/optimization/studies", response_model=OptimizationStudyRead)
    def api_create_optimization_study(payload: OptimizationStudyCreate, db: Session = Depends(get_db)):
        try:
            return optimization_service.create_study(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc)) from exc

    @app.get("/api/optimization/studies/{study_id}", response_model=OptimizationStudyRead)
    def api_get_optimization_study(study_id: int, db: Session = Depends(get_db)):
        study = optimization_service.get_study(db, study_id)
        if study is None:
            raise HTTPException(status_code=404, detail="Optimization study not found.")
        return study

    @app.put("/api/optimization/studies/{study_id}", response_model=OptimizationStudyRead)
    def api_update_optimization_study(study_id: int, payload: OptimizationStudyUpdate, db: Session = Depends(get_db)):
        try:
            study = optimization_service.update_study(db, study_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc)) from exc
        if study is None:
            raise HTTPException(status_code=404, detail="Optimization study not found.")
        return study

    @app.delete("/api/optimization/studies/{study_id}", status_code=204)
    def api_delete_optimization_study(study_id: int, db: Session = Depends(get_db)):
        try:
            deleted = optimization_service.delete_study(db, study_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Optimization study not found.")
        return None

    @app.post("/api/optimization/studies/{study_id}/start", response_model=OptimizationStudyRead)
    def api_start_optimization_study(study_id: int, db: Session = Depends(get_db)):
        try:
            study = optimization_service.start_study(db, study_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if study is None:
            raise HTTPException(status_code=404, detail="Optimization study not found.")
        return study

    @app.post("/api/optimization/studies/{study_id}/pause", response_model=OptimizationStudyRead)
    def api_pause_optimization_study(study_id: int, db: Session = Depends(get_db)):
        study = optimization_service.pause_study(db, study_id)
        if study is None:
            raise HTTPException(status_code=404, detail="Optimization study not found.")
        return study

    @app.post("/api/optimization/studies/{study_id}/resume", response_model=OptimizationStudyRead)
    def api_resume_optimization_study(study_id: int, db: Session = Depends(get_db)):
        study = optimization_service.resume_study(db, study_id)
        if study is None:
            raise HTTPException(status_code=404, detail="Optimization study not found.")
        return study

    @app.post("/api/optimization/studies/{study_id}/abort", response_model=OptimizationStudyRead)
    def api_abort_optimization_study(study_id: int, db: Session = Depends(get_db)):
        study = optimization_service.abort_study(db, study_id)
        if study is None:
            raise HTTPException(status_code=404, detail="Optimization study not found.")
        return study

    @app.post("/api/optimization/trials/{trial_id}/promote", response_model=TrainingPipelineRead)
    def api_promote_optimization_trial(trial_id: int, payload: OptimizationPromoteRequest, db: Session = Depends(get_db)):
        try:
            pipeline = optimization_service.promote_trial(db, trial_id, payload)
        except DuplicatePipelineError as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "existing_pipeline_id": exc.existing.id},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Optimization trial not found.")
        return pipeline

    @app.post("/api/datasets", response_model=DatasetRead)
    def api_create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
        try:
            return create_dataset(db, payload.name, payload.root_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="A dataset with this root path already exists.") from exc

    @app.post("/api/datasets/test-connection", response_model=DatasetConnectionTestResponse)
    def api_test_dataset_connection(payload: DatasetConnectionTestRequest):
        return test_dataset_connection(payload.root_path)

    @app.get("/api/datasets", response_model=list[DatasetRead])
    def api_list_datasets(db: Session = Depends(get_db)):
        return list_datasets(db)

    @app.get("/api/datasets/{dataset_id}", response_model=DatasetRead)
    def api_get_dataset(dataset_id: int, db: Session = Depends(get_db)):
        dataset = get_dataset_or_404(db, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return serialize_dataset(db, dataset)

    @app.post("/api/datasets/{dataset_id}/confirm-timestamp-format", response_model=DatasetRead)
    def api_confirm_timestamp_format(
        dataset_id: int, payload: TimestampFormatConfirm, db: Session = Depends(get_db)
    ):
        dataset = get_dataset_or_404(db, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        try:
            return scan_dataset(db, dataset, payload.timestamp_regex, payload.timestamp_format)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/datasets/{dataset_id}/rescan", response_model=DatasetRead)
    def api_rescan_dataset(dataset_id: int, db: Session = Depends(get_db)):
        dataset = get_dataset_or_404(db, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        if not dataset.timestamp_regex or not dataset.timestamp_format:
            raise HTTPException(status_code=400, detail="Timestamp format has not been confirmed yet.")
        try:
            return scan_dataset(db, dataset, dataset.timestamp_regex, dataset.timestamp_format)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    def api_delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
        try:
            deleted = delete_dataset(db, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return None

    @app.post("/api/training-datasets/preview", response_model=TrainingDatasetPreviewResponse)
    def api_preview_training_dataset(payload: TrainingDatasetPreviewRequest, db: Session = Depends(get_db)):
        try:
            return preview_training_dataset(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/training-datasets", response_model=TrainingDatasetRead)
    def api_create_training_dataset(payload: TrainingDatasetCreate, db: Session = Depends(get_db)):
        try:
            return create_training_dataset(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/training-datasets", response_model=list[TrainingDatasetRead] | list[TrainingDatasetSummaryRead])
    def api_list_training_datasets(summary: bool = Query(False), db: Session = Depends(get_db)):
        return list_training_datasets(db, summary=summary)

    @app.get("/api/training-datasets/{training_dataset_id}", response_model=TrainingDatasetRead)
    def api_get_training_dataset(training_dataset_id: int, db: Session = Depends(get_db)):
        training_dataset = get_training_dataset(db, training_dataset_id)
        if training_dataset is None:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        return training_dataset

    @app.put("/api/training-datasets/{training_dataset_id}", response_model=TrainingDatasetRead)
    def api_update_training_dataset(
        training_dataset_id: int, payload: TrainingDatasetCreate, db: Session = Depends(get_db)
    ):
        try:
            updated = update_training_dataset(db, training_dataset_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=_value_error_status(exc), detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        return updated

    @app.post("/api/training-datasets/{training_dataset_id}/cleanup-invalid-rules", response_model=TrainingDatasetRead)
    def api_cleanup_invalid_training_dataset_rules(training_dataset_id: int, db: Session = Depends(get_db)):
        try:
            updated = cleanup_invalid_training_dataset_rules(db, training_dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        return updated

    @app.post("/api/training-datasets/{training_dataset_id}/refresh-counts", response_model=TrainingDatasetRead)
    def api_refresh_training_dataset_counts(training_dataset_id: int, db: Session = Depends(get_db)):
        try:
            updated = refresh_training_dataset_counts(db, training_dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        return updated

    @app.delete("/api/training-datasets/{training_dataset_id}", status_code=204)
    def api_delete_training_dataset(training_dataset_id: int, db: Session = Depends(get_db)):
        try:
            deleted = delete_training_dataset(db, training_dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        return None

    @app.get("/api/preprocessing/steps", response_model=list[PreprocessingStepRead])
    def api_list_preprocessing_steps():
        return list_preprocessing_steps()

    @app.get(
        "/api/preprocessing/pipelines",
        response_model=list[PreprocessingPipelineRead] | list[PreprocessingPipelineSummaryRead],
    )
    def api_list_preprocessing_pipelines(summary: bool = Query(False), db: Session = Depends(get_db)):
        return list_preprocessing_pipelines(db, summary=summary)

    @app.post("/api/preprocessing/pipelines", response_model=PreprocessingPipelineRead)
    def api_create_preprocessing_pipeline(payload: PreprocessingPipelineCreate, db: Session = Depends(get_db)):
        try:
            return create_preprocessing_pipeline(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/preprocessing/pipelines/preview", response_model=PreprocessingPreviewResponse)
    def api_preview_preprocessing_pipeline(payload: PreprocessingPreviewRequest, db: Session = Depends(get_db)):
        try:
            return preview_preprocessing_pipeline(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/preprocessing/pipelines/{pipeline_id}", response_model=PreprocessingPipelineRead)
    def api_get_preprocessing_pipeline(pipeline_id: int, db: Session = Depends(get_db)):
        pipeline = get_preprocessing_pipeline(db, pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Preprocessing pipeline not found.")
        return pipeline

    @app.put("/api/preprocessing/pipelines/{pipeline_id}", response_model=PreprocessingPipelineRead)
    def api_update_preprocessing_pipeline(
        pipeline_id: int, payload: PreprocessingPipelineCreate, db: Session = Depends(get_db)
    ):
        try:
            updated = update_preprocessing_pipeline(db, pipeline_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=_value_error_status(exc), detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Preprocessing pipeline not found.")
        return updated

    @app.delete("/api/preprocessing/pipelines/{pipeline_id}", status_code=204)
    def api_delete_preprocessing_pipeline(pipeline_id: int, db: Session = Depends(get_db)):
        try:
            deleted = delete_preprocessing_pipeline(db, pipeline_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Preprocessing pipeline not found.")
        return None

    @app.get("/api/methods/definitions", response_model=list[MethodDefinitionRead])
    def api_list_method_definitions():
        return list_method_definitions()

    @app.get("/api/methods/definitions/{method_type}", response_model=MethodDefinitionRead)
    def api_get_method_definition(method_type: str):
        try:
            return get_method_definition(method_type)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/methods/layers", response_model=list[ModelLayerRead])
    def api_list_method_layers():
        return list_method_layers()

    @app.get(
        "/api/methods/configurations",
        response_model=list[MethodConfigurationRead] | list[MethodConfigurationSummaryRead],
    )
    def api_list_method_configurations(summary: bool = Query(False), db: Session = Depends(get_db)):
        return list_method_configurations(db, summary=summary)

    @app.post("/api/methods/configurations", response_model=MethodConfigurationRead)
    def api_create_method_configuration(payload: MethodConfigurationCreate, db: Session = Depends(get_db)):
        try:
            return create_method_configuration(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="A method with this name already exists.") from exc

    @app.post("/api/methods/configurations/validate", response_model=MethodConfigurationValidationResponse)
    def api_validate_method_configuration(payload: MethodConfigurationPayload):
        return validate_method_configuration(payload)

    @app.post("/api/methods/configurations/diagram", response_model=MethodConfigurationValidationResponse)
    def api_method_configuration_diagram(payload: MethodConfigurationPayload):
        return validate_method_configuration(payload)

    @app.post("/api/methods/configurations/torch-check", response_model=MethodTorchCheckResponse)
    def api_method_configuration_torch_check(payload: MethodConfigurationPayload):
        return run_method_torch_check(payload)

    @app.get("/api/methods/configurations/{configuration_id}", response_model=MethodConfigurationRead)
    def api_get_method_configuration(configuration_id: int, db: Session = Depends(get_db)):
        configuration = get_method_configuration(db, configuration_id)
        if configuration is None:
            raise HTTPException(status_code=404, detail="Method configuration not found.")
        return configuration

    @app.put("/api/methods/configurations/{configuration_id}", response_model=MethodConfigurationRead)
    def api_update_method_configuration(
        configuration_id: int, payload: MethodConfigurationCreate, db: Session = Depends(get_db)
    ):
        try:
            updated = update_method_configuration(db, configuration_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=_value_error_status(exc), detail=str(exc)) from exc
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="A method with this name already exists.") from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Method configuration not found.")
        return updated

    @app.delete("/api/methods/configurations/{configuration_id}", status_code=204)
    def api_delete_method_configuration(configuration_id: int, db: Session = Depends(get_db)):
        try:
            deleted = delete_method_configuration(db, configuration_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Method configuration not found.")
        return None

    @app.get("/api/training-pipelines", response_model=list[TrainingPipelineRead] | list[TrainingPipelineSummaryRead])
    def api_list_training_pipelines(summary: bool = Query(False), db: Session = Depends(get_db)):
        return list_training_pipelines(db, summary=summary)

    @app.post("/api/training-pipelines", response_model=TrainingPipelineRead)
    def api_create_training_pipeline(payload: TrainingPipelineCreate, db: Session = Depends(get_db)):
        try:
            return create_training_pipeline(db, payload)
        except DuplicatePipelineError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "existing_pipeline_id": exc.existing.id,
                    "existing_name": exc.existing.name,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=_value_error_status(exc), detail=str(exc)) from exc
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="A training pipeline with this name already exists.") from exc

    @app.post("/api/training-pipelines/resolve-duplicate", response_model=TrainingPipelineDuplicateResponse)
    def api_resolve_duplicate_training_pipeline(payload: TrainingPipelinePayload, db: Session = Depends(get_db)):
        try:
            existing = find_training_pipeline_by_signature(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        existing_read = get_training_pipeline(db, existing.id) if existing is not None else None
        return TrainingPipelineDuplicateResponse(existing_pipeline=existing_read)

    @app.post("/api/training-pipelines/dry-run", response_model=TrainingPipelineDryRunResponse)
    def api_dry_run_training_pipeline(payload: TrainingPipelineDryRunRequest, db: Session = Depends(get_db)):
        return dry_run_training_pipeline(db, payload)

    @app.get("/api/training-pipelines/{pipeline_id}", response_model=TrainingPipelineRead)
    def api_get_training_pipeline(pipeline_id: int, db: Session = Depends(get_db)):
        pipeline = get_training_pipeline(db, pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Training pipeline not found.")
        return pipeline

    @app.put("/api/training-pipelines/{pipeline_id}", response_model=TrainingPipelineRead)
    def api_update_training_pipeline(
        pipeline_id: int, payload: TrainingPipelineCreate, db: Session = Depends(get_db)
    ):
        try:
            updated = update_training_pipeline(db, pipeline_id, payload)
        except DuplicatePipelineError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "existing_pipeline_id": exc.existing.id,
                    "existing_name": exc.existing.name,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="A training pipeline with this name already exists.") from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Training pipeline not found.")
        return updated

    @app.delete("/api/training-pipelines/{pipeline_id}", status_code=204)
    def api_delete_training_pipeline(pipeline_id: int, db: Session = Depends(get_db)):
        try:
            deleted = delete_training_pipeline(db, pipeline_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Training pipeline not found.")
        return None

    @app.post("/api/training-runs", response_model=TrainingRunRead)
    def api_enqueue_training_run(payload: TrainingRunEnqueueRequest, db: Session = Depends(get_db)):
        try:
            return training_service.enqueue_training_run(db, payload.training_pipeline_id)
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/training-runs", response_model=list[TrainingRunRead])
    def api_list_training_runs(
        status: str | None = None,
        method_type: str | None = None,
        training_mode: str | None = None,
        builder_kind: str | None = None,
        search: str | None = None,
        min_val_loss: float | None = None,
        max_val_loss: float | None = None,
        min_train_loss: float | None = None,
        max_train_loss: float | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
        db: Session = Depends(get_db),
    ):
        return training_service.list_training_runs(
            db,
            status=status,
            method_type=method_type,
            training_mode=training_mode,
            builder_kind=builder_kind,
            search=search,
            min_val_loss=min_val_loss,
            max_val_loss=max_val_loss,
            min_train_loss=min_train_loss,
            max_train_loss=max_train_loss,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    @app.get("/api/training-runs/{run_id}", response_model=TrainingRunRead)
    def api_get_training_run(run_id: int, db: Session = Depends(get_db)):
        run = training_service.get_training_run(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Training run not found.")
        return run

    @app.get("/api/training-runs/{run_id}/log", response_model=TrainingRunLogResponse)
    def api_get_training_run_log(run_id: int, db: Session = Depends(get_db)):
        log = training_service.read_run_log(db, run_id)
        if log is None:
            raise HTTPException(status_code=404, detail="Training run not found.")
        return TrainingRunLogResponse(log=log)

    @app.post("/api/training-runs/{run_id}/abort", response_model=TrainingRunRead)
    def api_abort_training_run(run_id: int, db: Session = Depends(get_db)):
        try:
            run = training_service.abort_training_run(db, run_id)
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Training run not found.")
        return run

    @app.post("/api/training-runs/{run_id}/restart", response_model=TrainingRunRead)
    def api_restart_training_run(run_id: int, db: Session = Depends(get_db)):
        try:
            run = training_service.restart_training_run(db, run_id)
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Training run not found.")
        return run

    @app.delete("/api/training-runs/{run_id}", status_code=204)
    def api_delete_training_run(run_id: int, db: Session = Depends(get_db)):
        try:
            deleted = training_service.delete_training_run(db, run_id)
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Training run not found.")
        return None

    @app.get("/api/rois", response_model=list[RoiDefinitionRead])
    def api_list_rois(db: Session = Depends(get_db)):
        return testing_service.list_rois(db)

    @app.post("/api/rois", response_model=RoiDefinitionRead)
    def api_create_roi(payload: RoiDefinitionCreate, db: Session = Depends(get_db)):
        try:
            return testing_service.create_roi(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="An ROI with this name already exists.") from exc

    @app.post("/api/rois/preview", response_model=RoiPreviewResponse)
    def api_preview_roi(payload: RoiPreviewRequest, db: Session = Depends(get_db)):
        try:
            return testing_service.preview_roi_image(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/rois/{roi_id}", status_code=204)
    def api_delete_roi(roi_id: int, db: Session = Depends(get_db)):
        deleted = testing_service.delete_roi(db, roi_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="ROI not found.")
        return None

    @app.get("/api/testing-runs", response_model=list[TestingRunRead])
    def api_list_testing_runs(db: Session = Depends(get_db)):
        return testing_service.list_testing_runs(db)

    @app.post("/api/testing-runs", response_model=TestingRunRead)
    def api_enqueue_testing_run(payload: TestingRunCreate, db: Session = Depends(get_db)):
        try:
            return testing_service.enqueue_testing_run(db, payload)
        except TestingConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "existing_testing_run_id": exc.existing.id,
                    "existing_name": exc.existing.name,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/testing-runs/bulk", response_model=TestingRunBulkResponse)
    def api_bulk_enqueue_testing_runs(payload: TestingRunBulkCreate, db: Session = Depends(get_db)):
        try:
            return testing_service.bulk_enqueue_testing_runs(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/testing-runs/{run_id}", response_model=TestingRunRead)
    def api_get_testing_run(run_id: int, db: Session = Depends(get_db)):
        run = testing_service.get_testing_run(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Testing run not found.")
        return run

    @app.get("/api/testing-runs/{run_id}/results", response_model=TestingRunResultsResponse)
    def api_get_testing_run_results(
        run_id: int, max_points: int | None = None, db: Session = Depends(get_db)
    ):
        response = testing_service.get_testing_run_results(db, run_id, max_points=max_points)
        if response is None:
            raise HTTPException(status_code=404, detail="Testing run not found.")
        return response

    @app.get(
        "/api/testing-runs/{run_id}/results/{result_id}/image",
        response_model=TestingRunResultImageResponse,
    )
    def api_get_testing_run_result_image(run_id: int, result_id: int, db: Session = Depends(get_db)):
        try:
            response = testing_service.get_testing_run_result_image(db, run_id, result_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if response is None:
            raise HTTPException(status_code=404, detail="Testing result not found.")
        return response

    @app.get("/api/heatmaps", response_model=list[HeatmapRunSummary])
    def api_list_heatmaps(db: Session = Depends(get_db)):
        return testing_service.list_heatmap_runs(db)

    @app.get("/api/heatmaps/{run_id}", response_model=HeatmapRunRead)
    def api_get_heatmap(run_id: int, db: Session = Depends(get_db)):
        heatmap = testing_service.get_heatmap_run(db, run_id)
        if heatmap is None:
            raise HTTPException(status_code=404, detail="Heatmap not found.")
        return heatmap

    @app.post("/api/heatmaps", response_model=HeatmapRunRead)
    def api_create_heatmap(payload: HeatmapRunCreate, db: Session = Depends(get_db)):
        try:
            return testing_service.compute_heatmap_run(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/heatmaps", status_code=204)
    def api_clear_heatmaps(db: Session = Depends(get_db)):
        testing_service.clear_heatmap_runs(db)
        return None

    @app.post("/api/maintenance/vacuum", status_code=204)
    def api_vacuum_database():
        vacuum_database()
        return None

    @app.get("/api/heatmap-ranges", response_model=list[HeatmapRangeRunRead])
    def api_list_heatmap_ranges(db: Session = Depends(get_db)):
        return heatmap_service.list_heatmap_ranges(db)

    @app.post("/api/heatmap-ranges", response_model=HeatmapRangeRunRead)
    def api_create_heatmap_range(payload: HeatmapRangeRunCreate, db: Session = Depends(get_db)):
        try:
            return heatmap_service.enqueue_heatmap_range(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/heatmap-ranges/{run_id}", response_model=HeatmapRangeRunRead)
    def api_get_heatmap_range(run_id: int, db: Session = Depends(get_db)):
        run = heatmap_service.get_heatmap_range(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Heatmap range run not found.")
        return run

    @app.get("/api/heatmap-ranges/{run_id}/frames/{index}.png")
    def api_get_heatmap_range_frame(run_id: int, index: int, db: Session = Depends(get_db)):
        path = heatmap_service.frame_path(db, run_id, index)
        if path is None:
            raise HTTPException(status_code=404, detail="Frame not found.")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/heatmap-ranges/{run_id}/video.mp4")
    def api_get_heatmap_range_video(run_id: int, db: Session = Depends(get_db)):
        path = heatmap_service.video_path(db, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Heatmap MP4 not found.")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"heatmap-run-{run_id}.mp4",
            content_disposition_type="inline",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/heatmap-ranges/{run_id}/log", response_model=TrainingRunLogResponse)
    def api_get_heatmap_range_log(run_id: int, db: Session = Depends(get_db)):
        log = heatmap_service.read_heatmap_range_log(db, run_id)
        if log is None:
            raise HTTPException(status_code=404, detail="Heatmap range run not found.")
        return TrainingRunLogResponse(log=log)

    @app.post("/api/heatmap-ranges/{run_id}/abort", response_model=HeatmapRangeRunRead)
    def api_abort_heatmap_range(run_id: int, db: Session = Depends(get_db)):
        try:
            run = heatmap_service.abort_heatmap_range(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Heatmap range run not found.")
        return run

    @app.delete("/api/heatmap-ranges/{run_id}", status_code=204)
    def api_delete_heatmap_range(run_id: int, db: Session = Depends(get_db)):
        try:
            deleted = heatmap_service.delete_heatmap_range(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Heatmap range run not found.")
        return None

    # -- Data Manager (registry) ------------------------------------------------

    _REGISTRY_RESERVED_PARAMS = {"search", "sort", "order", "limit", "offset"}

    @app.get("/api/registry/summary")
    def api_registry_summary(db: Session = Depends(get_db)):
        return registry_service.registry_summary(db)

    @app.post("/api/registry/delete-preview")
    def api_registry_delete_preview(payload: RegistryDeleteRequest, db: Session = Depends(get_db)):
        try:
            return registry_service.delete_preview(db, [(item.entity_type, item.id) for item in payload.items])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/registry/delete")
    def api_registry_delete(payload: RegistryDeleteRequest, db: Session = Depends(get_db)):
        try:
            return registry_service.delete_entities(
                db, [(item.entity_type, item.id) for item in payload.items], cascade=payload.cascade
            )
        except registry_service.RegistryConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, RunConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/registry/{entity_type}")
    def api_registry_list(
        entity_type: str,
        request: Request,
        search: str | None = None,
        sort: str | None = None,
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
        db: Session = Depends(get_db),
    ):
        filters = {
            key: value
            for key, value in request.query_params.items()
            if key not in _REGISTRY_RESERVED_PARAMS and value
        }
        try:
            return registry_service.list_registry_rows(
                db, entity_type, search=search, filters=filters, sort=sort, order=order, limit=limit, offset=offset
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/registry/{entity_type}/{entity_id}")
    def api_registry_detail(entity_type: str, entity_id: int, db: Session = Depends(get_db)):
        try:
            detail = registry_service.get_registry_detail(db, entity_type, entity_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(status_code=404, detail="Object not found.")
        return detail

    @app.post("/api/inspect/preview", response_model=InspectPreviewResponse)
    def api_preview_inspect(payload: InspectPreviewRequest, db: Session = Depends(get_db)):
        try:
            return inspect_service.preview_inspect(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/inspect/previews/{token}.mp4")
    def api_get_inspect_preview_video(token: str):
        path = inspect_service.inspect_preview_video_path(token)
        if path is None:
            raise HTTPException(status_code=404, detail="Inspect preview not found.")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.get("/api/inspect/artifacts", response_model=InspectArtifactRunPage)
    def api_list_inspect_artifacts(
        page: int = Query(1, ge=1),
        training_dataset_id: int | None = Query(None),
        preprocessing_pipeline_id: int | None = Query(None),
        mode: str | None = Query(None),
        status: str | None = Query(None),
        db: Session = Depends(get_db),
    ):
        return inspect_service.list_inspect_artifacts(
            db,
            page=page,
            training_dataset_id=training_dataset_id,
            preprocessing_pipeline_id=preprocessing_pipeline_id,
            mode=mode,
            status=status,
        )

    @app.get("/api/inspect/runs/{run_id}/csv-data", response_model=InspectCsvData)
    def api_get_inspect_csv_data(run_id: int, db: Session = Depends(get_db)):
        result = inspect_service.read_inspect_csv_data(db, run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="CSV not found.")
        return result

    @app.post("/api/inspect/runs", response_model=InspectRunRead)
    def api_create_inspect_run(payload: InspectRunCreate, db: Session = Depends(get_db)):
        try:
            return inspect_service.create_inspect_run(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/inspect/runs", response_model=list[InspectRunRead])
    def api_list_inspect_runs(db: Session = Depends(get_db)):
        return inspect_service.list_inspect_runs(db)

    @app.get("/api/inspect/runs/{run_id}", response_model=InspectRunRead)
    def api_get_inspect_run(run_id: int, db: Session = Depends(get_db)):
        run = inspect_service.get_inspect_run(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Inspect run not found.")
        return run

    @app.post("/api/inspect/runs/{run_id}/abort", response_model=InspectRunRead)
    def api_abort_inspect_run(run_id: int, db: Session = Depends(get_db)):
        try:
            run = inspect_service.abort_inspect_run(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Inspect run not found.")
        return run

    @app.delete("/api/inspect/runs/{run_id}", status_code=204)
    def api_delete_inspect_run(run_id: int, db: Session = Depends(get_db)):
        try:
            deleted = inspect_service.delete_inspect_run(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Inspect run not found.")
        return None

    @app.get("/api/inspect/runs/{run_id}/frames/{index}.png")
    def api_get_inspect_frame(run_id: int, index: int, db: Session = Depends(get_db)):
        path = inspect_service.inspect_frame_path(db, run_id, index)
        if path is None:
            raise HTTPException(status_code=404, detail="Frame not found.")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/inspect/runs/{run_id}/video.mp4")
    def api_get_inspect_video(run_id: int, db: Session = Depends(get_db)):
        path = inspect_service.inspect_video_path(db, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Video not found.")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"inspect-run-{run_id}.mp4",
            content_disposition_type="inline",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/inspect/runs/{run_id}/results.csv")
    def api_get_inspect_csv(run_id: int, db: Session = Depends(get_db)):
        path = inspect_service.inspect_csv_path(db, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="CSV not found.")
        return FileResponse(path, media_type="text/csv", filename=f"inspect-run-{run_id}-results.csv")

    @app.get("/api/inspect/runs/{run_id}/summary.json")
    def api_get_inspect_summary(run_id: int, db: Session = Depends(get_db)):
        path = inspect_service.inspect_summary_path(db, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Summary not found.")
        return FileResponse(path, media_type="application/json", filename=f"inspect-run-{run_id}-summary.json")

    @app.get("/api/inspect/runs/{run_id}/plot-preview.png")
    def api_get_inspect_plot_preview(run_id: int, db: Session = Depends(get_db)):
        path = inspect_service.inspect_plot_preview_path(db, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Plot preview not found.")
        return FileResponse(path, media_type="image/png", filename=f"inspect-run-{run_id}-plot-preview.png")

    @app.get("/api/inspect/runs/{run_id}/log", response_model=TrainingRunLogResponse)
    def api_get_inspect_log(run_id: int, db: Session = Depends(get_db)):
        log = inspect_service.read_inspect_log(db, run_id)
        if log is None:
            raise HTTPException(status_code=404, detail="Inspect run not found.")
        return TrainingRunLogResponse(log=log)

    @app.get("/api/scheduler/settings", response_model=SchedulerSettingsRead)
    def api_get_scheduler_settings():
        return get_scheduler_settings()

    @app.put("/api/scheduler/settings", response_model=SchedulerSettingsRead)
    def api_update_scheduler_settings(payload: SchedulerSettingsUpdate):
        return update_scheduler_settings(payload.max_gpu_slots, payload.only_gpu)

    @app.get("/api/scheduler/jobs", response_model=list[SchedulerJobWithProjectRead])
    def api_list_scheduler_jobs(
        scope: str = Query(default="project", pattern="^(project|all)$"),
        x_mltrace_project_id: str | None = Header(default=None, alias="X-MLTrace-Project-ID"),
    ):
        initialize_catalog()
        if scope == "project" and not x_mltrace_project_id:
            raise HTTPException(status_code=400, detail="X-MLTrace-Project-ID header is required for project scope.")
        selected_projects = list_projects() if scope == "all" else [get_project(x_mltrace_project_id or "")]
        if any(project is None for project in selected_projects):
            raise HTTPException(status_code=404, detail="Project not found.")
        global_ranks = {
            (entry.project_id, entry.kind, entry.run_id): entry.queue_rank for entry in list_queue_entries()
        }
        jobs: list[dict] = []
        for project in selected_projects:
            assert project is not None
            with project_context(project.database_url, project.artifact_dir):
                db = SessionLocal()
                try:
                    groups = (
                        ("train", training_service.list_training_runs(db)),
                        ("test", testing_service.list_testing_runs(db)),
                        ("heatmap", heatmap_service.list_heatmap_ranges(db)),
                    )
                    for kind, runs in groups:
                        for run in runs:
                            payload = run.model_dump() if hasattr(run, "model_dump") else run
                            rank = global_ranks.get((project.id, kind, run.id))
                            if rank is not None and isinstance(payload, dict):
                                payload["queue_rank"] = rank
                            jobs.append({
                                "project_id": project.id,
                                "project_name": project.name,
                                "kind": kind,
                                "queue_rank": rank,
                                "run": payload,
                            })
                finally:
                    db.close()
        jobs.sort(key=lambda item: (
            item["run"].get("status") != "queued" if isinstance(item["run"], dict) else True,
            item["queue_rank"] if item["queue_rank"] is not None else 2**31,
            str(item["run"].get("created_at", "")) if isinstance(item["run"], dict) else "",
        ))
        return jobs

    @app.post("/api/scheduler/jobs/{kind}/{run_id}/move", response_model=SchedulerJobMoveResponse)
    def api_move_scheduler_job(
        kind: str,
        run_id: int,
        payload: SchedulerJobMoveRequest,
        request: Request,
        db: Session = Depends(get_db),
    ):
        try:
            project = getattr(request.state, "project", None)
            run = move_queued_job(db, kind, run_id, payload.direction, project.id if project else None)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Scheduler job not found.")
        return SchedulerJobMoveResponse(kind=kind, run_id=run.id, queue_rank=run.queue_rank)

    @app.get("/api/testing-runs/{run_id}/log", response_model=TrainingRunLogResponse)
    def api_get_testing_run_log(run_id: int, db: Session = Depends(get_db)):
        log = testing_service.read_testing_log(db, run_id)
        if log is None:
            raise HTTPException(status_code=404, detail="Testing run not found.")
        return TrainingRunLogResponse(log=log)

    @app.post("/api/testing-runs/{run_id}/abort", response_model=TestingRunRead)
    def api_abort_testing_run(run_id: int, db: Session = Depends(get_db)):
        try:
            run = testing_service.abort_testing_run(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Testing run not found.")
        return run

    @app.post("/api/testing-runs/{run_id}/restart", response_model=TestingRunRead)
    def api_restart_testing_run(run_id: int, db: Session = Depends(get_db)):
        try:
            run = testing_service.restart_testing_run(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Testing run not found.")
        return run

    @app.delete("/api/testing-runs/{run_id}", status_code=204)
    def api_delete_testing_run(run_id: int, db: Session = Depends(get_db)):
        try:
            deleted = testing_service.delete_testing_run(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Testing run not found.")
        return None

    # Temporary compatibility aliases for clients still calling the Phase 3 Models API.
    app.add_api_route("/api/models/architectures", api_list_method_definitions, methods=["GET"], response_model=list[MethodDefinitionRead])
    app.add_api_route(
        "/api/models/architectures/{method_type}",
        api_get_method_definition,
        methods=["GET"],
        response_model=MethodDefinitionRead,
    )
    app.add_api_route("/api/models/layers", api_list_method_layers, methods=["GET"], response_model=list[ModelLayerRead])
    app.add_api_route(
        "/api/models/configurations",
        api_list_method_configurations,
        methods=["GET"],
        response_model=list[MethodConfigurationRead],
    )
    app.add_api_route(
        "/api/models/configurations",
        api_create_method_configuration,
        methods=["POST"],
        response_model=MethodConfigurationRead,
    )
    app.add_api_route(
        "/api/models/configurations/validate",
        api_validate_method_configuration,
        methods=["POST"],
        response_model=MethodConfigurationValidationResponse,
    )
    app.add_api_route(
        "/api/models/configurations/diagram",
        api_method_configuration_diagram,
        methods=["POST"],
        response_model=MethodConfigurationValidationResponse,
    )
    app.add_api_route(
        "/api/models/configurations/torch-check",
        api_method_configuration_torch_check,
        methods=["POST"],
        response_model=MethodTorchCheckResponse,
    )
    app.add_api_route(
        "/api/models/configurations/{configuration_id}",
        api_get_method_configuration,
        methods=["GET"],
        response_model=MethodConfigurationRead,
    )
    app.add_api_route(
        "/api/models/configurations/{configuration_id}",
        api_update_method_configuration,
        methods=["PUT"],
        response_model=MethodConfigurationRead,
    )
    app.add_api_route(
        "/api/models/configurations/{configuration_id}",
        api_delete_method_configuration,
        methods=["DELETE"],
        status_code=204,
    )

    return app


app = create_app()
