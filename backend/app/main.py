from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.schemas import (
    DatasetConnectionTestRequest,
    DatasetConnectionTestResponse,
    DatasetCreate,
    DatasetRead,
    HeatmapRunCreate,
    HeatmapRunRead,
    MethodConfigurationCreate,
    MethodConfigurationPayload,
    MethodConfigurationRead,
    MethodTorchCheckResponse,
    MethodConfigurationValidationResponse,
    MethodDefinitionRead,
    ModelLayerRead,
    PreprocessingPipelineCreate,
    PreprocessingPipelineRead,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResponse,
    PreprocessingStepRead,
    RoiDefinitionCreate,
    RoiDefinitionRead,
    RoiPreviewRequest,
    RoiPreviewResponse,
    SchedulerSettingsRead,
    SchedulerSettingsUpdate,
    TestingRunCreate,
    TestingRunResultImageResponse,
    TestingRunRead,
    TestingRunResultsResponse,
    TimestampFormatConfirm,
    TrainingDatasetCreate,
    TrainingDatasetPreviewRequest,
    TrainingDatasetPreviewResponse,
    TrainingDatasetRead,
    TrainingPipelineCreate,
    TrainingPipelineDryRunRequest,
    TrainingPipelineDryRunResponse,
    TrainingPipelineDuplicateResponse,
    TrainingPipelinePayload,
    TrainingPipelineRead,
    TrainingRunEnqueueRequest,
    TrainingRunLogResponse,
    TrainingRunRead,
)
from app.testing import service as testing_service
from app.testing.service import TestingConflict
from app.training import service as training_service
from app.training.scheduler import get_scheduler_settings, scheduler, update_scheduler_settings
from app.training.service import RunConflict
from app.services import (
    DuplicatePipelineError,
    create_dataset,
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
    run_method_torch_check,
    scan_dataset,
    test_dataset_connection,
    update_method_configuration,
    update_preprocessing_pipeline,
    update_training_pipeline,
    validate_method_configuration,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background training scheduler with the API process. Detached
    # worker subprocesses survive an API restart and are reconciled on startup.
    scheduler.start()
    try:
        yield
    finally:
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

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
        return dataset

    @app.post("/api/datasets/{dataset_id}/confirm-timestamp-format", response_model=DatasetRead)
    def api_confirm_timestamp_format(
        dataset_id: int, payload: TimestampFormatConfirm, db: Session = Depends(get_db)
    ):
        dataset = get_dataset_or_404(db, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return scan_dataset(db, dataset, payload.timestamp_regex, payload.timestamp_format)

    @app.post("/api/datasets/{dataset_id}/rescan", response_model=DatasetRead)
    def api_rescan_dataset(dataset_id: int, db: Session = Depends(get_db)):
        dataset = get_dataset_or_404(db, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        if not dataset.timestamp_regex or not dataset.timestamp_format:
            raise HTTPException(status_code=400, detail="Timestamp format has not been confirmed yet.")
        return scan_dataset(db, dataset, dataset.timestamp_regex, dataset.timestamp_format)

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

    @app.get("/api/training-datasets", response_model=list[TrainingDatasetRead])
    def api_list_training_datasets(db: Session = Depends(get_db)):
        return list_training_datasets(db)

    @app.get("/api/training-datasets/{training_dataset_id}", response_model=TrainingDatasetRead)
    def api_get_training_dataset(training_dataset_id: int, db: Session = Depends(get_db)):
        training_dataset = get_training_dataset(db, training_dataset_id)
        if training_dataset is None:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        return training_dataset

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

    @app.get("/api/preprocessing/pipelines", response_model=list[PreprocessingPipelineRead])
    def api_list_preprocessing_pipelines(db: Session = Depends(get_db)):
        return list_preprocessing_pipelines(db)

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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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

    @app.get("/api/methods/configurations", response_model=list[MethodConfigurationRead])
    def api_list_method_configurations(db: Session = Depends(get_db)):
        return list_method_configurations(db)

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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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

    @app.get("/api/training-pipelines", response_model=list[TrainingPipelineRead])
    def api_list_training_pipelines(db: Session = Depends(get_db)):
        return list_training_pipelines(db)

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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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

    @app.get("/api/testing-runs/{run_id}", response_model=TestingRunRead)
    def api_get_testing_run(run_id: int, db: Session = Depends(get_db)):
        run = testing_service.get_testing_run(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Testing run not found.")
        return run

    @app.get("/api/testing-runs/{run_id}/results", response_model=TestingRunResultsResponse)
    def api_get_testing_run_results(run_id: int, db: Session = Depends(get_db)):
        response = testing_service.get_testing_run_results(db, run_id)
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

    @app.get("/api/heatmaps", response_model=list[HeatmapRunRead])
    def api_list_heatmaps(db: Session = Depends(get_db)):
        return testing_service.list_heatmap_runs(db)

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

    @app.get("/api/scheduler/settings", response_model=SchedulerSettingsRead)
    def api_get_scheduler_settings():
        return get_scheduler_settings()

    @app.put("/api/scheduler/settings", response_model=SchedulerSettingsRead)
    def api_update_scheduler_settings(payload: SchedulerSettingsUpdate):
        return update_scheduler_settings(payload.max_gpu_slots, payload.only_gpu)

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
