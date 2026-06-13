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
    TimestampFormatConfirm,
    TrainingDatasetCreate,
    TrainingDatasetPreviewRequest,
    TrainingDatasetPreviewResponse,
    TrainingDatasetRead,
    TrainingPipelineCreate,
    TrainingPipelineDryRunRequest,
    TrainingPipelineDryRunResponse,
    TrainingPipelineRead,
)
from app.services import (
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


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="MLTrace API", version="0.1.0")
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="A training pipeline with this name already exists.") from exc

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
        deleted = delete_training_pipeline(db, pipeline_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Training pipeline not found.")
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
