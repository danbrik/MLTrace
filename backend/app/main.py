from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.schemas import (
    DatasetCreate,
    DatasetRead,
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
)
from app.services import (
    create_dataset,
    create_preprocessing_pipeline,
    create_training_dataset,
    delete_preprocessing_pipeline,
    delete_training_dataset,
    get_dataset_or_404,
    get_preprocessing_pipeline,
    get_training_dataset,
    list_datasets,
    list_preprocessing_pipelines,
    list_preprocessing_steps,
    list_training_datasets,
    preview_preprocessing_pipeline,
    preview_training_dataset,
    scan_dataset,
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
        deleted = delete_training_dataset(db, training_dataset_id)
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

    @app.get("/api/preprocessing/pipelines/{pipeline_id}", response_model=PreprocessingPipelineRead)
    def api_get_preprocessing_pipeline(pipeline_id: int, db: Session = Depends(get_db)):
        pipeline = get_preprocessing_pipeline(db, pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Preprocessing pipeline not found.")
        return pipeline

    @app.delete("/api/preprocessing/pipelines/{pipeline_id}", status_code=204)
    def api_delete_preprocessing_pipeline(pipeline_id: int, db: Session = Depends(get_db)):
        deleted = delete_preprocessing_pipeline(db, pipeline_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Preprocessing pipeline not found.")
        return None

    @app.post("/api/preprocessing/pipelines/preview", response_model=PreprocessingPreviewResponse)
    def api_preview_preprocessing_pipeline(payload: PreprocessingPreviewRequest, db: Session = Depends(get_db)):
        try:
            return preview_preprocessing_pipeline(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
