from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base


def json_type():
    return JSON().with_variant(JSONB(), "postgresql")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    root_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="awaiting_confirmation")
    timestamp_regex: Mapped[str | None] = mapped_column(Text)
    timestamp_format: Mapped[str | None] = mapped_column(String(128))
    timestamp_example: Mapped[str | None] = mapped_column(String(255))
    scan_error: Mapped[str | None] = mapped_column(Text)
    scan_summary: Mapped[dict | None] = mapped_column(json_type())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    folders: Mapped[list["DatasetFolder"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    images: Mapped[list["DatasetImage"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    training_datasets: Mapped[list["TrainingDataset"]] = relationship(back_populates="dataset")


class DatasetFolder(Base):
    __tablename__ = "dataset_folders"
    __table_args__ = (UniqueConstraint("dataset_id", "relative_path", name="uq_folder_per_dataset"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    last_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    extension_summary: Mapped[dict | None] = mapped_column(json_type())
    resolution_summary: Mapped[dict | None] = mapped_column(json_type())
    image_metadata: Mapped[dict | None] = mapped_column(json_type())
    cadence_summary: Mapped[dict | None] = mapped_column(json_type())
    filename_template: Mapped[dict | None] = mapped_column(json_type())

    dataset: Mapped[Dataset] = relationship(back_populates="folders")
    images: Mapped[list["DatasetImage"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )
    training_rules: Mapped[list["TrainingDatasetRule"]] = relationship(back_populates="folder")


class DatasetImage(Base):
    __tablename__ = "dataset_images"
    __table_args__ = (
        Index("ix_dataset_images_dataset_timestamp", "dataset_id", "timestamp_parsed"),
        Index("ix_dataset_images_folder_timestamp", "folder_id", "timestamp_parsed"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    folder_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_folders.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    extension: Mapped[str] = mapped_column(String(16), nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    timestamp_raw: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp_parsed: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    modified_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))

    dataset: Mapped[Dataset] = relationship(back_populates="images")
    folder: Mapped[DatasetFolder] = relationship(back_populates="images")


class TrainingDataset(Base):
    __tablename__ = "training_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id", ondelete="RESTRICT"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    usage_label: Mapped[str] = mapped_column(String(32), nullable=False, default="train")
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

    dataset: Mapped[Dataset | None] = relationship(back_populates="training_datasets")
    rules: Mapped[list["TrainingDatasetRule"]] = relationship(
        back_populates="training_dataset", cascade="all, delete-orphan"
    )


class TrainingDatasetRule(Base):
    __tablename__ = "training_dataset_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="CASCADE"), nullable=False
    )
    folder_id: Mapped[int] = mapped_column(ForeignKey("dataset_folders.id", ondelete="RESTRICT"), nullable=False)
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    stride: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Persisted at preview/create/update/refresh time. Listing Train/Test Sets
    # must not enumerate large image folders just to render metadata.
    matching_images: Mapped[int | None] = mapped_column(Integer)
    selected_images: Mapped[int | None] = mapped_column(Integer)

    training_dataset: Mapped[TrainingDataset] = relationship(back_populates="rules")
    folder: Mapped[DatasetFolder] = relationship(back_populates="training_rules")


class PreprocessingPipeline(Base):
    __tablename__ = "preprocessing_pipelines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    graph: Mapped[dict] = mapped_column(json_type(), nullable=False)
    preview_folder_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_folders.id", ondelete="SET NULL"))
    # Design resolution the pipeline was built/optimised on (captured from a preview).
    input_width: Mapped[int | None] = mapped_column(Integer)
    input_height: Mapped[int | None] = mapped_column(Integer)
    output_width: Mapped[int | None] = mapped_column(Integer)
    output_height: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )


class MethodConfiguration(Base):
    __tablename__ = "method_configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    method_type: Mapped[str] = mapped_column(String(128), nullable=False)
    method_family: Mapped[str] = mapped_column(String(128), nullable=False)
    method_version: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    training_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    requires_training: Mapped[bool] = mapped_column(nullable=False, default=True)
    supports_training_pipeline: Mapped[bool] = mapped_column(nullable=False, default=True)
    artifact_kind: Mapped[str] = mapped_column(String(128), nullable=False)
    builder_kind: Mapped[str] = mapped_column(String(128), nullable=False)
    method_graph: Mapped[dict] = mapped_column(json_type(), nullable=False)
    method_config: Mapped[dict] = mapped_column(json_type(), nullable=False)
    training_config: Mapped[dict] = mapped_column(json_type(), nullable=False)
    inference_config: Mapped[dict] = mapped_column(json_type(), nullable=False)
    diagram: Mapped[dict] = mapped_column(json_type(), nullable=False)
    validation: Mapped[dict | None] = mapped_column(json_type())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    parameters: Mapped[list["MethodConfigurationParameter"]] = relationship(
        back_populates="method_configuration", cascade="all, delete-orphan"
    )


class MethodConfigurationParameter(Base):
    __tablename__ = "method_configuration_parameters"
    __table_args__ = (
        Index("ix_method_config_parameters_path_text", "path", "value_text"),
        Index("ix_method_config_parameters_path_number", "path", "value_number"),
        Index("ix_method_config_parameters_path_bool", "path", "value_bool"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    method_configuration_id: Mapped[int] = mapped_column(
        ForeignKey("method_configurations.id", ondelete="CASCADE"), nullable=False
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(32), nullable=False)
    value_text: Mapped[str | None] = mapped_column(Text)
    value_number: Mapped[float | None] = mapped_column()
    value_bool: Mapped[bool | None] = mapped_column()

    method_configuration: Mapped[MethodConfiguration] = relationship(back_populates="parameters")


class TrainingPipeline(Base):
    """A saved training composition: N training datasets -> one preprocessing
    pipeline -> one method configuration, plus a frozen copy of the training
    parameters.

    This is a declarative definition only; executing the training run happens
    elsewhere. Building blocks are referenced by FK (not snapshotted), so the
    delete services guard against removing blocks that are still in use.
    """

    __tablename__ = "training_pipelines"
    __table_args__ = (Index("ix_training_pipelines_config_signature", "config_signature"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    # Hash of the full configuration (datasets + preprocessing + method + shuffle
    # + training params), independent of name. Used to block duplicate pipelines.
    config_signature: Mapped[str | None] = mapped_column(String(64))
    preprocessing_pipeline_id: Mapped[int] = mapped_column(
        ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"), nullable=False
    )
    method_configuration_id: Mapped[int] = mapped_column(
        ForeignKey("method_configurations.id", ondelete="RESTRICT"), nullable=False
    )
    # Whether the combined training sets get shuffled when the run is executed.
    shuffle: Mapped[bool] = mapped_column(nullable=False, default=True)
    # Final merged training parameters (method training_config + user overrides),
    # validated against the method definition's training_schema at save time.
    training_parameters: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    entries: Mapped[list["TrainingPipelineDataset"]] = relationship(
        back_populates="training_pipeline",
        cascade="all, delete-orphan",
        order_by="TrainingPipelineDataset.position",
    )
    preprocessing_pipeline: Mapped[PreprocessingPipeline] = relationship()
    method_configuration: Mapped[MethodConfiguration] = relationship()


class TrainingPipelineDataset(Base):
    """Ordered association between a training pipeline and its training datasets."""

    __tablename__ = "training_pipeline_datasets"
    __table_args__ = (
        UniqueConstraint("training_pipeline_id", "training_dataset_id", name="uq_training_pipeline_dataset"),
        UniqueConstraint("training_pipeline_id", "position", name="uq_training_pipeline_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_pipeline_id: Mapped[int] = mapped_column(
        ForeignKey("training_pipelines.id", ondelete="CASCADE"), nullable=False
    )
    training_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    training_pipeline: Mapped[TrainingPipeline] = relationship(back_populates="entries")
    training_dataset: Mapped[TrainingDataset] = relationship()


class TrainingRun(Base):
    """Execution state of a training pipeline: queue position, live status,
    metrics, and the resulting artifact.

    Exactly one run exists per training pipeline (UNIQUE training_pipeline_id) —
    "restart" resets this row rather than creating history. Filterable pipeline
    properties are denormalized onto the row so the runs overview can be queried
    and sorted from a single indexed table without joins.
    """

    __tablename__ = "training_runs"
    __table_args__ = (
        UniqueConstraint("training_pipeline_id", name="uq_training_run_pipeline"),
        Index("ix_training_runs_status", "status"),
        Index("ix_training_runs_method_type", "method_type"),
        Index("ix_training_runs_training_mode", "training_mode"),
        Index("ix_training_runs_builder_kind", "builder_kind"),
        Index("ix_training_runs_created_at", "created_at"),
        Index("ix_training_runs_val_loss", "val_loss"),
        Index("ix_training_runs_train_loss", "train_loss"),
        Index("ix_training_runs_duration", "duration_seconds"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_pipeline_id: Mapped[int] = mapped_column(
        ForeignKey("training_pipelines.id", ondelete="CASCADE"), nullable=False
    )

    # Execution lifecycle.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    gpu_index: Mapped[int | None] = mapped_column(Integer)
    # Actual compute device used by the worker ("CPU" or "GPU:<index>"), set at
    # runtime — reflects the CPU fallback when no CUDA device is available.
    device: Mapped[str | None] = mapped_column(String(32))
    pid: Mapped[int | None] = mapped_column(Integer)
    log_path: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    # Progress and metrics (train/val loss are null for fit-style methods).
    epochs_total: Mapped[int | None] = mapped_column(Integer)
    epochs_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float)
    best_val_loss: Mapped[float | None] = mapped_column(Float)
    image_count: Mapped[int | None] = mapped_column(Integer)

    # Artifact (model weights or mean image) written to disk.
    artifact_kind: Mapped[str | None] = mapped_column(String(64))
    artifact_path: Mapped[str | None] = mapped_column(Text)
    artifact_size_bytes: Mapped[int | None] = mapped_column(BigInteger)

    # Denormalized pipeline snapshot for single-table filtering / sorting.
    training_pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    method_type: Mapped[str] = mapped_column(String(128), nullable=False)
    method_family: Mapped[str] = mapped_column(String(128), nullable=False)
    training_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    builder_kind: Mapped[str] = mapped_column(String(128), nullable=False)
    preprocessing_pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_names: Mapped[list] = mapped_column(json_type(), nullable=False)
    dataset_names_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    shuffle: Mapped[bool] = mapped_column(nullable=False, default=True)
    input_resolution: Mapped[str | None] = mapped_column(String(32))
    epochs: Mapped[int | None] = mapped_column(Integer)
    learning_rate: Mapped[float | None] = mapped_column(Float)
    training_parameters: Mapped[dict] = mapped_column(json_type(), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    training_pipeline: Mapped[TrainingPipeline] = relationship()
    metrics: Mapped[list["TrainingRunMetric"]] = relationship(
        back_populates="training_run",
        cascade="all, delete-orphan",
        order_by="TrainingRunMetric.epoch",
    )


class TrainingRunMetric(Base):
    """Per-epoch loss curve point for a gradient-trained run."""

    __tablename__ = "training_run_metrics"
    __table_args__ = (Index("ix_training_run_metrics_run_epoch", "training_run_id", "epoch"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False
    )
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

    training_run: Mapped[TrainingRun] = relationship(back_populates="metrics")


class RoiDefinition(Base):
    """Reusable ROI in preprocessed image coordinates.

    Older rows may only contain the rectangular x/y/width/height fields. New
    rows store four ordered points (top-left, top-right, bottom-right,
    bottom-left) so perspective-like quadrilateral ROIs can be reused across
    testing runs.
    """

    __tablename__ = "roi_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    x: Mapped[int] = mapped_column(Integer, nullable=False)
    y: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    geometry_type: Mapped[str] = mapped_column(String(32), nullable=False, default="polygon")
    points: Mapped[list | None] = mapped_column(json_type())
    tile_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    tile_cols: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )


class TestingRun(Base):
    """A saved testing execution over one train/test dataset and one trained artifact."""

    __tablename__ = "testing_runs"
    __table_args__ = (
        Index("ix_testing_runs_status", "status"),
        Index("ix_testing_runs_created_at", "created_at"),
        Index("ix_testing_runs_score_mean", "score_mean"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="RESTRICT"), nullable=False)
    training_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    roi_id: Mapped[int | None] = mapped_column(ForeignKey("roi_definitions.id", ondelete="SET NULL"))

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    # Scheduler/queue fields, mirroring TrainingRun (testing now runs as queued
    # background jobs through the shared scheduler instead of synchronously).
    gpu_index: Mapped[int | None] = mapped_column(Integer)
    device: Mapped[str | None] = mapped_column(String(32))
    pid: Mapped[int | None] = mapped_column(Integer)
    log_path: Mapped[str | None] = mapped_column(Text)

    image_count: Mapped[int | None] = mapped_column(Integer)
    expected_image_count: Mapped[int | None] = mapped_column(Integer)
    score_mean: Mapped[float | None] = mapped_column(Float)
    score_min: Mapped[float | None] = mapped_column(Float)
    score_max: Mapped[float | None] = mapped_column(Float)
    full_mse_mean: Mapped[float | None] = mapped_column(Float)
    roi_mse_mean: Mapped[float | None] = mapped_column(Float)
    results_path: Mapped[str | None] = mapped_column(Text)
    results_size_bytes: Mapped[int | None] = mapped_column(BigInteger)

    # Denormalized snapshot for stable filtering/display even when source
    # objects are renamed later.
    training_run_name: Mapped[str] = mapped_column(String(255), nullable=False)
    training_pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    training_dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    preprocessing_pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    method_type: Mapped[str] = mapped_column(String(128), nullable=False)
    method_family: Mapped[str] = mapped_column(String(128), nullable=False)
    training_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    roi_name: Mapped[str | None] = mapped_column(String(255))
    roi_geometry: Mapped[dict | None] = mapped_column(json_type())

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    training_run: Mapped[TrainingRun] = relationship()
    training_dataset: Mapped[TrainingDataset] = relationship()
    roi: Mapped[RoiDefinition | None] = relationship()
    results: Mapped[list["TestingRunResult"]] = relationship(
        back_populates="testing_run",
        cascade="all, delete-orphan",
        order_by="TestingRunResult.position",
    )


class TestingRunResult(Base):
    """Per-image reconstruction/error row for one testing run."""

    __tablename__ = "testing_run_results"
    __table_args__ = (
        Index("ix_testing_run_results_run_position", "testing_run_id", "position"),
        Index("ix_testing_run_results_timestamp", "timestamp"),
        Index("ix_testing_run_results_score", "score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    testing_run_id: Mapped[int] = mapped_column(ForeignKey("testing_runs.id", ondelete="CASCADE"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    full_mse: Mapped[float] = mapped_column(Float, nullable=False)
    roi_mse: Mapped[float | None] = mapped_column(Float)
    tile_scores: Mapped[list | None] = mapped_column(json_type())
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

    testing_run: Mapped[TestingRun] = relationship(back_populates="results")


class HeatmapRun(Base):
    """Cached CPU per-pixel reconstruction error heatmap for one testing image."""

    __tablename__ = "heatmap_runs"
    __table_args__ = (
        UniqueConstraint("testing_run_id", "testing_result_id", name="uq_heatmap_result"),
        Index("ix_heatmap_runs_testing_run_timestamp", "testing_run_id", "timestamp"),
        Index("ix_heatmap_runs_created_at", "created_at"),
        Index("ix_heatmap_runs_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    testing_run_id: Mapped[int] = mapped_column(ForeignKey("testing_runs.id", ondelete="CASCADE"), nullable=False)
    testing_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("testing_run_results.id", ondelete="CASCADE"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="finished")
    error_message: Mapped[str | None] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, nullable=False)
    dtype: Mapped[str] = mapped_column(String(64), nullable=False)
    max_error: Mapped[float] = mapped_column(Float, nullable=False)
    mean_error: Mapped[float] = mapped_column(Float, nullable=False)
    max_x: Mapped[int] = mapped_column(Integer, nullable=False)
    max_y: Mapped[int] = mapped_column(Integer, nullable=False)
    source_image_data_url: Mapped[str] = mapped_column(Text, nullable=False)
    reconstruction_image_data_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Full-resolution (pixel-exact) raw per-pixel error grid for the scientific
    # Plotly heatmap (colorbar, axes). Null for heatmaps computed before 0024.
    error_matrix: Mapped[list | None] = mapped_column(json_type())
    heatmap_image_data_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    testing_run: Mapped[TestingRun] = relationship()
    testing_result: Mapped[TestingRunResult | None] = relationship()


class HeatmapRangeRun(Base):
    """A queued batch job that renders pixel-error overlay PNG frames for a time
    range of one testing run, played back as a fast heatmap video. Runs through
    the shared scheduler (kind ``heatmap``) as a GPU-pinned worker subprocess."""

    __tablename__ = "heatmap_range_runs"
    __table_args__ = (
        Index("ix_heatmap_range_runs_status", "status"),
        Index("ix_heatmap_range_runs_created_at", "created_at"),
        Index("ix_heatmap_range_runs_signature", "config_signature"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    testing_run_id: Mapped[int] = mapped_column(ForeignKey("testing_runs.id", ondelete="CASCADE"), nullable=False)

    # Scheduler/queue fields, mirroring TestingRun.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    gpu_index: Mapped[int | None] = mapped_column(Integer)
    device: Mapped[str | None] = mapped_column(String(32))
    pid: Mapped[int | None] = mapped_column(Integer)
    log_path: Mapped[str | None] = mapped_column(Text)

    # Range selection + render parameters.
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    stride: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    scale_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="per_frame")
    global_vmax: Mapped[float | None] = mapped_column(Float)

    # Progress counter (done_count / frame_count) + output location.
    frame_count: Mapped[int | None] = mapped_column(Integer)
    done_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frames_dir: Mapped[str | None] = mapped_column(Text)

    # Dedup signature over (testing_run_id, start, end, stride, scale_mode).
    config_signature: Mapped[str] = mapped_column(String(64), nullable=False)

    # Denormalized snapshot for stable display.
    testing_run_name: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    testing_run: Mapped[TestingRun] = relationship()


ModelConfiguration = MethodConfiguration
ModelConfigurationParameter = MethodConfigurationParameter
