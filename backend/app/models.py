from datetime import UTC, datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base


def json_type():
    return JSON().with_variant(JSONB(), "postgresql")


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utc_now, server_default=func.now(), onupdate=utc_now
    )

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
    queue_rank: Mapped[int | None] = mapped_column(Integer)
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
    # Corrupt/unreadable source images skipped during the run ("skip + report").
    skipped_image_count: Mapped[int | None] = mapped_column(Integer)
    skipped_images: Mapped[list | None] = mapped_column(json_type())

    # Artifact (model weights or mean image) written to disk.
    artifact_kind: Mapped[str | None] = mapped_column(String(64))
    artifact_path: Mapped[str | None] = mapped_column(Text)
    # Stable digest of the completed artifact.  Evaluation uses this rather
    # than display names to ensure all score sources belong to the same model.
    artifact_signature: Mapped[str | None] = mapped_column(String(64))
    artifact_size_bytes: Mapped[int | None] = mapped_column(BigInteger)

    # Single durable resume checkpoint for long gradient trainings.
    checkpoint_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    checkpoint_epoch: Mapped[int | None] = mapped_column(Integer)
    checkpoint_phase: Mapped[str | None] = mapped_column(String(32))
    checkpoint_iteration: Mapped[int | None] = mapped_column(Integer)
    checkpoint_path: Mapped[str | None] = mapped_column(Text)
    checkpoint_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checkpoint_signature: Mapped[str | None] = mapped_column(String(64))
    checkpoint_warning: Mapped[str | None] = mapped_column(Text)
    restart_mode: Mapped[str | None] = mapped_column(String(32))
    resume_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    auto_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))

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
    queue_rank: Mapped[int | None] = mapped_column(Integer)
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
    # Corrupt/unreadable source images skipped during the run ("skip + report").
    skipped_image_count: Mapped[int | None] = mapped_column(Integer)
    skipped_images: Mapped[list | None] = mapped_column(json_type())
    score_mean: Mapped[float | None] = mapped_column(Float)
    score_min: Mapped[float | None] = mapped_column(Float)
    score_max: Mapped[float | None] = mapped_column(Float)
    full_mse_mean: Mapped[float | None] = mapped_column(Float)
    roi_mse_mean: Mapped[float | None] = mapped_column(Float)
    results_path: Mapped[str | None] = mapped_column(Text)
    results_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    # Durable resume marker for long scheduled inference runs. Result rows are
    # already committed batch-by-batch; this state records the last consistent
    # source cursor and running aggregates that may safely be resumed.
    checkpoint_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    checkpoint_input_count: Mapped[int | None] = mapped_column(Integer)
    checkpoint_result_count: Mapped[int | None] = mapped_column(Integer)
    checkpoint_state: Mapped[dict | None] = mapped_column(json_type())
    restart_mode: Mapped[str | None] = mapped_column(String(32))

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
    artifact_signature: Mapped[str | None] = mapped_column(String(64))
    # Monotone generation of the persisted score rows.  Evaluation snapshots
    # use this instead of scanning the (potentially very large) result table.
    result_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    roi_name: Mapped[str | None] = mapped_column(String(255))
    roi_geometry: Mapped[dict | None] = mapped_column(json_type())
    inference_config: Mapped[dict | None] = mapped_column(json_type())

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
    anomaly_detection_runs: Mapped[list["AnomalyDetectionRun"]] = relationship(
        back_populates="testing_run",
        cascade="all, delete-orphan",
        foreign_keys="AnomalyDetectionRun.testing_run_id",
    )


class TestingRunResult(Base):
    """Per-image reconstruction/error row for one testing run."""

    __tablename__ = "testing_run_results"
    __table_args__ = (
        Index("ix_testing_run_results_run_position", "testing_run_id", "position"),
        Index("ix_testing_run_results_run_timestamp", "testing_run_id", "timestamp"),
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
    result_metadata: Mapped[dict | None] = mapped_column(json_type())
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

    testing_run: Mapped[TestingRun] = relationship(back_populates="results")


class AnomalyDetectionRun(Base):
    """Saved causal anomaly detection over one testing-run score series."""

    __tablename__ = "anomaly_detection_runs"
    __table_args__ = (
        Index("ix_anomaly_detection_runs_created_at", "created_at"),
        Index("ix_anomaly_detection_runs_testing_run", "testing_run_id"),
        Index("ix_anomaly_detection_runs_threshold_testing_run", "threshold_testing_run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    testing_run_id: Mapped[int] = mapped_column(
        ForeignKey("testing_runs.id", ondelete="CASCADE"), nullable=False
    )
    testing_run_name: Mapped[str] = mapped_column(String(255), nullable=False)
    threshold_testing_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("testing_runs.id", ondelete="CASCADE")
    )
    threshold_testing_run_name: Mapped[str | None] = mapped_column(String(255))
    threshold_start_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    threshold_end_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    resolved_threshold: Mapped[float | None] = mapped_column(Float)
    score_series: Mapped[str] = mapped_column(String(32), nullable=False, default="score")
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(32), nullable=False, default="robust_cusum_v1")
    config: Mapped[dict] = mapped_column(json_type(), nullable=False)
    point_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    anomaly_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    testing_run: Mapped[TestingRun] = relationship(
        back_populates="anomaly_detection_runs",
        foreign_keys=[testing_run_id],
    )
    threshold_testing_run: Mapped[TestingRun | None] = relationship(
        foreign_keys=[threshold_testing_run_id]
    )
    events: Mapped[list["AnomalyDetectionEvent"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AnomalyDetectionEvent.warning_start",
    )


class AnomalyDetectionEvent(Base):
    """One warning interval, optionally promoted to a confirmed anomaly."""

    __tablename__ = "anomaly_detection_events"
    __table_args__ = (Index("ix_anomaly_detection_events_run_start", "run_id", "warning_start"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("anomaly_detection_runs.id", ondelete="CASCADE"), nullable=False
    )
    warning_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    peak_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    max_score: Mapped[float] = mapped_column(Float, nullable=False)
    max_robust_z: Mapped[float | None] = mapped_column(Float)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    max_smoothed_score: Mapped[float | None] = mapped_column(Float)
    mean_smoothed_score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)

    run: Mapped[AnomalyDetectionRun] = relationship(back_populates="events")


class HeatmapRun(Base):
    """Cached CPU per-pixel reconstruction error heatmap for one testing image."""

    __tablename__ = "heatmap_runs"
    __table_args__ = (
        UniqueConstraint(
            "testing_run_id",
            "testing_result_id",
            "config_signature",
            name="uq_heatmap_result_config",
        ),
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
    # Heavy artifacts (source/reconstruction/overlay PNG + error matrix) live on
    # disk under ``artifacts_dir`` to keep the DB small; these inline columns stay
    # empty for new rows and are only read as a fallback for pre-disk rows.
    artifacts_dir: Mapped[str | None] = mapped_column(Text)
    source_image_data_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reconstruction_image_data_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Full-resolution configured pixel-error grid for the interactive Plotly
    # overlay. It may be signed when directional deviations are enabled.
    error_matrix: Mapped[list | None] = mapped_column(json_type())
    heatmap_image_data_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    visualization_config: Mapped[dict] = mapped_column(json_type(), nullable=False, default=dict)
    config_signature: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    render_version: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
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
    queue_rank: Mapped[int | None] = mapped_column(Integer)
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
    fps: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    scale_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="per_frame")
    stae_view: Mapped[str] = mapped_column(String(32), nullable=False, default="reconstruction")
    prediction_horizon: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    global_vmax: Mapped[float | None] = mapped_column(Float)
    frame_max_errors: Mapped[list | None] = mapped_column(json_type())
    visualization_config: Mapped[dict] = mapped_column(json_type(), nullable=False, default=dict)
    render_version: Mapped[int] = mapped_column(Integer, nullable=False, default=4)

    # Progress counter (done_count / frame_count) + output location.
    frame_count: Mapped[int | None] = mapped_column(Integer)
    done_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frames_dir: Mapped[str | None] = mapped_column(Text)
    video_path: Mapped[str | None] = mapped_column(Text)

    # Dedup signature includes source range, scale mode, visualization config,
    # and render version.
    config_signature: Mapped[str] = mapped_column(String(64), nullable=False)

    # Denormalized snapshot for stable display.
    testing_run_name: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    testing_run: Mapped[TestingRun] = relationship()


class ImageDistributionRun(Base):
    """Queued temporal image-distribution analysis shown in the shared scheduler."""

    __tablename__ = "image_distribution_runs"
    __table_args__ = (
        Index("ix_image_distribution_runs_status", "status"),
        Index("ix_image_distribution_runs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    preprocessing_pipeline_id: Mapped[int] = mapped_column(
        ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    queue_rank: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    gpu_index: Mapped[int | None] = mapped_column(Integer)
    device: Mapped[str | None] = mapped_column(String(32))
    pid: Mapped[int | None] = mapped_column(Integer)
    log_path: Mapped[str | None] = mapped_column(Text)
    current_step: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    total_images: Mapped[int | None] = mapped_column(Integer)
    processed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_key: Mapped[str | None] = mapped_column(String(64))
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    csv_path: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict | None] = mapped_column(json_type())
    training_dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    usage_label: Mapped[str] = mapped_column(String(32), nullable=False)
    preprocessing_pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )


class InspectRun(Base):
    """CPU-only preprocessing inspection video over a selected train/test range."""

    __tablename__ = "inspect_runs"
    __table_args__ = (
        Index("ix_inspect_runs_status", "status"),
        Index("ix_inspect_runs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    preprocessing_pipeline_id: Mapped[int] = mapped_column(
        ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"), nullable=False
    )

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    device: Mapped[str | None] = mapped_column(String(32))
    pid: Mapped[int | None] = mapped_column(Integer)
    log_path: Mapped[str | None] = mapped_column(Text)

    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    stride: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    fps: Mapped[int] = mapped_column(Integer, nullable=False, default=12)
    content_mode: Mapped[str] = mapped_column(String(64), nullable=False, default="final_preprocessed_output")

    analysis_mode: Mapped[str] = mapped_column(String(64), nullable=False, default="preprocessed_video")
    analysis_config: Mapped[dict | None] = mapped_column(json_type())
    roi_id: Mapped[int | None] = mapped_column(ForeignKey("roi_definitions.id", ondelete="SET NULL"))
    generate_video: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    contrast_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    contrast_reference_frames: Mapped[int | None] = mapped_column(Integer)
    contrast_shift: Mapped[float | None] = mapped_column(Float)
    contrast_vmax: Mapped[float | None] = mapped_column(Float)
    contrast_ma_radius: Mapped[int | None] = mapped_column(Integer)

    frame_count: Mapped[int | None] = mapped_column(Integer)
    done_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frames_dir: Mapped[str | None] = mapped_column(Text)
    video_path: Mapped[str | None] = mapped_column(Text)
    csv_path: Mapped[str | None] = mapped_column(Text)
    summary_json_path: Mapped[str | None] = mapped_column(Text)
    plot_preview_path: Mapped[str | None] = mapped_column(Text)
    overlay_video_path: Mapped[str | None] = mapped_column(Text)

    training_dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    preprocessing_pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    training_dataset: Mapped[TrainingDataset] = relationship()
    preprocessing_pipeline: Mapped[PreprocessingPipeline] = relationship()
    roi: Mapped[RoiDefinition | None] = relationship()


class AnalysisLayout(Base):
    """Named, reusable Analysis board state.

    The JSON payload stores UI configuration only: plot definitions, selected
    pipeline/ROI/source filters, and the current draft. Result rows and heatmap
    caches are intentionally reloaded from their source APIs when a layout is
    opened again.
    """

    __tablename__ = "analysis_layouts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    layout: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )


class OptimizationStudy(Base):
    """An Optuna-style hyperparameter study orchestrated through MLTrace runs."""

    __tablename__ = "optimization_studies"
    __table_args__ = (
        Index("ix_optimization_studies_status", "status"),
        Index("ix_optimization_studies_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    objective_name: Mapped[str] = mapped_column(String(64), nullable=False, default="median_anomaly_minus_p95_normal")
    direction: Mapped[str] = mapped_column(String(16), nullable=False, default="maximize")
    n_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    max_parallel_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    sampler: Mapped[str] = mapped_column(String(64), nullable=False, default="tpe")

    preprocessing_pipeline_id: Mapped[int] = mapped_column(
        ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"), nullable=False
    )
    method_configuration_ids: Mapped[list] = mapped_column(json_type(), nullable=False, default=list)
    normal_train_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    normal_validation_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    anomaly_validation_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    normal_holdout_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT")
    )
    anomaly_holdout_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT")
    )

    search_space: Mapped[dict] = mapped_column(json_type(), nullable=False, default=dict)
    split_config: Mapped[dict] = mapped_column(json_type(), nullable=False, default=dict)
    objective_config: Mapped[dict] = mapped_column(json_type(), nullable=False, default=dict)
    best_trial_id: Mapped[int | None] = mapped_column(Integer)
    best_value: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    preprocessing_pipeline: Mapped[PreprocessingPipeline] = relationship()
    normal_train_dataset: Mapped[TrainingDataset] = relationship(foreign_keys=[normal_train_dataset_id])
    normal_validation_dataset: Mapped[TrainingDataset] = relationship(foreign_keys=[normal_validation_dataset_id])
    anomaly_validation_dataset: Mapped[TrainingDataset] = relationship(foreign_keys=[anomaly_validation_dataset_id])
    normal_holdout_dataset: Mapped[TrainingDataset | None] = relationship(foreign_keys=[normal_holdout_dataset_id])
    anomaly_holdout_dataset: Mapped[TrainingDataset | None] = relationship(foreign_keys=[anomaly_holdout_dataset_id])
    trials: Mapped[list["OptimizationTrial"]] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
        order_by="OptimizationTrial.number",
    )


class OptimizationTrial(Base):
    """One sampled candidate inside an optimization study."""

    __tablename__ = "optimization_trials"
    __table_args__ = (
        UniqueConstraint("study_id", "number", name="uq_optimization_trial_number"),
        Index("ix_optimization_trials_study_status", "study_id", "status"),
        Index("ix_optimization_trials_value", "objective_value"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("optimization_studies.id", ondelete="CASCADE"), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="waiting")
    phase: Mapped[str] = mapped_column(String(64), nullable=False, default="waiting")
    sampled_params: Mapped[dict] = mapped_column(json_type(), nullable=False, default=dict)
    method_configuration_id: Mapped[int | None] = mapped_column(
        ForeignKey("method_configurations.id", ondelete="SET NULL")
    )
    training_pipeline_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_pipelines.id", ondelete="SET NULL")
    )
    training_run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id", ondelete="SET NULL"))
    normal_testing_run_id: Mapped[int | None] = mapped_column(ForeignKey("testing_runs.id", ondelete="SET NULL"))
    anomaly_testing_run_id: Mapped[int | None] = mapped_column(ForeignKey("testing_runs.id", ondelete="SET NULL"))
    normal_holdout_testing_run_id: Mapped[int | None] = mapped_column(ForeignKey("testing_runs.id", ondelete="SET NULL"))
    anomaly_holdout_testing_run_id: Mapped[int | None] = mapped_column(ForeignKey("testing_runs.id", ondelete="SET NULL"))
    objective_value: Mapped[float | None] = mapped_column(Float)
    metrics: Mapped[dict | None] = mapped_column(json_type())
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    study: Mapped[OptimizationStudy] = relationship(back_populates="trials")
    method_configuration: Mapped[MethodConfiguration | None] = relationship()
    training_pipeline: Mapped[TrainingPipeline | None] = relationship()
    training_run: Mapped[TrainingRun | None] = relationship(foreign_keys=[training_run_id])
    normal_testing_run: Mapped[TestingRun | None] = relationship(foreign_keys=[normal_testing_run_id])
    anomaly_testing_run: Mapped[TestingRun | None] = relationship(foreign_keys=[anomaly_testing_run_id])
    normal_holdout_testing_run: Mapped[TestingRun | None] = relationship(foreign_keys=[normal_holdout_testing_run_id])
    anomaly_holdout_testing_run: Mapped[TestingRun | None] = relationship(foreign_keys=[anomaly_holdout_testing_run_id])


class EvaluationProfile(Base):
    """Reusable defaults for the three independent evaluation stages."""

    __tablename__ = "evaluation_profiles"
    __table_args__ = (UniqueConstraint("name", name="uq_evaluation_profile_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    normal_window_duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=3600.0)
    normal_window_buffer_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drift_window_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=3600.0)
    false_alarm_horizon_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=3600.0)
    anticipation_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    epsilon: Mapped[float] = mapped_column(Float, nullable=False, default=1e-12)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )


class EvaluationLabelSet(Base):
    """Versioned ground-truth intervals bound to one inference dataset."""

    __tablename__ = "evaluation_label_sets"
    __table_args__ = (
        UniqueConstraint("training_dataset_id", "name", name="uq_evaluation_label_set_dataset_name"),
        Index("ix_evaluation_label_sets_dataset", "training_dataset_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    training_dataset: Mapped[TrainingDataset] = relationship()
    events: Mapped[list["EvaluationLabelEvent"]] = relationship(
        back_populates="label_set",
        cascade="all, delete-orphan",
        order_by="EvaluationLabelEvent.start_timestamp",
    )


class EvaluationLabelEvent(Base):
    __tablename__ = "evaluation_label_events"
    __table_args__ = (
        UniqueConstraint("label_set_id", "event_id", name="uq_evaluation_label_event_key"),
        Index("ix_evaluation_label_events_set_range", "label_set_id", "start_timestamp", "end_timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label_set_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_label_sets.id", ondelete="CASCADE"), nullable=False
    )
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    category: Mapped[str | None] = mapped_column(String(128))
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    label_set: Mapped[EvaluationLabelSet] = relationship(back_populates="events")


class ModelEvaluation(Base):
    """A mutable single-model evaluation draft or immutable final snapshot."""

    __tablename__ = "model_evaluations"
    __table_args__ = (
        Index("ix_model_evaluations_status", "status"),
        Index("ix_model_evaluations_created_at", "created_at"),
        Index("ix_model_evaluations_evaluation_run", "evaluation_testing_run_id"),
        Index("ix_model_evaluations_reference_run", "reference_testing_run_id"),
        Index("ix_model_evaluations_calibration_run", "calibration_testing_run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")

    evaluation_testing_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("testing_runs.id", ondelete="RESTRICT")
    )
    reference_testing_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("testing_runs.id", ondelete="RESTRICT")
    )
    calibration_testing_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("testing_runs.id", ondelete="RESTRICT")
    )
    profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_profiles.id", ondelete="RESTRICT")
    )
    label_set_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_label_sets.id", ondelete="RESTRICT")
    )
    score_series: Mapped[str] = mapped_column(String(32), nullable=False, default="score")

    evaluation_start_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    evaluation_end_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    reference_start_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    reference_end_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    calibration_start_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    calibration_end_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))

    selected_categories: Mapped[list | None] = mapped_column(json_type())
    normal_window_overrides: Mapped[dict | None] = mapped_column(json_type())
    profile_overrides: Mapped[dict | None] = mapped_column(json_type())
    profile_snapshot: Mapped[dict | None] = mapped_column(json_type())
    label_snapshot: Mapped[dict | None] = mapped_column(json_type())
    source_snapshot: Mapped[dict | None] = mapped_column(json_type())
    config_signature: Mapped[str | None] = mapped_column(String(64))

    separation_status: Mapped[str] = mapped_column(String(24), nullable=False, default="not_calculated")
    separation_config_signature: Mapped[str | None] = mapped_column(String(64))
    separation_result: Mapped[dict | None] = mapped_column(json_type())
    separation_error: Mapped[str | None] = mapped_column(Text)
    drift_status: Mapped[str] = mapped_column(String(24), nullable=False, default="not_calculated")
    drift_config_signature: Mapped[str | None] = mapped_column(String(64))
    drift_result: Mapped[dict | None] = mapped_column(json_type())
    drift_error: Mapped[str | None] = mapped_column(Text)
    detection_status: Mapped[str] = mapped_column(String(24), nullable=False, default="not_calculated")
    detection_config_signature: Mapped[str | None] = mapped_column(String(64))
    detection_result: Mapped[dict | None] = mapped_column(json_type())
    detection_error: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list | None] = mapped_column(json_type())

    sep_median: Mapped[float | None] = mapped_column(Float)
    sep_min: Mapped[float | None] = mapped_column(Float)
    drift_mean: Mapped[float | None] = mapped_column(Float)
    drift_max: Mapped[float | None] = mapped_column(Float)
    event_recall: Mapped[float | None] = mapped_column(Float)
    median_delay_seconds: Mapped[float | None] = mapped_column(Float)
    frame_fpr: Mapped[float | None] = mapped_column(Float)
    false_alarm_rate_t0: Mapped[float | None] = mapped_column(Float)
    active_quantile: Mapped[float] = mapped_column(Float, nullable=False, default=0.999)

    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    evaluation_testing_run: Mapped[TestingRun | None] = relationship(
        foreign_keys=[evaluation_testing_run_id]
    )
    reference_testing_run: Mapped[TestingRun | None] = relationship(
        foreign_keys=[reference_testing_run_id]
    )
    calibration_testing_run: Mapped[TestingRun | None] = relationship(
        foreign_keys=[calibration_testing_run_id]
    )
    profile: Mapped[EvaluationProfile | None] = relationship()
    label_set: Mapped[EvaluationLabelSet | None] = relationship()


class EvaluationModelWorkspace(Base):
    """Model/artifact-scoped aggregate shown by the new evaluation UI."""

    __tablename__ = "evaluation_model_workspaces"
    __table_args__ = (
        UniqueConstraint("training_run_id", "artifact_signature", name="uq_evaluation_workspace_artifact"),
        Index("ix_evaluation_workspaces_training_run", "training_run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="RESTRICT"), nullable=False)
    artifact_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    sep_median: Mapped[float | None] = mapped_column(Float)
    sep_min: Mapped[float | None] = mapped_column(Float)
    drift_mean: Mapped[float | None] = mapped_column(Float)
    drift_max: Mapped[float | None] = mapped_column(Float)
    active_drift_calculation_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), onupdate=func.now())


class EvaluationSeparationLayout(Base):
    __tablename__ = "evaluation_separation_layouts"
    __table_args__ = (
        UniqueConstraint("training_dataset_id", "name", name="uq_eval_sep_layout_dataset_name"),
        Index("ix_eval_sep_layouts_dataset", "training_dataset_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_dataset_id: Mapped[int] = mapped_column(ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), onupdate=func.now())
    pairs: Mapped[list["EvaluationSeparationPair"]] = relationship(back_populates="layout", cascade="all, delete-orphan", order_by="EvaluationSeparationPair.position")


class EvaluationSeparationPair(Base):
    __tablename__ = "evaluation_separation_pairs"
    __table_args__ = (UniqueConstraint("layout_id", "pair_key", name="uq_eval_sep_pair_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    layout_id: Mapped[int] = mapped_column(ForeignKey("evaluation_separation_layouts.id", ondelete="CASCADE"), nullable=False)
    pair_key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normal_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    normal_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    anomaly_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    anomaly_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    layout: Mapped[EvaluationSeparationLayout] = relationship(back_populates="pairs")


class EvaluationDriftLayout(Base):
    __tablename__ = "evaluation_drift_layouts"
    __table_args__ = (
        UniqueConstraint("training_dataset_id", "name", name="uq_eval_drift_layout_dataset_name"),
        Index("ix_eval_drift_layouts_dataset", "training_dataset_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_dataset_id: Mapped[int] = mapped_column(ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    reference_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    reference_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    analysis_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    analysis_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    bucket_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    reference_exclusion_action: Mapped[str] = mapped_column(String(24), nullable=False, default="filter_points")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), onupdate=func.now())
    exclusions: Mapped[list["EvaluationDriftExclusion"]] = relationship(back_populates="layout", cascade="all, delete-orphan", order_by="EvaluationDriftExclusion.start_timestamp")
    buckets: Mapped[list["EvaluationDriftBucket"]] = relationship(back_populates="layout", cascade="all, delete-orphan", order_by="EvaluationDriftBucket.position")


class EvaluationDriftExclusion(Base):
    __tablename__ = "evaluation_drift_exclusions"
    __table_args__ = (UniqueConstraint("layout_id", "exclusion_key", name="uq_eval_drift_exclusion_key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    layout_id: Mapped[int] = mapped_column(ForeignKey("evaluation_drift_layouts.id", ondelete="CASCADE"), nullable=False)
    exclusion_key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    layout: Mapped[EvaluationDriftLayout] = relationship(back_populates="exclusions")


class EvaluationDriftBucket(Base):
    __tablename__ = "evaluation_drift_buckets"
    __table_args__ = (UniqueConstraint("layout_id", "bucket_key", name="uq_eval_drift_bucket_key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    layout_id: Mapped[int] = mapped_column(ForeignKey("evaluation_drift_layouts.id", ondelete="CASCADE"), nullable=False)
    bucket_key: Mapped[str] = mapped_column(String(64), nullable=False)
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(String(24), nullable=False, default="include")
    layout: Mapped[EvaluationDriftLayout] = relationship(back_populates="buckets")


class EvaluationSeparationCalculation(Base):
    __tablename__ = "evaluation_separation_calculations"
    __table_args__ = (Index("ix_eval_sep_calcs_workspace", "workspace_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("evaluation_model_workspaces.id", ondelete="CASCADE"), nullable=False)
    testing_run_id: Mapped[int] = mapped_column(ForeignKey("testing_runs.id", ondelete="RESTRICT"), nullable=False)
    layout_id: Mapped[int | None] = mapped_column(ForeignKey("evaluation_separation_layouts.id", ondelete="SET NULL"))
    layout_version: Mapped[int] = mapped_column(Integer, nullable=False)
    layout_snapshot: Mapped[dict] = mapped_column(json_type(), nullable=False)
    score_series: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    source_result_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    results: Mapped[list["EvaluationSeparationResult"]] = relationship(cascade="all, delete-orphan")


class EvaluationSeparationResult(Base):
    __tablename__ = "evaluation_separation_results"
    __table_args__ = (UniqueConstraint("calculation_id", "pair_key", name="uq_eval_sep_result_pair"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    calculation_id: Mapped[int] = mapped_column(ForeignKey("evaluation_separation_calculations.id", ondelete="CASCADE"), nullable=False)
    pair_key: Mapped[str] = mapped_column(String(64), nullable=False)
    pair_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normal_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    normal_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    anomaly_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    anomaly_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    normal_median: Mapped[float] = mapped_column(Float, nullable=False)
    normal_mad: Mapped[float] = mapped_column(Float, nullable=False)
    robust_scale: Mapped[float] = mapped_column(Float, nullable=False)
    normal_point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    separation: Mapped[float] = mapped_column(Float, nullable=False)
    separation_p95: Mapped[float | None] = mapped_column(Float)
    included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class EvaluationDriftCalculation(Base):
    __tablename__ = "evaluation_drift_calculations"
    __table_args__ = (Index("ix_eval_drift_calcs_workspace", "workspace_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("evaluation_model_workspaces.id", ondelete="CASCADE"), nullable=False)
    testing_run_id: Mapped[int] = mapped_column(ForeignKey("testing_runs.id", ondelete="RESTRICT"), nullable=False)
    layout_id: Mapped[int | None] = mapped_column(ForeignKey("evaluation_drift_layouts.id", ondelete="SET NULL"))
    layout_version: Mapped[int] = mapped_column(Integer, nullable=False)
    layout_snapshot: Mapped[dict] = mapped_column(json_type(), nullable=False)
    score_series: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    source_result_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    reference_iqr: Mapped[float] = mapped_column(Float, nullable=False)
    reference_point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_mean: Mapped[float | None] = mapped_column(Float)
    drift_max: Mapped[float | None] = mapped_column(Float)
    stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    results: Mapped[list["EvaluationDriftBucketResult"]] = relationship(cascade="all, delete-orphan")


class EvaluationDriftBucketResult(Base):
    __tablename__ = "evaluation_drift_bucket_results"
    __table_args__ = (UniqueConstraint("calculation_id", "bucket_key", name="uq_eval_drift_result_bucket"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    calculation_id: Mapped[int] = mapped_column(ForeignKey("evaluation_drift_calculations.id", ondelete="CASCADE"), nullable=False)
    bucket_key: Mapped[str] = mapped_column(String(64), nullable=False)
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    original_point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    used_point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    exclusion_overlap: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    wasserstein_1: Mapped[float | None] = mapped_column(Float)
    normalized_drift: Mapped[float | None] = mapped_column(Float)
    included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class RedundancyCsvSource(Base):
    """Immutable project-local CSV source used by redundancy analyses."""

    __tablename__ = "redundancy_csv_sources"
    __table_args__ = (
        UniqueConstraint("sha256", name="uq_redundancy_csv_source_sha256"),
        Index("ix_redundancy_csv_sources_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    delimiter: Mapped[str] = mapped_column(String(8), nullable=False)
    encoding: Mapped[str] = mapped_column(String(32), nullable=False, default="utf-8")
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    headers: Mapped[list] = mapped_column(json_type(), nullable=False)
    column_profiles: Mapped[list] = mapped_column(json_type(), nullable=False)
    preview_rows: Mapped[list] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )
    analyses: Mapped[list["RedundancyAnalysis"]] = relationship(back_populates="source")


class RedundancyAnalysis(Base):
    """Persisted draft or immutable snapshot of one redundancy calculation."""

    __tablename__ = "redundancy_analyses"
    __table_args__ = (
        Index("ix_redundancy_analyses_source", "source_id"),
        Index("ix_redundancy_analyses_status", "status", "job_status"),
        Index("ix_redundancy_analyses_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("redundancy_csv_sources.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="draft")
    job_status: Mapped[str] = mapped_column(String(24), nullable=False, default="not_calculated")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    time_column: Mapped[str] = mapped_column(String(255), nullable=False)
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    selected_columns: Mapped[list] = mapped_column(json_type(), nullable=False)
    config: Mapped[dict] = mapped_column(json_type(), nullable=False)
    active_cutoff: Mapped[float] = mapped_column(Float, nullable=False, default=0.9)
    result: Mapped[dict | None] = mapped_column(json_type())
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )
    source: Mapped[RedundancyCsvSource] = relationship(back_populates="analyses")


ModelConfiguration = MethodConfiguration
ModelConfigurationParameter = MethodConfigurationParameter
