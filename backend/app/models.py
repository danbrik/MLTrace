from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
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


ModelConfiguration = MethodConfiguration
ModelConfigurationParameter = MethodConfigurationParameter
