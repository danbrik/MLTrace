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
