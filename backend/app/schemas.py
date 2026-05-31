from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    root_path: str = Field(min_length=1)


class TimestampFormatConfirm(BaseModel):
    timestamp_regex: str = Field(min_length=1)
    timestamp_format: str = Field(min_length=1)


class DatasetFolderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    relative_path: str
    image_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    extension_summary: dict | None
    resolution_summary: dict | None
    cadence_summary: dict | None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    root_path: str
    status: str
    timestamp_regex: str | None
    timestamp_format: str | None
    timestamp_example: str | None
    scan_error: str | None
    scan_summary: dict | None
    created_at: datetime
    updated_at: datetime
    folders: list[DatasetFolderRead] = []


class TrainingDatasetRuleInput(BaseModel):
    folder_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int = Field(default=1, ge=1)

    @field_validator("end_timestamp")
    @classmethod
    def validate_range(cls, value: datetime, info):
        start = info.data.get("start_timestamp")
        if start and value < start:
            raise ValueError("end_timestamp must be after start_timestamp")
        return value


class TrainingDatasetPreviewRequest(BaseModel):
    rules: list[TrainingDatasetRuleInput]


class TrainingDatasetRulePreview(BaseModel):
    folder_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    matching_images: int
    selected_images: int


class TrainingDatasetPreviewResponse(BaseModel):
    total_matching_images: int
    total_selected_images: int
    rules: list[TrainingDatasetRulePreview]


class TrainingDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    notes: str | None = None
    rules: list[TrainingDatasetRuleInput] = Field(min_length=1)


class TrainingDatasetRuleRead(BaseModel):
    id: int
    folder_id: int
    dataset_id: int
    dataset_name: str
    dataset_root_path: str
    folder_relative_path: str
    folder_first_timestamp: datetime | None
    folder_last_timestamp: datetime | None
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    matching_images: int
    selected_images: int


class TrainingDatasetRead(BaseModel):
    id: int
    name: str
    notes: str | None
    created_at: datetime
    dataset_names: list[str]
    total_matching_images: int
    total_selected_images: int
    rules: list[TrainingDatasetRuleRead] = []
