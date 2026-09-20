from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class DatasetUpdate(BaseModel):
    name: Name
    selected_columns: list[str] = Field(min_length=2)


class DatasetImport(DatasetUpdate):
    timestamp_column: str
    timestamp_format: str = "ISO8601"
    label_column: str | None = None
    auto_split: bool = False
    split_name: Name | None = None


class IntervalInput(BaseModel):
    id: Name
    start: str
    end: str
    subset: Literal["train", "test", "validation"]
    tags: list[Name] = Field(default_factory=list)


class SplitInput(BaseModel):
    name: Name
    dataset_id: int
    tags: list[Name] = Field(default_factory=list)
    intervals: list[IntervalInput] = Field(min_length=1)

    @field_validator("tags")
    @classmethod
    def unique_tags(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Tags müssen eindeutig sein.")
        return value


class ModelInput(BaseModel):
    model_config = {"extra": "forbid"}
    name: Name
    kind: Literal["usad", "tcn_ae", "lstm_vae"]
    config: dict = Field(default_factory=dict)


class PipelinePreview(BaseModel):
    model_config = {"extra": "forbid"}
    split_id: int = Field(gt=0)
    model_id: int | None = Field(default=None, gt=0)
    window_length: int | None = Field(default=None, ge=1, le=100000)
    training: dict = Field(default_factory=dict)


class PipelineInput(PipelinePreview):
    name: Name
    model_id: int = Field(gt=0)
    window_length: int = Field(ge=1, le=100000)
