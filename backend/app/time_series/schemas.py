from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class DatasetUpdate(BaseModel):
    name: Name
    selected_columns: list[str] = Field(min_length=2)


class DatasetImport(DatasetUpdate):
    timestamp_column: str
    timestamp_format: str = "ISO8601"


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
