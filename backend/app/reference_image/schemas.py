"""Project-scoped reference image analysis contracts."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas import _dataset_local_naive


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: datetime
    end: datetime
    sampling_rate: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_range(self):
        self.start = _dataset_local_naive(self.start)
        self.end = _dataset_local_naive(self.end)
        if self.end < self.start:
            raise ValueError("Das Ende darf nicht vor dem Beginn liegen.")
        return self


class ReferenceInterval(Interval):
    mode: Literal["regular", "random"] = "regular"
    count: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0, le=4294967295)


class ReferenceImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_dataset_id: int = Field(ge=1)
    preprocessing_pipeline_id: int = Field(ge=1)
    reference: ReferenceInterval
    anomaly: Interval
    processing_mode: Literal["shift_clip", "signed"] = "shift_clip"
    shift: float = Field(default=10000, allow_inf_nan=False)
    clip_min: int = Field(default=0, ge=0, le=65535)
    clip_max: int = Field(default=12000, ge=0, le=65535)
    scale_mode: Literal["auto", "manual"] = "auto"
    scale_limit: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fps: int = Field(default=10, ge=1, le=120)

    @model_validator(mode="after")
    def validate_scale(self):
        if self.clip_min >= self.clip_max:
            raise ValueError("Clip-Minimum muss kleiner als Clip-Maximum sein (0 bis 65535).")
        if self.processing_mode == "signed" and self.scale_mode == "manual" and self.scale_limit is None:
            raise ValueError("Für manuellen Kontrast ist ein positiver Grenzwert erforderlich.")
        return self


class SelectionCount(BaseModel):
    available: int
    selected: int
    remainder: int


class SelectionPreview(BaseModel):
    reference: SelectionCount
    anomaly: SelectionCount
    errors: list[str]


def stored_config(value):
    """Old frozen runs retain their original signed rendering."""
    if isinstance(value, dict) and "processing_mode" not in value:
        value = {**value, "processing_mode": "signed"}
    return value


class ReferenceImageRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @field_validator("config", mode="before")
    @classmethod
    def read_stored_config(cls, value):
        return stored_config(value)

    id: int
    training_dataset_id: int
    training_dataset_name: str
    status: str
    current_step: str
    processed_images: int
    total_images: int | None
    config: ReferenceImageConfig
    pipeline_snapshot: dict
    dataset_snapshot: dict
    result: dict | None
    error_message: str | None
    cancel_requested: bool
    queue_rank: int | None
    enqueued_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float | None
    heartbeat_at: datetime | None
    device: str | None
    gpu_index: int | None
    created_at: datetime


class FrameRead(BaseModel):
    index: int
    timestamp: datetime
    distance: float


class FrameLookup(BaseModel):
    requested_timestamp: datetime
    exact: bool
    frame: FrameRead
