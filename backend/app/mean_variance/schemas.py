from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.reference_image.schemas import Interval, ReferenceInterval, SelectionPreview
from app.schemas import _dataset_local_naive


class HeatmapScale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["auto", "manual"] = "auto"
    limit: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_limit(self):
        if self.mode == "manual" and self.limit is None:
            raise ValueError("Für eine manuelle Farbskala ist ein positiver Grenzwert erforderlich.")
        return self


class MeanVarianceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_dataset_id: int = Field(ge=1)
    preprocessing_pipeline_id: int = Field(ge=1)
    reference: ReferenceInterval
    anomaly: Interval
    mean_scale: HeatmapScale = Field(default_factory=HeatmapScale)
    variance_scale: HeatmapScale = Field(default_factory=HeatmapScale)


class Period(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def ordered(self):
        self.start = _dataset_local_naive(self.start)
        self.end = _dataset_local_naive(self.end)
        if self.end < self.start:
            raise ValueError("Das Ende darf nicht vor dem Beginn liegen.")
        return self


class VariancePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    normal: Period
    anomaly: Period


class VarianceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[2] = 2
    training_dataset_id: int = Field(ge=1)
    preprocessing_pipeline_id: int = Field(ge=1)
    sampling_rate: int = Field(default=1, ge=1, strict=True)
    pairs: list[VariancePair] = Field(min_length=1, max_length=6)
    variance_scale: HeatmapScale = Field(default_factory=HeatmapScale)
    difference_scale: HeatmapScale = Field(default_factory=HeatmapScale)

    def selection_config(self, pair):
        return MeanVarianceConfig(
            training_dataset_id=self.training_dataset_id,
            preprocessing_pipeline_id=self.preprocessing_pipeline_id,
            reference=ReferenceInterval(**pair.normal.model_dump(), sampling_rate=self.sampling_rate),
            anomaly=Interval(**pair.anomaly.model_dump(), sampling_rate=self.sampling_rate),
        )


class VariancePreview(BaseModel):
    version: Literal[2] = 2
    pairs: list[SelectionPreview]
    errors: list[str]


StoredConfig = VarianceConfig | MeanVarianceConfig


def parse_config(value):
    return (VarianceConfig if value.get("version") == 2 else MeanVarianceConfig).model_validate(value)


class MeanVarianceRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    training_dataset_id: int
    training_dataset_name: str
    status: str
    current_step: str
    processed_images: int
    total_images: int | None
    config: StoredConfig
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
