from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.reference_image.schemas import Interval, ReferenceInterval, SelectionPreview


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


class MeanVarianceRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    training_dataset_id: int
    training_dataset_name: str
    status: str
    current_step: str
    processed_images: int
    total_images: int | None
    config: MeanVarianceConfig
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
