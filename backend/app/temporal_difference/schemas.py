from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.schemas import _dataset_local_naive


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


class TemporalDifferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selection_version: Literal[1, 2] = 1
    block_seconds: int = Field(default=300, strict=True, gt=0, le=9007199254740991)
    seed: int = Field(default=42, strict=True, ge=0, le=9007199254740991)
    training_dataset_id: int = Field(ge=1)
    preprocessing_pipeline_id: int = Field(ge=1)
    reference: Period
    comparison: Period
    deltas_seconds: list[Annotated[int, Field(strict=True, gt=0, le=9007199254740991)]] = Field(
        default_factory=lambda: [1, 2, 5, 15, 30, 60], min_length=1)

    @model_validator(mode="after")
    def unique_deltas(self):
        if len(set(self.deltas_seconds)) != len(self.deltas_seconds):
            raise ValueError("Zeitabstände dürfen nicht doppelt vorkommen.")
        self.deltas_seconds.sort()
        return self


class AxisRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum >= self.maximum:
            raise ValueError("Die Achsenuntergrenze muss kleiner als die Obergrenze sein.")
        return self


class PlotSettings(BaseModel):
    unit_version: Literal[1, 2] = 1
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="Zeitabstands-Analyse", max_length=250)
    x_title: str = Field(default="Zeitabstand Δt (s)", max_length=250)
    y_title: str = Field(default="Mittlere absolute Pixeländerung (Pipeline-Einheiten)", max_length=250)
    x_range: AxisRange | None = None
    y_range: AxisRange | None = None
    reference_color: str = Field(default="#1971c2", pattern=r"^#[0-9a-fA-F]{6}$")
    comparison_color: str = Field(default="#e8590c", pattern=r"^#[0-9a-fA-F]{6}$")


class TemporalDifferenceRunRead(BaseModel):
    display_unit: Literal["percent"] = "percent"
    normalization_divisor: Literal[65535] = 65535
    model_config = ConfigDict(from_attributes=True)
    id: int
    training_dataset_id: int
    training_dataset_name: str
    config: TemporalDifferenceConfig
    pipeline_snapshot: dict
    dataset_snapshot: dict
    plot_settings: PlotSettings
    result: dict | None
    status: str
    current_step: str
    processed_images: int
    total_images: int | None
    cancel_requested: bool
    error_message: str | None
    queue_rank: int | None
    enqueued_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    heartbeat_at: datetime | None
    duration_seconds: float | None
    device: str | None
    gpu_index: int | None
    created_at: datetime


class MatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["reference", "comparison"]
    deltas_seconds: list[Annotated[int, Field(strict=True, gt=0)]] = Field(min_length=1, max_length=3)
    start_times: list[str] = Field(min_length=1, max_length=3)
    top_percent: float | None = Field(default=None, ge=.01, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.deltas_seconds)) != len(self.deltas_seconds) or len(set(self.start_times)) != len(self.start_times):
            raise ValueError("Zeitabstände und Startpunkte müssen unterschiedlich sein.")
        self.deltas_seconds.sort()
        return self
