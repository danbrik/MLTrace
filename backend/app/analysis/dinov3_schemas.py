"""Public contracts for exploratory image representations."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas import _dataset_local_naive

Label = Literal["normal", "anomaly", "buffer"]


class RepresentationInterval(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    label: Label
    start: datetime
    end: datetime
    sampling_rate: int = Field(default=30, ge=1)
    random: bool = False
    event_id: str | None = Field(default=None, pattern=r"^A[1-9][0-9]*$")

    @model_validator(mode="after")
    def validate_interval(self):
        self.start = _dataset_local_naive(self.start)
        self.end = _dataset_local_naive(self.end)
        if self.end <= self.start:
            raise ValueError("Bereichsende muss nach dem Beginn liegen.")
        if self.label != "anomaly":
            self.event_id = None
        return self


class RepresentationConfig(BaseModel):
    training_dataset_id: int = Field(ge=1)
    preprocessing_pipeline_id: int = Field(ge=1)
    intervals: list[RepresentationInterval] = Field(min_length=1)
    cluster_count: int = Field(default=3, ge=2)
    pca_variance: float = Field(default=0.95, gt=0, lt=1)
    seed: int = Field(default=42, ge=0, le=4294967295)

    @model_validator(mode="after")
    def validate_intervals(self):
        if len({item.id for item in self.intervals}) != len(self.intervals):
            raise ValueError("Bereichs-IDs müssen eindeutig sein.")
        ordered = sorted(self.intervals, key=lambda item: (item.start, item.end, item.id))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start < previous.end:
                raise ValueError(f"Bereiche '{previous.name}' und '{current.name}' überlappen.")
        used = [item.event_id for item in ordered if item.event_id]
        if len(set(used)) != len(used):
            raise ValueError("Jeder Anomaliebereich benötigt eine eigene Ereignis-ID.")
        next_id = max((int(value[1:]) for value in used), default=0) + 1
        for item in ordered:
            if item.label == "anomaly" and item.event_id is None:
                item.event_id = f"A{next_id}"
                next_id += 1
        return self


class IntervalCount(BaseModel):
    id: str
    name: str
    label: Label
    event_id: str | None
    available: int
    selected: int
    remainder: int


class SelectionPreview(BaseModel):
    intervals: list[IntervalCount]
    label_counts: dict[str, int]
    total: int
    errors: list[str]
    image: str | None = None
    model_image: str | None = None
    timestamp: str | None = None


class RepresentationMetrics(BaseModel):
    sample_count: int
    feature_dimension: int
    pca_2d_variance: list[float]
    pca_components: int
    pca_retained_variance: float
    cluster_count: int
    ari: float
    nmi: float
    label_counts: dict[str, int]
    contingency: list[dict[str, int]]


class RepresentationRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    training_dataset_id: int
    training_dataset_name: str
    status: str
    current_step: str
    processed_images: int
    total_images: int | None
    config: RepresentationConfig
    pipeline_snapshot: dict
    dataset_snapshot: dict
    model_snapshot: dict | None
    result: RepresentationMetrics | None
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


class RepresentationPoint(BaseModel):
    file_path: str
    timestamp: str
    interval_id: str
    interval_name: str
    label: Label
    event_id: str | None
    pca_x: float
    pca_y: float
    umap_x: float
    umap_y: float
    cluster: int


class RepresentationResults(BaseModel):
    metrics: RepresentationMetrics
    points: list[RepresentationPoint]
