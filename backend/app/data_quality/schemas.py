from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Parameters(BaseModel):
    source_id: int
    time_column: str
    selected_columns: list[str] = Field(min_length=1)
    data_types: dict[str, Literal['numeric', 'text']]
    start_timestamp: datetime
    end_timestamp: datetime
    interval_seconds: int = Field(default=60, ge=1)

    @model_validator(mode='after')
    def validate_parameters(self):
        if self.start_timestamp.tzinfo or self.end_timestamp.tzinfo:
            raise ValueError('Use CSV-local timestamps without a timezone.')
        if self.end_timestamp < self.start_timestamp:
            raise ValueError('End must be at or after start.')
        self.selected_columns = sorted(set(self.selected_columns))
        if self.time_column in self.selected_columns:
            raise ValueError('The time column cannot also be a sensor.')
        if any(name not in self.data_types for name in self.selected_columns):
            raise ValueError('Each selected sensor needs a data type.')
        self.data_types = {name: self.data_types[name] for name in self.selected_columns}
        return self


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source_id: int
    name: str
    parameters: dict
    job_status: str
    progress: float
    stage: str
    elapsed_seconds: float
    eta_seconds: float | None
    started_at: datetime | None
    error_message: str | None
    result: dict | None
    created_at: datetime
    updated_at: datetime
