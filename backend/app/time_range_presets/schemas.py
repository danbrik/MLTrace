from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas import _dataset_local_naive


class TimeRangePresetWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=255)
    start: datetime
    end: datetime

    @field_validator("name", mode="before")
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("start", "end")
    @classmethod
    def local_time(cls, value):
        return _dataset_local_naive(value)

    @model_validator(mode="after")
    def ordered(self):
        if self.start > self.end:
            raise ValueError("Beginn darf nicht nach dem Ende liegen.")
        return self


class TimeRangePresetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    start: datetime
    end: datetime
    created_at: datetime
    updated_at: datetime
