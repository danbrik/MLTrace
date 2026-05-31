from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PreprocessingStep(Protocol):
    type: str
    label: str
    category: str
    input_kind: str
    output_kind: str
    config_schema: dict
    default_config: dict

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        ...


@dataclass(frozen=True)
class StepDefinition:
    type: str
    label: str
    category: str
    input_kind: str
    output_kind: str
    config_schema: dict
    default_config: dict


class PreprocessingRegistry:
    def __init__(self) -> None:
        self._steps: dict[str, PreprocessingStep] = {}

    def register(self, step: PreprocessingStep) -> None:
        if step.type in self._steps:
            raise ValueError(f"Preprocessing step is already registered: {step.type}")
        self._steps[step.type] = step

    def get(self, step_type: str) -> PreprocessingStep:
        try:
            return self._steps[step_type]
        except KeyError as exc:
            raise ValueError(f"Unknown preprocessing step: {step_type}") from exc

    def list_definitions(self) -> list[StepDefinition]:
        return [
            StepDefinition(
                type=step.type,
                label=step.label,
                category=step.category,
                input_kind=step.input_kind,
                output_kind=step.output_kind,
                config_schema=step.config_schema,
                default_config=step.default_config,
            )
            for step in sorted(self._steps.values(), key=lambda item: (item.category, item.label))
        ]


registry = PreprocessingRegistry()
