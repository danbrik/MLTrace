from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePreprocessingStep(ABC):
    """Base class for modular preprocessing building blocks."""

    type: str
    label: str
    category: str
    input_kind: str = "image"
    output_kind: str = "image"
    default_config: dict = {}
    config_schema: dict = {"type": "object", "properties": {}}

    def merged_config(self, config: dict | None) -> dict:
        merged = dict(self.default_config)
        merged.update(config or {})
        return merged

    @abstractmethod
    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        """Transform an image and return the next image in the pipeline."""

