from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageSpec:
    """Symbolic description of the image flowing between steps.

    Used to validate a pipeline's type chain without running it. ``channels`` is known
    symbolically (e.g. rgb -> 3, grayscale -> 1); ``width``/``height`` may be ``None`` when
    they only become known once a concrete image is loaded.
    """

    channels: int | None = None
    width: int | None = None
    height: int | None = None
    dtype: str | None = None


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

    def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
        """Validate the incoming image spec and return the produced spec.

        Raise ``ValueError`` if this step cannot consume ``spec_in``. The default contract
        requires an image input and passes channels/size through unchanged. Steps override
        this to declare requirements (e.g. needs color) or shape changes (resize, crop, warp).
        """
        if spec_in is None:
            raise ValueError(f"{self.type} requires an input image.")
        return spec_in

    @abstractmethod
    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        """Transform an image and return the next image in the pipeline."""
