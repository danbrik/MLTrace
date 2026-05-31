from __future__ import annotations

import cv2
import numpy as np

from app.preprocessing.base import BasePreprocessingStep, ImageSpec


class GrayscaleStep(BasePreprocessingStep):
    type = "grayscale"
    label = "Grayscale"
    category = "Color"
    input_kind = "image ndarray"
    output_kind = "grayscale ndarray"
    default_config = {}
    config_schema = {"type": "object", "properties": {}}

    def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
        if spec_in is None:
            raise ValueError("grayscale requires an input image.")
        return ImageSpec(channels=1, width=spec_in.width, height=spec_in.height, dtype=spec_in.dtype)

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("grayscale requires an input image.")
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
