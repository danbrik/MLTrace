from __future__ import annotations

import cv2
import numpy as np

from app.preprocessing.base import BasePreprocessingStep
from app.preprocessing.utils import interpolation_flag


class ResizeStep(BasePreprocessingStep):
    type = "resize"
    label = "Resize"
    category = "Geometry"
    input_kind = "image ndarray"
    output_kind = "resized image ndarray"
    default_config = {"width": 128, "height": 128, "interpolation": "area"}
    config_schema = {
        "type": "object",
        "properties": {
            "width": {"type": "integer", "label": "Width", "minimum": 1, "default": 128},
            "height": {"type": "integer", "label": "Height", "minimum": 1, "default": 128},
            "interpolation": {
                "type": "string",
                "label": "Interpolation",
                "enum": ["nearest", "linear", "area", "cubic"],
                "default": "area",
            },
        },
    }

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("resize requires an input image.")
        cfg = self.merged_config(config)
        return cv2.resize(
            image,
            (int(cfg["width"]), int(cfg["height"])),
            interpolation=interpolation_flag(cfg["interpolation"]),
        )
