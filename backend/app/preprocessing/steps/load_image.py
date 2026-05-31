from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.preprocessing.base import BasePreprocessingStep, ImageSpec


class LoadImageStep(BasePreprocessingStep):
    type = "load_image"
    label = "Load image"
    category = "Input"
    input_kind = "TIFF file path"
    output_kind = "image ndarray"
    default_config = {"mode": "rgb"}
    config_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "label": "Mode",
                "enum": ["rgb", "grayscale"],
                "default": "rgb",
            }
        },
    }

    def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
        mode = self.merged_config(config)["mode"]
        return ImageSpec(channels=1 if mode == "grayscale" else 3, dtype="uint8")

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        path = Path(context["source_image_path"])
        mode = self.merged_config(config)["mode"]
        with Image.open(path) as loaded:
            if mode == "grayscale":
                return np.asarray(loaded.convert("L"))
            return np.asarray(loaded.convert("RGB"))
