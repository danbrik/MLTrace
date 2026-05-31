from __future__ import annotations

import numpy as np

from app.preprocessing.base import BasePreprocessingStep


class CropStep(BasePreprocessingStep):
    type = "crop"
    label = "Crop"
    category = "Geometry"
    input_kind = "image ndarray"
    output_kind = "cropped image ndarray"
    default_config = {"x": 0, "y": 0, "width": 128, "height": 128}
    config_schema = {
        "type": "object",
        "ui_control": "crop_box",
        "properties": {
            "x": {"type": "integer", "label": "X", "minimum": 0, "default": 0},
            "y": {"type": "integer", "label": "Y", "minimum": 0, "default": 0},
            "width": {"type": "integer", "label": "Width", "minimum": 1, "default": 128},
            "height": {"type": "integer", "label": "Height", "minimum": 1, "default": 128},
        },
    }

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("crop requires an input image.")
        cfg = self.merged_config(config)
        image_height, image_width = image.shape[:2]
        x = max(0, min(int(cfg["x"]), image_width - 1))
        y = max(0, min(int(cfg["y"]), image_height - 1))
        width = max(1, int(cfg["width"]))
        height = max(1, int(cfg["height"]))
        return image[y : min(image_height, y + height), x : min(image_width, x + width)]
