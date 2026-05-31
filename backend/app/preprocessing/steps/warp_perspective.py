from __future__ import annotations

import cv2
import numpy as np

from app.preprocessing.base import BasePreprocessingStep
from app.preprocessing.utils import default_quad, interpolation_flag


class WarpPerspectiveStep(BasePreprocessingStep):
    type = "warp_perspective"
    label = "Warp perspective"
    category = "Geometry"
    input_kind = "image ndarray"
    output_kind = "warped image ndarray"
    default_config = {
        "source_points": [],
        "output_width": 128,
        "output_height": 128,
        "interpolation": "linear",
    }
    config_schema = {
        "type": "object",
        "ui_control": "point_picker",
        "properties": {
            "source_points": {
                "type": "array",
                "label": "Source points",
                "ui_control": "point_picker",
                "minItems": 4,
                "maxItems": 4,
                "default": [],
            },
            "output_width": {"type": "integer", "label": "Output width", "minimum": 1, "default": 128},
            "output_height": {"type": "integer", "label": "Output height", "minimum": 1, "default": 128},
            "interpolation": {
                "type": "string",
                "label": "Interpolation",
                "enum": ["nearest", "linear", "area", "cubic"],
                "default": "linear",
            },
        },
    }

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("warp_perspective requires an input image.")
        cfg = self.merged_config(config)
        output_width = int(cfg["output_width"])
        output_height = int(cfg["output_height"])
        height, width = image.shape[:2]
        points = cfg.get("source_points") or default_quad(width, height)
        if len(points) != 4:
            points = default_quad(width, height)

        source = np.float32([[float(point["x"]), float(point["y"])] for point in points])
        target = np.float32(
            [
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1],
            ]
        )
        matrix = cv2.getPerspectiveTransform(source, target)
        return cv2.warpPerspective(
            image,
            matrix,
            (output_width, output_height),
            flags=interpolation_flag(cfg["interpolation"]),
        )
