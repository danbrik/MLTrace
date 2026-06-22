from __future__ import annotations

import cv2
import numpy as np

from app.preprocessing.base import BasePreprocessingStep, ImageSpec
from app.preprocessing.utils import default_quad, interpolation_flag, order_quad_points, rectified_quad_size


class WarpPerspectiveStep(BasePreprocessingStep):
    type = "warp_perspective"
    label = "Warp perspective"
    category = "Geometry"
    input_kind = "image ndarray"
    output_kind = "warped image ndarray"
    default_config = {
        "source_points": [],
        "output_shape_mode": "preserve_rectangle",
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
            "output_shape_mode": {
                "type": "string",
                "label": "Output shape",
                "enum": ["preserve_rectangle", "manual"],
                "default": "preserve_rectangle",
            },
            "output_width": {
                "type": "integer",
                "label": "Output width",
                "minimum": 1,
                "default": 128,
                "default_from": "input_width",
                "visible_when": {"output_shape_mode": "manual"},
            },
            "output_height": {
                "type": "integer",
                "label": "Output height",
                "minimum": 1,
                "default": 128,
                "default_from": "input_height",
                "visible_when": {"output_shape_mode": "manual"},
            },
            "interpolation": {
                "type": "string",
                "label": "Interpolation",
                "enum": ["nearest", "linear", "area", "cubic"],
                "default": "linear",
            },
        },
    }

    def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
        if spec_in is None:
            raise ValueError("warp_perspective requires an input image.")
        cfg = self.merged_config(config)
        output_width: int | None
        output_height: int | None
        if cfg["output_shape_mode"] == "preserve_rectangle":
            points = cfg.get("source_points") or []
            if len(points) == 4:
                output_width, output_height = rectified_quad_size(points)
            elif spec_in.width is not None and spec_in.height is not None:
                output_width, output_height = rectified_quad_size(default_quad(spec_in.width, spec_in.height))
            else:
                output_width, output_height = None, None
        else:
            output_width = int(cfg["output_width"])
            output_height = int(cfg["output_height"])
        return ImageSpec(
            channels=spec_in.channels,
            width=output_width,
            height=output_height,
            dtype=spec_in.dtype,
        )

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("warp_perspective requires an input image.")
        cfg = self.merged_config(config)
        height, width = image.shape[:2]
        points = cfg.get("source_points") or default_quad(width, height)
        if len(points) != 4:
            points = default_quad(width, height)
        source = order_quad_points(points)
        if cfg["output_shape_mode"] == "preserve_rectangle":
            output_width, output_height = rectified_quad_size(source)
        else:
            output_width = int(cfg["output_width"])
            output_height = int(cfg["output_height"])
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
