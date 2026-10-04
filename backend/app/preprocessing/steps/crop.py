from __future__ import annotations

import cv2
import numpy as np
from app.image_geometry import RotatedRectangle, aligned_crop, validate_selection

from app.preprocessing.base import BasePreprocessingStep, ImageSpec
from app.preprocessing.utils import interpolation_flag


class CropStep(BasePreprocessingStep):
    type = "crop"
    label = "Crop"
    category = "Geometry"
    input_kind = "image ndarray"
    output_kind = "cropped image ndarray"
    default_config = {"x": 0, "y": 0, "width": 128, "height": 128, "output_size": "cropped", "interpolation": "area"}
    config_schema = {
        "type": "object",
        "ui_control": "crop_box",
        "properties": {
            "roi": {"type": "object", "label": "Drehbares Rechteck",
                    "properties": RotatedRectangle.model_json_schema()["properties"],
                    "required": RotatedRectangle.model_json_schema()["required"], "additionalProperties": False},
            "x": {"type": "integer", "label": "X", "minimum": 0, "default": 0},
            "y": {"type": "integer", "label": "Y", "minimum": 0, "default": 0},
            "width": {"type": "integer", "label": "Width", "minimum": 1, "default": 128, "default_from": "input_width"},
            "height": {"type": "integer", "label": "Height", "minimum": 1, "default": 128, "default_from": "input_height"},
            "output_size": {
                "type": "string",
                "label": "Ausgabegröße",
                "enum": ["cropped", "input", "source"],
                "default": "cropped",
            },
            "interpolation": {
                "type": "string",
                "label": "Interpolation der Größenanpassung",
                "enum": ["nearest", "linear", "area", "cubic"],
                "default": "area",
            },
        },
    }

    def merged_config(self, config: dict | None) -> dict:
        cfg = super().merged_config(config)
        if "roi" in cfg:
            for key in ("x", "y", "width", "height"):
                cfg.pop(key, None)
        return cfg

    def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
        if spec_in is None:
            raise ValueError("crop requires an input image.")
        cfg = self.merged_config(config)
        roi = RotatedRectangle.model_validate(cfg["roi"]) if "roi" in cfg else None
        if roi is not None and spec_in.width is not None and spec_in.height is not None:
            validate_selection(roi, spec_in.width, spec_in.height)
        mode = cfg["output_size"]
        if mode == "input":
            return ImageSpec(channels=spec_in.channels, width=spec_in.width, height=spec_in.height, dtype=spec_in.dtype)
        if mode == "source":
            # The original size is only known once a concrete image is loaded.
            return ImageSpec(channels=spec_in.channels, width=None, height=None, dtype=spec_in.dtype)
        return ImageSpec(channels=spec_in.channels, width=roi.width if roi else int(cfg["width"]), height=roi.height if roi else int(cfg["height"]), dtype=spec_in.dtype)

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("crop requires an input image.")
        cfg = self.merged_config(config)
        image_height, image_width = image.shape[:2]
        if "roi" in cfg:
            roi = RotatedRectangle.model_validate(cfg["roi"])
            validate_selection(roi, image_width, image_height)
            cropped = aligned_crop(image, roi)
        else:
            # Preserve historical clamping for frozen pipelines without rotated geometry.
            x = max(0, min(int(cfg["x"]), image_width - 1))
            y = max(0, min(int(cfg["y"]), image_height - 1))
            width = max(1, int(cfg["width"]))
            height = max(1, int(cfg["height"]))
            cropped = image[y : min(image_height, y + height), x : min(image_width, x + width)]

        mode = cfg["output_size"]
        if mode == "cropped":
            return cropped
        if mode == "input":
            target = (image_width, image_height)
        else:  # source: interpolate back to the pipeline's original image size
            source_shape = context.get("source_shape")
            target = (
                (int(source_shape[1]), int(source_shape[0])) if source_shape is not None else (image_width, image_height)
            )
        return cv2.resize(cropped, target, interpolation=interpolation_flag(cfg["interpolation"]))
