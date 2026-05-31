from __future__ import annotations

import numpy as np

from app.preprocessing.base import BasePreprocessingStep
from app.preprocessing.utils import normalize_to_uint8


class NormalizeForPreviewStep(BasePreprocessingStep):
    type = "normalize_for_preview"
    label = "Normalize for preview"
    category = "Intensity"
    input_kind = "image ndarray"
    output_kind = "uint8 preview ndarray"
    default_config = {"mode": "minmax"}
    config_schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "label": "Mode", "enum": ["minmax", "clip_0_255"], "default": "minmax"}
        },
    }

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("normalize_for_preview requires an input image.")
        cfg = self.merged_config(config)
        if cfg["mode"] == "clip_0_255":
            return np.clip(image.astype(np.float32), 0, 255).astype(np.uint8)
        return normalize_to_uint8(image)
