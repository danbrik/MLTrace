from __future__ import annotations

import cv2
import numpy as np

from app.preprocessing.base import BasePreprocessingStep


class GaussianBlurStep(BasePreprocessingStep):
    type = "gaussian_blur"
    label = "Gaussian blur"
    category = "Filters"
    input_kind = "image ndarray"
    output_kind = "blurred image ndarray"
    default_config = {"kernel_size": 5, "sigma": 0.0}
    config_schema = {
        "type": "object",
        "properties": {
            "kernel_size": {
                "type": "integer",
                "label": "Kernel size (odd)",
                "minimum": 1,
                "default": 5,
            },
            "sigma": {
                "type": "number",
                "label": "Sigma (0 = auto)",
                "minimum": 0,
                "default": 0.0,
            },
        },
    }

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("gaussian_blur requires an input image.")
        cfg = self.merged_config(config)
        kernel = max(1, int(cfg["kernel_size"]))
        if kernel % 2 == 0:  # OpenCV requires an odd kernel size.
            kernel += 1
        sigma = float(cfg["sigma"])  # 0 lets OpenCV derive sigma from the kernel size.
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma)
