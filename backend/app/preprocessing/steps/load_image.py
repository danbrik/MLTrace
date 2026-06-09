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
    # lock_size / lock_width / lock_height are managed by the UI (not rendered as raw fields);
    # when lock_size is on, loading an image of a different size fails.
    default_config = {
        "mode": "unchanged",
        "dtype": "source",
        "lock_size": False,
        "lock_width": None,
        "lock_height": None,
    }
    config_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "label": "Mode",
                "enum": ["unchanged", "rgb", "grayscale"],
                "default": "unchanged",
            },
            "dtype": {
                "type": "string",
                "label": "Dtype",
                "enum": ["source", "uint8", "uint16", "int16", "float32", "float64"],
                "default": "source",
            },
        },
    }

    def _lock_dims(self, config: dict) -> tuple[int, int] | None:
        cfg = self.merged_config(config)
        if not cfg.get("lock_size"):
            return None
        width, height = cfg.get("lock_width"), cfg.get("lock_height")
        if width and height:
            return int(width), int(height)
        return None

    def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
        cfg = self.merged_config(config)
        mode = cfg["mode"]
        lock = self._lock_dims(config)
        if mode == "unchanged":
            channels = int(cfg["source_channels"]) if cfg.get("source_channels") else None
        else:
            channels = 1 if mode == "grayscale" else 3
        dtype = cfg.get("source_dtype") if cfg["dtype"] == "source" else cfg["dtype"]
        return ImageSpec(
            channels=channels,
            width=lock[0] if lock else None,
            height=lock[1] if lock else None,
            dtype=dtype,
        )

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        path = Path(context["source_image_path"])
        cfg = self.merged_config(config)
        mode = cfg["mode"]
        dtype = cfg["dtype"]
        with Image.open(path) as loaded:
            lock = self._lock_dims(config)
            if lock is not None and (loaded.width, loaded.height) != lock:
                raise ValueError(
                    f"Input size is locked to {lock[0]}x{lock[1]}, but the selected image is "
                    f"{loaded.width}x{loaded.height}."
                )
            if mode == "unchanged":
                array = np.asarray(loaded)
            elif mode == "grayscale":
                array = np.asarray(loaded.convert("L"))
            else:
                array = np.asarray(loaded.convert("RGB"))
            if dtype == "source":
                return array
            return array.astype(dtype)
