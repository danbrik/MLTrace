# Preprocessing Backend Extension Guide

MLTrace preprocessing is intentionally modular. A saved pipeline is JSON, while
the backend resolves that JSON through a registry of `BasePreprocessingStep`
classes. Training and inference use the same step classes as preview.

## Step Contract

Add one file under `backend/app/preprocessing/steps/`, for example
`gaussian_blur.py`, and define one concrete class that inherits from
`BasePreprocessingStep`.

Each step must define:

- `type`: stable machine id stored in pipeline JSON, for example `gaussian_blur`.
- `label`: human-readable UI label.
- `category`: palette grouping in the UI.
- `default_config`: complete runtime defaults for the step.
- `config_schema`: small JSON-schema-like object consumed by the UI and backend validation.
- `output_spec(spec_in, config)`: symbolic shape/type propagation.
- `apply(image, config, context)`: real image transform.

Example:

```python
from __future__ import annotations

import cv2
import numpy as np

from app.preprocessing.base import BasePreprocessingStep, ImageSpec


class ThresholdStep(BasePreprocessingStep):
    type = "threshold"
    label = "Threshold"
    category = "Intensity"
    default_config = {"value": 128}
    config_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "integer", "label": "Value", "minimum": 0, "maximum": 65535, "default": 128},
        },
    }

    def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
        if spec_in is None:
            raise ValueError("threshold requires an input image.")
        return spec_in

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("threshold requires an input image.")
        value = int(self.merged_config(config)["value"])
        return np.where(image >= value, image, 0).astype(image.dtype, copy=False)
```

## Registration

`backend/app/preprocessing/steps/__init__.py` auto-discovers concrete step
classes in that package and registers them. In normal cases no manual registry
edit is needed. Keep the class importable at module top level so discovery,
tests, and DataLoader workers can import it.

## Config Schema

Supported field types are `integer`, `number`, `string`, and `boolean`.
`enum`, `minimum`, `maximum`, `default`, `label`, and `description` are used by
both validation and the UI. Use `description` for hover help.

For custom interactive controls, add `ui_control` to the step schema and make
sure the frontend has a matching control registered. Existing controls include:

- `point_picker`: writes a four-point `source_points` config.
- `crop_box`: writes `x`, `y`, `width`, and `height`.

## Runtime Context

`apply(...)` receives a mutable `context` dict. Common keys:

- `source_image_path`: absolute path passed to the `load_image` step.
- `source_shape`: shape after `load_image`, set by the pipeline runner.

Avoid relying on UI-only state. A step must work in preview, training, testing,
and heatmap execution.

## Preview Display Contract

Preprocessing previews use an absolute, dtype-based display scale rather than
per-image min/max stretching. `uint8` is displayed unchanged, unsigned integers
use their full dtype range, signed integers map their complete dtype range, and
floating-point arrays are expected in `[0, 1]` and clipped for display. This
only affects the PNG shown in the UI; the ndarray passed to the next step is not
modified. The explicit `normalize_for_preview` step is different: as a graph
node it changes the real pipeline output and therefore also affects training.

`warp_perspective` supports `output_shape_mode`:

- `preserve_rectangle` derives width and height from the ordered quadrilateral
  side lengths and avoids forcing non-square regions into a square.
- `manual` uses configured `output_width` and `output_height` and remains useful
  for legacy-compatible output geometry.

## Performance Rules

Training may execute a step millions of times. Keep `apply(...)` lean:

- Do not validate the whole graph inside a step.
- Do not scan folders or touch unrelated files.
- Do not allocate large intermediate arrays unless required.
- Preserve dtype when possible; let the training hot path normalize final arrays.
- Prefer NumPy/OpenCV vectorized operations over Python loops.

The pipeline runner compiles a saved graph once into resolved step instances and
merged configs. This preserves modularity while avoiding per-image graph
validation, registry lookup, and config merging.
