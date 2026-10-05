from __future__ import annotations

import math

from app.modeling.base import BaseModelArchitecture
from app.metrics.aggregation import FRAME_SCORE_AGGREGATION_OPTIONS


class StatisticalReferenceArchitecture(BaseModelArchitecture):
    type = "statistical_reference"
    label = "Statistical Reference Baseline"
    category = "Statistical baseline"
    description = "Pixel-wise normal mean and population standard deviation; anomaly map |I − μ| / (σ + ε)."
    framework = "numpy"
    method_family = "statistical_baseline"
    method_version = "1"
    training_mode = "fit"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "statistical_reference"
    builder_kind = "form"
    capabilities = {"input_kind": "image_collection", "output_kind": "anomaly_map", "supports_layer_builder": False, "supports_training": True}
    default_method_config = {"epsilon": 1e-6}
    method_schema = {"type": "object", "required": ["epsilon"], "properties": {
        "epsilon": {"type": "number", "label": "Epsilon", "default": 1e-6, "minimum": 0,
                    "description": "Positive stabilizer in pipeline intensity units. Population standard deviation (ddof=0), float64; no intensity normalization."}}}
    default_inference_config = {"error_metric": "normalized_deviation", "frame_score_aggregation": "mean"}
    inference_schema = {"type": "object", "properties": {
        "error_metric": {"type": "string", "label": "Anomaly measure", "enum": ["normalized_deviation"], "default": "normalized_deviation"},
        "frame_score_aggregation": {"type": "string", "label": "Spatial aggregation", "enum": FRAME_SCORE_AGGREGATION_OPTIONS, "default": "mean"}}}

    def validate_config(self, method_graph, method_config, training_config=None, inference_config=None):
        super().validate_config(method_graph, method_config, training_config, inference_config)
        epsilon = float((method_config or {}).get("epsilon", 1e-6))
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Epsilon must be finite and greater than zero.")
        if any((method_graph or {}).get(key) for key in ("encoder", "decoder", "nodes", "edges")):
            raise ValueError("Statistical Reference Baseline does not use a layer graph.")
