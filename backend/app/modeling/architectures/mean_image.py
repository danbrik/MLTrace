from __future__ import annotations

from app.modeling.architectures.ssim_schema import SSIM_ERROR_METRIC_OPTIONS, SSIM_INFERENCE_PROPERTIES
from app.modeling.base import BaseModelArchitecture


class MeanImageArchitecture(BaseModelArchitecture):
    type = "mean_image"
    label = "Mean Image"
    category = "Baseline reconstruction"
    description = "Stores a mean-image baseline configuration for later artifact computation."
    framework = "numpy"
    method_family = "statistical_baseline"
    method_version = "1"
    training_mode = "fit"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "mean_image"
    builder_kind = "form"
    capabilities = {
        "input_kind": "image_collection",
        "output_kind": "reference_image",
        "supports_layer_builder": False,
        "supports_training": True,
    }
    default_method_config = {
        "aggregation": "mean",
        "accumulator_dtype": "float32",
        "output_dtype_policy": "source",
        "normalization_mode": "none",
    }
    method_schema = {
        "type": "object",
        "required": ["aggregation", "accumulator_dtype", "output_dtype_policy", "normalization_mode"],
        "properties": {
            "aggregation": {
                "type": "string",
                "label": "Aggregation",
                "enum": ["mean"],
                "default": "mean",
                "description": "How the normal-state images are combined into the reference artifact. V1 supports mean aggregation.",
            },
            "accumulator_dtype": {
                "type": "string",
                "label": "Accumulator dtype",
                "enum": ["float32", "float64"],
                "default": "float32",
                "description": "Numeric dtype used while accumulating pixel values. float64 is more precise; float32 uses less memory.",
            },
            "output_dtype_policy": {
                "type": "string",
                "label": "Output dtype",
                "enum": ["source", "float32"],
                "default": "source",
                "description": "Controls the dtype policy for the stored mean-image artifact. source keeps the input dtype policy; float32 stores a floating-point reference.",
            },
            "normalization_mode": {
                "type": "string",
                "label": "Normalization",
                "enum": ["none", "minmax_for_preview"],
                "default": "none",
                "description": "Optional normalization mode for display/preview behavior. none leaves artifact values unchanged.",
            },
        },
    }
    default_inference_config = {"error_metric": "mse"}
    inference_schema = {
        "type": "object",
        "properties": {
            "error_metric": {"type": "string", "label": "Error metric", "enum": SSIM_ERROR_METRIC_OPTIONS, "default": "mse"},
            **SSIM_INFERENCE_PROPERTIES,
        },
    }

    def validate_config(
        self,
        method_graph: dict | None,
        method_config: dict | None,
        training_config: dict | None = None,
        inference_config: dict | None = None,
    ) -> None:
        super().validate_config(method_graph, method_config, training_config, inference_config)
        graph = method_graph or {}
        if not isinstance(graph, dict):
            raise ValueError("Mean Image model_graph must be an object.")
        if graph.get("encoder") or graph.get("decoder") or graph.get("nodes") or graph.get("edges"):
            raise ValueError("Mean Image uses a form builder and cannot contain a layer graph.")
