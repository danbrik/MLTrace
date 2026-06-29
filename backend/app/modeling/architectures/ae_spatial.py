from __future__ import annotations

from app.modeling.architectures.common import validate_sequential_model_graph
from app.modeling.architectures.ssim_schema import (
    SSIM_ERROR_METRIC_OPTIONS,
    SSIM_INFERENCE_PROPERTIES,
    SSIM_LOSS_OPTIONS,
    SSIM_TRAINING_PROPERTIES,
)
from app.modeling.base import BaseModelArchitecture


class AeSpatialArchitecture(BaseModelArchitecture):
    type = "ae_spatial"
    label = "AESpatial"
    category = "Neural reconstruction"
    description = "Autoencoder with a spatial bottleneck tensor c x h x w instead of a flat latent vector."
    framework = "torch_optional"
    method_family = "neural_reconstruction"
    method_version = "1"
    training_mode = "gradient"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "weights"
    builder_kind = "sequential_spatial_autoencoder"
    capabilities = {
        "input_kind": "image",
        "output_kind": "reconstruction",
        "supports_layer_builder": True,
        "supports_training": True,
        "bottleneck_kind": "spatial",
    }
    default_method_config = {
        "input_channels": 1,
        "input_width": 256,
        "input_height": 256,
        "bottleneck_channels": 16,
        "output_activation": "sigmoid",
    }
    method_schema = {
        "type": "object",
        "required": ["input_channels", "input_width", "input_height", "bottleneck_channels", "output_activation"],
        "properties": {
            "input_channels": {"type": "integer", "label": "Input channels", "minimum": 1, "maximum": 16, "default": 1},
            "input_width": {"type": "integer", "label": "Input width", "minimum": 1, "default": 256},
            "input_height": {"type": "integer", "label": "Input height", "minimum": 1, "default": 256},
            "bottleneck_channels": {
                "type": "integer",
                "label": "Bottleneck channels",
                "minimum": 1,
                "default": 16,
            },
            "output_activation": {
                "type": "string",
                "label": "Output activation",
                "enum": ["none", "sigmoid", "tanh"],
                "default": "sigmoid",
            },
        },
    }
    default_training_config = {
        "epochs": 1000,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "loss": "mse",
        "optimizer": "adam",
        "weight_decay": 0.00001,
        "early_stopping_enabled": True,
        "early_stopping_patience": 10,
        "num_workers": 16,
        "prefetch_factor": 2,
        "validation_fraction": 0.0,
        "amp_enabled": True,
        "log_interval_batches": 50,
    }
    training_schema = {
        "type": "object",
        "required": ["epochs", "batch_size", "learning_rate", "loss", "optimizer"],
        "properties": {
            "epochs": {"type": "integer", "label": "Max epochs", "minimum": 1, "default": 1000},
            "batch_size": {"type": "integer", "label": "Batch size", "minimum": 1, "default": 32},
            "learning_rate": {"type": "number", "label": "Learning rate", "minimum": 0, "default": 0.0001},
            "loss": {"type": "string", "label": "Loss", "enum": SSIM_LOSS_OPTIONS, "default": "mse"},
            **SSIM_TRAINING_PROPERTIES,
            "optimizer": {"type": "string", "label": "Optimizer", "enum": ["adam"], "default": "adam"},
            "weight_decay": {"type": "number", "label": "Weight decay", "minimum": 0, "default": 0.00001},
            "early_stopping_enabled": {"type": "boolean", "label": "Early stopping", "default": True},
            "early_stopping_patience": {
                "type": "integer",
                "label": "Early stopping patience",
                "minimum": 1,
                "default": 10,
            },
            "num_workers": {"type": "integer", "label": "DataLoader workers", "minimum": 0, "default": 16},
            "prefetch_factor": {"type": "integer", "label": "Prefetch factor", "minimum": 1, "default": 2},
            "validation_fraction": {
                "type": "number",
                "label": "Validation fraction",
                "minimum": 0,
                "maximum": 0.9,
                "default": 0.0,
            },
            "amp_enabled": {"type": "boolean", "label": "AMP mixed precision", "default": True},
            "log_interval_batches": {"type": "integer", "label": "Log interval batches", "minimum": 1, "default": 50},
        },
    }
    default_inference_config = {
        "error_metric": "mse",
        "residual_mode": "squared",
        "frame_score_aggregation": "mean",
    }
    inference_schema = {
        "type": "object",
        "properties": {
            "error_metric": {"type": "string", "label": "Error metric", "enum": SSIM_ERROR_METRIC_OPTIONS, "default": "mse"},
            **SSIM_INFERENCE_PROPERTIES,
            "residual_mode": {
                "type": "string",
                "label": "Residual mode",
                "enum": ["squared", "absolute"],
                "default": "squared",
            },
            "frame_score_aggregation": {
                "type": "string",
                "label": "Frame score aggregation",
                "enum": ["mean", "p95"],
                "default": "mean",
            },
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
        validate_sequential_model_graph(method_graph, self.builder_kind)
