from __future__ import annotations

from app.modeling.architectures.common import validate_sequential_model_graph
from app.modeling.base import BaseModelArchitecture


class CnnAutoencoderArchitecture(BaseModelArchitecture):
    type = "cnn_autoencoder"
    label = "CNN Autoencoder"
    category = "Neural reconstruction"
    description = "Sequential convolutional autoencoder configuration for image reconstruction."
    framework = "torch_optional"
    method_family = "neural_reconstruction"
    method_version = "1"
    training_mode = "gradient"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "weights"
    builder_kind = "sequential_autoencoder"
    capabilities = {
        "input_kind": "image",
        "output_kind": "reconstruction",
        "supports_layer_builder": True,
        "supports_training": True,
    }
    default_method_config = {
        "input_channels": 1,
        "input_width": 160,
        "input_height": 120,
        "latent_dim": 64,
        "output_activation": "sigmoid",
    }
    method_schema = {
        "type": "object",
        "required": ["input_channels", "input_width", "input_height", "latent_dim", "output_activation"],
        "properties": {
            "input_channels": {"type": "integer", "label": "Input channels", "minimum": 1, "maximum": 16, "default": 1},
            "input_width": {"type": "integer", "label": "Input width", "minimum": 1, "default": 160},
            "input_height": {"type": "integer", "label": "Input height", "minimum": 1, "default": 120},
            "latent_dim": {"type": "integer", "label": "Latent dim", "minimum": 1, "default": 64},
            "output_activation": {
                "type": "string",
                "label": "Output activation",
                "enum": ["none", "sigmoid", "tanh"],
                "default": "sigmoid",
            },
        },
    }
    default_training_config = {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "loss": "mse",
        "num_workers": 16,
        "prefetch_factor": 2,
        "validation_fraction": 0.0,
        "amp_enabled": True,
        "log_interval_batches": 50,
    }
    training_schema = {
        "type": "object",
        "required": ["epochs", "batch_size", "learning_rate", "loss"],
        "properties": {
            "epochs": {"type": "integer", "label": "Epochs", "minimum": 1, "default": 50},
            "batch_size": {"type": "integer", "label": "Batch size", "minimum": 1, "default": 16},
            "learning_rate": {"type": "number", "label": "Learning rate", "minimum": 0, "default": 0.001},
            "loss": {"type": "string", "label": "Loss", "enum": ["mse", "l1", "smooth_l1"], "default": "mse"},
            "num_workers": {
                "type": "integer",
                "label": "DataLoader workers",
                "minimum": 0,
                "default": 16,
                "description": "Parallel worker processes used for image loading and preprocessing. Legacy CUDA runs used 16.",
            },
            "prefetch_factor": {
                "type": "integer",
                "label": "Prefetch factor",
                "minimum": 1,
                "default": 2,
                "description": "Number of batches each worker preloads. Only used when DataLoader workers are enabled.",
            },
            "validation_fraction": {
                "type": "number",
                "label": "Validation fraction",
                "minimum": 0,
                "maximum": 0.9,
                "default": 0.0,
                "description": "Fraction held out for validation per epoch. Use 0.0 to match the legacy training path.",
            },
            "amp_enabled": {
                "type": "boolean",
                "label": "AMP mixed precision",
                "default": True,
                "description": "Use automatic mixed precision on CUDA. Disable it for strict legacy comparisons.",
            },
            "log_interval_batches": {
                "type": "integer",
                "label": "Log interval batches",
                "minimum": 1,
                "default": 50,
                "description": "How often training logs batch throughput and timing diagnostics.",
            },
        },
    }
    default_inference_config = {"error_metric": "mse"}
    inference_schema = {
        "type": "object",
        "properties": {
            "error_metric": {"type": "string", "label": "Error metric", "enum": ["mse", "mae"], "default": "mse"},
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
