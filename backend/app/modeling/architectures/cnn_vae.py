from __future__ import annotations

from app.modeling.architectures.common import validate_sequential_model_graph
from app.modeling.architectures.ssim_schema import (
    SSIM_ERROR_METRIC_OPTIONS,
    SSIM_INFERENCE_PROPERTIES,
    SSIM_LOSS_OPTIONS,
    SSIM_TRAINING_PROPERTIES,
)
from app.modeling.base import BaseModelArchitecture


class CnnVaeArchitecture(BaseModelArchitecture):
    type = "cnn_vae"
    label = "CNN Variational Autoencoder"
    category = "Neural reconstruction"
    description = "Sequential convolutional VAE configuration with latent sampling metadata."
    framework = "torch_optional"
    method_family = "neural_reconstruction"
    method_version = "1"
    training_mode = "gradient"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "weights"
    builder_kind = "sequential_variational_autoencoder"
    capabilities = {
        "input_kind": "image",
        "output_kind": "reconstruction",
        "supports_layer_builder": True,
        "supports_training": True,
        "uses_reparameterization": True,
    }
    default_method_config = {
        "input_channels": 1,
        "input_width": 256,
        "input_height": 256,
        "latent_dim": 128,
        "kl_weight": 1.0,
        "output_activation": "sigmoid",
    }
    method_schema = {
        "type": "object",
        "required": ["input_channels", "input_width", "input_height", "latent_dim", "kl_weight", "output_activation"],
        "properties": {
            "input_channels": {
                "type": "integer",
                "label": "Input channels",
                "minimum": 1,
                "maximum": 16,
                "default": 1,
                "description": "Number of image channels expected after preprocessing, e.g. 1 for grayscale.",
            },
            "input_width": {
                "type": "integer",
                "label": "Input width",
                "minimum": 1,
                "default": 256,
                "description": "Width in pixels expected after preprocessing. The VAE preset uses 256.",
            },
            "input_height": {
                "type": "integer",
                "label": "Input height",
                "minimum": 1,
                "default": 256,
                "description": "Height in pixels expected after preprocessing. The VAE preset uses 256.",
            },
            "latent_dim": {
                "type": "integer",
                "label": "Latent dim",
                "minimum": 1,
                "default": 128,
                "description": "Length of mu, logvar, and z. Baur-style VAE uses d=128.",
            },
            "kl_weight": {
                "type": "number",
                "label": "KL weight",
                "minimum": 0,
                "default": 1.0,
                "description": "Weight of the KL divergence term. This is the beta parameter in beta-VAE; Baur uses 1.0.",
            },
            "output_activation": {
                "type": "string",
                "label": "Output activation",
                "enum": ["none", "sigmoid", "tanh"],
                "default": "sigmoid",
                "description": "Activation applied to the reconstruction. Sigmoid constrains output to [0, 1].",
            },
        },
    }
    default_training_config = {
        "epochs": 1000,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "reconstruction_loss": "l1",
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
        "required": ["epochs", "batch_size", "learning_rate", "reconstruction_loss", "optimizer"],
        "properties": {
            "epochs": {
                "type": "integer",
                "label": "Max epochs",
                "minimum": 1,
                "default": 1000,
                "description": "Maximum epochs. With early stopping enabled, training can stop earlier.",
            },
            "batch_size": {
                "type": "integer",
                "label": "Batch size",
                "minimum": 1,
                "default": 32,
                "description": "Number of images per optimizer step.",
            },
            "learning_rate": {
                "type": "number",
                "label": "Learning rate",
                "minimum": 0,
                "default": 0.0001,
                "description": "Adam step size. Baur Table II reports 1e-4.",
            },
            "reconstruction_loss": {
                "type": "string",
                "label": "Reconstruction loss",
                "enum": SSIM_LOSS_OPTIONS,
                "default": "l1",
                "description": "Reconstruction term L_rec. L1 is Baur-style; MSE/L2 emphasizes larger deviations.",
            },
            **SSIM_TRAINING_PROPERTIES,
            "optimizer": {
                "type": "string",
                "label": "Optimizer",
                "enum": ["adam"],
                "default": "adam",
                "description": "Optimizer used for gradient training. V1 supports Adam.",
            },
            "weight_decay": {
                "type": "number",
                "label": "Weight decay",
                "minimum": 0,
                "default": 0.00001,
                "description": "L2 regularization applied by AdamW-style weight decay in the optimizer.",
            },
            "early_stopping_enabled": {
                "type": "boolean",
                "label": "Early stopping",
                "default": True,
                "description": "Stops before max epochs when the monitored loss no longer improves.",
            },
            "early_stopping_patience": {
                "type": "integer",
                "label": "Early stopping patience",
                "minimum": 1,
                "default": 10,
                "description": "Number of epochs without improvement before early stopping triggers.",
            },
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
    default_inference_config = {"error_metric": "mse", "sample_count": 1}
    inference_schema = {
        "type": "object",
        "properties": {
            "error_metric": {
                "type": "string",
                "label": "Error metric",
                "enum": SSIM_ERROR_METRIC_OPTIONS,
                "default": "mse",
                "description": "Stored scalar error metric for inference summaries. Heatmaps have separate visualization settings.",
            },
            **SSIM_INFERENCE_PROPERTIES,
            "sample_count": {
                "type": "integer",
                "label": "Samples",
                "minimum": 1,
                "maximum": 100,
                "default": 1,
                "description": "VAE reconstruction samples per image. 1 uses deterministic mu and is fast; 100 is Baur-style Monte Carlo and much slower.",
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
