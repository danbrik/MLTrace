from __future__ import annotations

from app.modeling.architectures.common import validate_spatiotemporal_model_graph
from app.modeling.architectures.ssim_schema import (
    SSIM_ERROR_METRIC_OPTIONS,
    SSIM_INFERENCE_PROPERTIES,
    SSIM_LOSS_OPTIONS,
    SSIM_TRAINING_PROPERTIES,
)
from app.modeling.base import BaseModelArchitecture


class SpatioTemporalAutoencoderArchitecture(BaseModelArchitecture):
    type = "spatiotemporal_autoencoder"
    label = "SpatioTemporal Autoencoder"
    category = "Neural reconstruction"
    description = "3D-convolutional autoencoder for clip reconstruction with optional future-frame prediction."
    framework = "torch_optional"
    method_family = "spatiotemporal_reconstruction"
    method_version = "1"
    training_mode = "gradient"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "weights"
    builder_kind = "spatiotemporal_autoencoder"
    capabilities = {
        "input_kind": "image_sequence",
        "output_kind": "clip_reconstruction",
        "supports_layer_builder": True,
        "supports_training": True,
        "supports_future_prediction": True,
        "tensor_format": "N,C,T,H,W",
    }
    default_method_config = {
        "input_channels": 1,
        "input_width": 256,
        "input_height": 256,
        "clip_length": 8,
        "future_length": 1,
        "temporal_stride": 1,
        "future_stride": 1,
        "sequence_contiguity_mode": "ordered_index",
        "missing_frame_policy": "skip",
        "score_timestamp_mode": "last_input",
        "prediction_branch": True,
        "output_activation": "sigmoid",
    }
    method_schema = {
        "type": "object",
        "required": [
            "input_channels",
            "input_width",
            "input_height",
            "clip_length",
            "future_length",
            "temporal_stride",
            "future_stride",
            "sequence_contiguity_mode",
            "missing_frame_policy",
            "score_timestamp_mode",
            "prediction_branch",
            "output_activation",
        ],
        "properties": {
            "input_channels": {"type": "integer", "label": "Input channels", "minimum": 1, "maximum": 16, "default": 1},
            "input_width": {"type": "integer", "label": "Input width", "minimum": 1, "default": 256},
            "input_height": {"type": "integer", "label": "Input height", "minimum": 1, "default": 256},
            "clip_length": {
                "type": "integer",
                "label": "Clip length",
                "minimum": 2,
                "default": 8,
                "description": "Number of input frames per sample. The model receives a tensor N,C,T,H,W.",
            },
            "future_length": {
                "type": "integer",
                "label": "Future length",
                "minimum": 0,
                "default": 1,
                "description": "Number of future frames used as prediction targets when the prediction branch is enabled.",
            },
            "temporal_stride": {
                "type": "integer",
                "label": "Temporal stride",
                "minimum": 1,
                "default": 1,
                "description": "Spacing between frames inside the input clip, measured in dataset frame steps.",
            },
            "future_stride": {
                "type": "integer",
                "label": "Future stride",
                "minimum": 1,
                "default": 1,
                "description": "Spacing between predicted future frames. Default 1 predicts the immediate next frame(s).",
            },
            "sequence_contiguity_mode": {
                "type": "string",
                "label": "Sequence continuity",
                "enum": ["ordered_index", "timestamp_cadence"],
                "default": "ordered_index",
                "description": (
                    "Frame order is recommended for videos and frame folders; timestamp cadence requires "
                    "real timestamp gaps to match the folder cadence and is useful for sensor time series."
                ),
            },
            "missing_frame_policy": {
                "type": "string",
                "label": "Missing frame policy",
                "enum": ["skip", "fail"],
                "default": "skip",
                "description": (
                    "Only used with timestamp cadence. skip ignores clips with missing expected frames; "
                    "fail aborts the run with a clear error."
                ),
            },
            "score_timestamp_mode": {
                "type": "string",
                "label": "Score timestamp",
                "enum": ["last_input", "first_future", "center_input"],
                "default": "last_input",
                "description": "Timestamp assigned to a clip result for plotting and sorting.",
            },
            "prediction_branch": {
                "type": "boolean",
                "label": "Prediction branch",
                "default": True,
                "description": "When enabled, the model trains an additional decoder that predicts future frames.",
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
        "batch_size": 8,
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "weight_decay": 0.00001,
        "reconstruction_loss": "mse",
        "prediction_loss": "mse",
        "training_objective": "reconstruction_prediction",
        "prediction_weight": 1.0,
        "prediction_weight_schedule": "linear_decay",
        "prediction_min_weight": 0.2,
        "early_stopping_enabled": True,
        "early_stopping_patience": 10,
        "num_workers": 16,
        "prefetch_factor": 2,
        "validation_fraction": 0.0,
        "amp_enabled": True,
        "log_interval_batches": 20,
    }
    training_schema = {
        "type": "object",
        "required": ["epochs", "batch_size", "learning_rate", "optimizer", "training_objective"],
        "properties": {
            "epochs": {"type": "integer", "label": "Max epochs", "minimum": 1, "default": 1000},
            "batch_size": {"type": "integer", "label": "Batch size", "minimum": 1, "default": 8},
            "learning_rate": {"type": "number", "label": "Learning rate", "minimum": 0, "default": 0.0001},
            "optimizer": {"type": "string", "label": "Optimizer", "enum": ["adam"], "default": "adam"},
            "weight_decay": {"type": "number", "label": "Weight decay", "minimum": 0, "default": 0.00001},
            "reconstruction_loss": {
                "type": "string",
                "label": "Reconstruction loss",
                "enum": SSIM_LOSS_OPTIONS,
                "default": "mse",
            },
            "prediction_loss": {
                "type": "string",
                "label": "Prediction loss",
                "enum": SSIM_LOSS_OPTIONS,
                "default": "mse",
            },
            **SSIM_TRAINING_PROPERTIES,
            "training_objective": {
                "type": "string",
                "label": "Training objective",
                "enum": ["reconstruction", "reconstruction_prediction"],
                "default": "reconstruction_prediction",
                "description": "Use reconstruction only or train reconstruction plus future prediction together.",
            },
            "prediction_weight": {
                "type": "number",
                "label": "Prediction weight",
                "minimum": 0,
                "default": 1.0,
            },
            "prediction_weight_schedule": {
                "type": "string",
                "label": "Prediction weight schedule",
                "enum": ["constant", "linear_decay", "exponential_decay"],
                "default": "linear_decay",
            },
            "prediction_min_weight": {
                "type": "number",
                "label": "Prediction min weight",
                "minimum": 0,
                "default": 0.2,
            },
            "early_stopping_enabled": {"type": "boolean", "label": "Early stopping", "default": True},
            "early_stopping_patience": {"type": "integer", "label": "Early stopping patience", "minimum": 1, "default": 10},
            "num_workers": {"type": "integer", "label": "DataLoader workers", "minimum": 0, "default": 16},
            "prefetch_factor": {"type": "integer", "label": "Prefetch factor", "minimum": 1, "default": 2},
            "validation_fraction": {"type": "number", "label": "Validation fraction", "minimum": 0, "maximum": 0.9, "default": 0.0},
            "amp_enabled": {"type": "boolean", "label": "AMP mixed precision", "default": True},
            "log_interval_batches": {"type": "integer", "label": "Log interval batches", "minimum": 1, "default": 20},
        },
    }
    default_inference_config = {
        "score_mode": "weighted_sum",
        "reconstruction_weight": 1.0,
        "prediction_weight": 1.0,
        "error_metric": "mae",
        "residual_mode": "absolute",
        "frame_score_aggregation": "mean",
        "prediction_horizon": 1,
    }
    inference_schema = {
        "type": "object",
        "properties": {
            "score_mode": {
                "type": "string",
                "label": "Score mode",
                "enum": ["reconstruction_only", "prediction_only", "weighted_sum"],
                "default": "weighted_sum",
            },
            "reconstruction_weight": {"type": "number", "label": "Reconstruction weight", "minimum": 0, "default": 1.0},
            "prediction_weight": {"type": "number", "label": "Prediction weight", "minimum": 0, "default": 1.0},
            "error_metric": {
                "type": "string",
                "label": "Error metric",
                "enum": SSIM_ERROR_METRIC_OPTIONS,
                "default": "mae",
            },
            **SSIM_INFERENCE_PROPERTIES,
            "residual_mode": {
                "type": "string",
                "label": "Residual mode",
                "enum": ["absolute", "squared"],
                "default": "absolute",
            },
            "frame_score_aggregation": {
                "type": "string",
                "label": "Frame score aggregation",
                "enum": ["mean", "p95"],
                "default": "mean",
            },
            "prediction_horizon": {"type": "integer", "label": "Prediction horizon", "minimum": 1, "default": 1},
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
        validate_spatiotemporal_model_graph(method_graph, self.builder_kind, self.merged_method_config(method_config))
