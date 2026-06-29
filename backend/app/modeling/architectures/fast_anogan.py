from __future__ import annotations

from app.modeling.base import BaseModelArchitecture


_SSIM_INFERENCE_FIELDS = {
    "ssim_window_size": {"type": "integer", "default": 11, "minimum": 3, "label": "SSIM window"},
    "ssim_k1": {
        "type": "number",
        "default": 0.01,
        "minimum": 0.0,
        "label": "SSIM K1",
        "description": "Standard SSIM K constant. MLTrace computes C1=(K1*data_range)^2.",
    },
    "ssim_k2": {
        "type": "number",
        "default": 0.03,
        "minimum": 0.0,
        "label": "SSIM K2",
        "description": "Standard SSIM K constant. MLTrace computes C2=(K2*data_range)^2.",
    },
    "ssim_data_range": {"type": "number", "default": 1.0, "minimum": 0.000001, "label": "SSIM data range"},
}


class FastAnoganArchitecture(BaseModelArchitecture):
    type = "fast_anogan"
    label = "fastAnoGAN"
    category = "GAN reconstruction"
    description = (
        "Two-stage fast AnoGAN: WGAN-GP learns normal image generation, then an encoder learns fast image-to-latent "
        "mapping for residual and discriminator-feature anomaly scoring."
    )
    framework = "torch"
    method_family = "gan_reconstruction"
    method_version = "1"
    training_mode = "gradient"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "gan_bundle"
    builder_kind = "fast_anogan"
    capabilities = {"reconstruction": True, "feature_score": True, "pixel_heatmap": True}

    default_method_config = {
        "input_channels": 1,
        "input_width": 64,
        "input_height": 64,
        "latent_dim": 128,
        "latent_distribution": "normal",
        "encoder_output_activation": "tanh",
        "generator_seed_size": 4,
        "feature_layer": "critic_blocks",
        "output_activation": "tanh",
        "kappa": 1.0,
    }
    method_schema = {
        "type": "object",
        "required": ["input_channels", "input_width", "input_height", "latent_dim"],
        "properties": {
            "input_channels": {"type": "integer", "default": 1, "minimum": 1, "label": "Input channels"},
            "input_width": {"type": "integer", "default": 64, "minimum": 16, "label": "Input width"},
            "input_height": {"type": "integer", "default": 64, "minimum": 16, "label": "Input height"},
            "latent_dim": {
                "type": "integer",
                "default": 128,
                "minimum": 1,
                "label": "Latent dim",
                "description": "Length of z. Schlegl uses a 128-dimensional normally distributed latent space.",
            },
            "latent_distribution": {
                "type": "string",
                "enum": ["normal"],
                "default": "normal",
                "label": "Latent distribution",
            },
            "encoder_output_activation": {
                "type": "string",
                "enum": ["tanh", "none"],
                "default": "tanh",
                "label": "Encoder output activation",
                "description": "tanh constrains encoded z approximately to [-1, 1], matching the paper's +/-1 sigma constraint.",
            },
            "generator_seed_size": {"type": "integer", "default": 4, "minimum": 1, "label": "Generator seed size"},
            "feature_layer": {
                "type": "string",
                "enum": ["critic_blocks"],
                "default": "critic_blocks",
                "label": "Feature layer",
                "description": "Critic feature tensor used for the discriminator feature residual.",
            },
            "output_activation": {
                "type": "string",
                "enum": ["tanh", "sigmoid", "none"],
                "default": "tanh",
                "label": "Generator output activation",
            },
            "kappa": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "label": "Kappa",
                "description": "Weight for the discriminator feature residual in izif training and combined scoring.",
            },
        },
    }

    default_training_config = {
        "wgan_iterations": 100000,
        "encoder_iterations": 50000,
        "batch_size": 64,
        "critic_updates_per_generator": 5,
        "gradient_penalty_lambda": 10.0,
        "wgan_optimizer": "adam",
        "wgan_learning_rate": 0.0001,
        "encoder_optimizer": "rmsprop",
        "encoder_learning_rate": 0.00005,
        "encoder_training_mode": "izif",
        "kappa": 1.0,
        "num_workers": 16,
        "prefetch_factor": 2,
        "amp_enabled": True,
        "log_interval_iterations": 100,
    }
    training_schema = {
        "type": "object",
        "properties": {
            "wgan_iterations": {"type": "integer", "default": 100000, "minimum": 1, "label": "WGAN iterations"},
            "encoder_iterations": {"type": "integer", "default": 50000, "minimum": 1, "label": "Encoder iterations"},
            "batch_size": {"type": "integer", "default": 64, "minimum": 1, "label": "Batch size"},
            "critic_updates_per_generator": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "label": "Critic updates per generator",
                "description": "Schlegl follows WGAN-GP and updates the critic 5 times per generator update.",
            },
            "gradient_penalty_lambda": {"type": "number", "default": 10.0, "minimum": 0.0, "label": "GP lambda"},
            "wgan_optimizer": {"type": "string", "enum": ["adam"], "default": "adam", "label": "WGAN optimizer"},
            "wgan_learning_rate": {"type": "number", "default": 0.0001, "minimum": 0.0, "label": "WGAN LR"},
            "encoder_optimizer": {"type": "string", "enum": ["rmsprop"], "default": "rmsprop", "label": "Encoder optimizer"},
            "encoder_learning_rate": {"type": "number", "default": 0.00005, "minimum": 0.0, "label": "Encoder LR"},
            "encoder_training_mode": {
                "type": "string",
                "enum": ["izif", "izi", "ziz"],
                "default": "izif",
                "label": "Encoder training mode",
                "description": "izif = image residual plus critic feature residual; izi = image residual only; ziz = latent regression on generated images.",
            },
            "kappa": {"type": "number", "default": 1.0, "minimum": 0.0, "label": "Kappa"},
            "num_workers": {"type": "integer", "default": 16, "minimum": 0, "label": "DataLoader workers"},
            "prefetch_factor": {"type": "integer", "default": 2, "minimum": 1, "label": "Prefetch factor"},
            "amp_enabled": {"type": "boolean", "default": True, "label": "AMP"},
            "log_interval_iterations": {"type": "integer", "default": 100, "minimum": 1, "label": "Log interval"},
        },
    }

    default_inference_config = {
        "score_mode": "combined",
        "residual_metric": "mse",
        "kappa": 1.0,
        "heatmap_source": "pixel_residual",
    }
    inference_schema = {
        "type": "object",
        "properties": {
            "score_mode": {
                "type": "string",
                "enum": ["combined", "residual_only", "feature_only"],
                "default": "combined",
                "label": "Score mode",
            },
            "residual_metric": {
                "type": "string",
                "enum": ["mse", "mae", "ssim_distance"],
                "default": "mse",
                "label": "Residual metric",
            },
            "kappa": {"type": "number", "default": 1.0, "minimum": 0.0, "label": "Kappa"},
            "heatmap_source": {
                "type": "string",
                "enum": ["pixel_residual"],
                "default": "pixel_residual",
                "label": "Heatmap source",
            },
            **_SSIM_INFERENCE_FIELDS,
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
