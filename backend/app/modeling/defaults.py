from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import models
from app.schemas import MethodConfigurationCreate
from app.services import _normalize_method_payload, _replace_method_parameter_index, create_method_configuration

logger = logging.getLogger("mltrace.modeling.defaults")


INPUT_SIZE = 256
INPUT_CHANNELS = 1
ENCODER_CHANNELS = [32, 64, 128, 256, 256]
BOTTLENECK_SPATIAL_SIZE = 8
BOTTLENECK_FEATURES = 256 * BOTTLENECK_SPATIAL_SIZE * BOTTLENECK_SPATIAL_SIZE


DEFAULT_TRAINING_CONFIG = {
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

DEFAULT_INFERENCE_CONFIG = {
    "error_metric": "mse",
    "residual_mode": "squared",
    "frame_score_aggregation": "mean",
}

VAE_TRAINING_CONFIG = {
    **{key: value for key, value in DEFAULT_TRAINING_CONFIG.items() if key != "loss"},
    "reconstruction_loss": "l1",
}

VAE_INFERENCE_CONFIG = {
    "error_metric": "mse",
    "sample_count": 1,
}

STAE_TRAINING_CONFIG = {
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

STAE_INFERENCE_CONFIG = {
    "score_mode": "weighted_sum",
    "reconstruction_weight": 1.0,
    "prediction_weight": 1.0,
    "residual_mode": "absolute",
    "frame_score_aggregation": "mean",
    "prediction_horizon": 1,
}

STAE_RECONSTRUCTION_INFERENCE_CONFIG = {
    "score_mode": "reconstruction_only",
    "reconstruction_weight": 1.0,
    "prediction_weight": 0.0,
    "residual_mode": "absolute",
    "frame_score_aggregation": "mean",
    "prediction_horizon": 1,
}

FAST_ANOGAN_TRAINING_CONFIG = {
    "optimizer": "adam",
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

FAST_ANOGAN_INFERENCE_CONFIG = {
    "score_mode": "combined",
    "residual_metric": "mse",
    "kappa": 1.0,
    "heatmap_source": "pixel_residual",
}


def _layer(layer_id: str, layer_type: str, **config) -> dict:
    return {"id": layer_id, "type": layer_type, "config": config}


def _encoder_prefix() -> list[dict]:
    layers: list[dict] = []
    for index, out_channels in enumerate(ENCODER_CHANNELS, start=1):
        layers.append(
            _layer(
                f"enc-conv-{index}",
                "Conv2d",
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )
        )
        layers.append(_layer(f"enc-relu-{index}", "ReLU", inplace=False))
    return layers


def _decoder_upsampling_layers(prefix: str = "dec") -> list[dict]:
    layers: list[dict] = []
    for index, out_channels in enumerate([256, 128, 64, 32, INPUT_CHANNELS], start=1):
        layers.append(
            _layer(
                f"{prefix}-deconv-{index}",
                "ConvTranspose2d",
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True,
            )
        )
        if index < 5:
            layers.append(_layer(f"{prefix}-relu-{index}", "ReLU", inplace=False))
    return layers


def _dense_graph(latent_dim: int) -> dict:
    return {
        "builder_kind": "sequential_autoencoder",
        "encoder": [
            *_encoder_prefix(),
            _layer("enc-flatten", "Flatten", start_dim=1, end_dim=-1),
            _layer("enc-latent", "Linear", out_features=latent_dim, bias=True),
        ],
        "latent": {"latent_dim": latent_dim, "bottleneck_kind": "dense"},
        "decoder": [
            _layer("dec-expand", "Linear", out_features=BOTTLENECK_FEATURES, bias=True),
            _layer("dec-unflatten", "Unflatten", channels=256, height=BOTTLENECK_SPATIAL_SIZE, width=BOTTLENECK_SPATIAL_SIZE),
            *_decoder_upsampling_layers(),
        ],
    }


def _vae_graph(latent_dim: int) -> dict:
    return {
        "builder_kind": "sequential_variational_autoencoder",
        "encoder": _encoder_prefix(),
        "latent": {
            "latent_dim": latent_dim,
            "kl_weight": 1.0,
            "reparameterization": True,
            "bottleneck_kind": "variational_dense",
        },
        "decoder": _decoder_upsampling_layers(),
    }


def _spatial_graph(bottleneck_channels: int) -> dict:
    return {
        "builder_kind": "sequential_spatial_autoencoder",
        "encoder": [
            *_encoder_prefix(),
            _layer(
                "enc-spatial-bottleneck",
                "Conv2d",
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        ],
        "latent": {
            "bottleneck_kind": "spatial",
            "bottleneck_channels": bottleneck_channels,
            "height": BOTTLENECK_SPATIAL_SIZE,
            "width": BOTTLENECK_SPATIAL_SIZE,
        },
        "decoder": [
            _layer(
                "dec-spatial-seed",
                "Conv2d",
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            *_decoder_upsampling_layers(),
        ],
    }


def _stae_conv(layer_id: str, out_channels: int) -> dict:
    return _layer(
        layer_id,
        "Conv3d",
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )


def _stae_deconv(layer_id: str, out_channels: int, *, temporal_upsample: bool) -> dict:
    return _layer(
        layer_id,
        "ConvTranspose3d",
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        output_padding=0,
        stride_t=2 if temporal_upsample else 1,
        stride_xy=2,
        output_padding_t=1 if temporal_upsample else 0,
        output_padding_xy=1,
        bias=True,
    )


def _stae_graph() -> dict:
    encoder: list[dict] = []
    for index, out_channels in enumerate([32, 64, 128], start=1):
        encoder.extend(
            [
                _stae_conv(f"stae-enc-conv-{index}", out_channels),
                _layer(f"stae-enc-bn-{index}", "BatchNorm3d", num_features=out_channels, eps=0.00001, momentum=0.1),
                _layer(f"stae-enc-act-{index}", "LeakyReLU", negative_slope=0.01, inplace=False),
                _layer(f"stae-enc-pool-{index}", "MaxPool3d", kernel_size=2, stride=2, padding=0),
            ]
        )
    recon_decoder: list[dict] = []
    pred_decoder: list[dict] = []
    for index, out_channels in enumerate([64, 32, INPUT_CHANNELS], start=1):
        recon_decoder.append(_stae_deconv(f"stae-rec-deconv-{index}", out_channels, temporal_upsample=True))
        pred_decoder.append(_stae_deconv(f"stae-pred-deconv-{index}", out_channels, temporal_upsample=False))
        if index < 3:
            recon_decoder.append(_layer(f"stae-rec-act-{index}", "LeakyReLU", negative_slope=0.01, inplace=False))
            pred_decoder.append(_layer(f"stae-pred-act-{index}", "LeakyReLU", negative_slope=0.01, inplace=False))
    return {
        "builder_kind": "spatiotemporal_autoencoder",
        "encoder": encoder,
        "latent": {"bottleneck_kind": "spatiotemporal", "shape": "128x1x32x32"},
        "decoder": recon_decoder,
        "prediction_decoder": pred_decoder,
    }


def _fast_anogan_block(prefix: str, index: int, out_channels: int, *, direction: str, normalization: str) -> dict:
    return {
        "id": f"{prefix}-{index}",
        "block_type": "residual",
        "direction": direction,
        "out_channels": out_channels,
        "normalization": normalization,
    }


def _fast_anogan_graph() -> dict:
    return {
        "builder_kind": "fast_anogan",
        "generator_blocks": [
            _fast_anogan_block("gan-gen-up", index, channels, direction="up", normalization="none")
            for index, channels in enumerate([512, 256, 128, 64], start=1)
        ],
        "critic_blocks": [
            _fast_anogan_block("gan-critic-down", index, channels, direction="down", normalization="layer_norm")
            for index, channels in enumerate([128, 256, 512, 512], start=1)
        ],
        "encoder_blocks": [
            _fast_anogan_block("gan-enc-down", index, channels, direction="down", normalization="none")
            for index, channels in enumerate([128, 256, 512, 512], start=1)
        ],
        "feature_layer": "critic_blocks",
    }


def _fast_anogan_method_config() -> dict:
    return {
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


def _stae_paper_graph(*, prediction_branch: bool) -> dict:
    encoder: list[dict] = []
    for index, out_channels in enumerate([32, 48, 64], start=1):
        encoder.extend(
            [
                _layer(
                    f"paper-stae-enc-conv-{index}",
                    "Conv3d",
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                _layer(
                    f"paper-stae-enc-bn-{index}",
                    "BatchNorm3d",
                    num_features=out_channels,
                    eps=0.00001,
                    momentum=0.1,
                ),
                _layer(
                    f"paper-stae-enc-act-{index}",
                    "LeakyReLU",
                    negative_slope=0.01,
                    inplace=False,
                ),
                _layer(
                    f"paper-stae-enc-pool-{index}",
                    "MaxPool3d",
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
            ]
        )
    encoder.extend(
        [
            _layer(
                "paper-stae-enc-conv-4",
                "Conv3d",
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            _layer(
                "paper-stae-enc-bn-4",
                "BatchNorm3d",
                num_features=64,
                eps=0.00001,
                momentum=0.1,
            ),
            _layer(
                "paper-stae-enc-act-4",
                "LeakyReLU",
                negative_slope=0.01,
                inplace=False,
            ),
        ]
    )

    reconstruction_decoder: list[dict] = []
    for index, out_channels in enumerate([64, 48, INPUT_CHANNELS], start=1):
        reconstruction_decoder.append(
            _layer(
                f"paper-stae-rec-deconv-{index}",
                "ConvTranspose3d",
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0,
                stride_t=2,
                stride_xy=2,
                output_padding_t=1,
                output_padding_xy=1,
                bias=True,
            )
        )
        if index < 3:
            reconstruction_decoder.append(
                _layer(
                    f"paper-stae-rec-act-{index}",
                    "LeakyReLU",
                    negative_slope=0.01,
                    inplace=False,
                )
            )

    prediction_decoder: list[dict] = []
    if prediction_branch:
        for index, out_channels in enumerate([64, 48, INPUT_CHANNELS], start=1):
            prediction_decoder.extend(
                [
                    _layer(
                        f"paper-stae-pred-deconv-{index}",
                        "ConvTranspose3d",
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        output_padding=0,
                        stride_t=2,
                        stride_xy=2,
                        output_padding_t=1,
                        output_padding_xy=1,
                        bias=True,
                    ),
                    _layer(
                        f"paper-stae-pred-act-{index}",
                        "LeakyReLU",
                        negative_slope=0.01,
                        inplace=False,
                    ),
                ]
            )

    return {
        "builder_kind": "spatiotemporal_autoencoder",
        "encoder": encoder,
        "latent": {"bottleneck_kind": "spatiotemporal", "shape": "64x2x16x16"},
        "decoder": reconstruction_decoder,
        "prediction_decoder": prediction_decoder,
    }


def _base_method_config() -> dict:
    return {
        "input_channels": INPUT_CHANNELS,
        "input_width": INPUT_SIZE,
        "input_height": INPUT_SIZE,
        "output_activation": "sigmoid",
    }


def _paper_stae_method_config(*, prediction_branch: bool) -> dict:
    return {
        "input_channels": INPUT_CHANNELS,
        "input_width": 128,
        "input_height": 128,
        "clip_length": 16,
        "future_length": 16 if prediction_branch else 0,
        "temporal_stride": 1,
        "future_stride": 1,
        "missing_frame_policy": "skip",
        "score_timestamp_mode": "last_input",
        "prediction_branch": prediction_branch,
        "output_activation": "sigmoid",
    }


def _paper_stae_training_config(*, prediction_branch: bool) -> dict:
    if prediction_branch:
        return {
            **STAE_TRAINING_CONFIG,
            "training_objective": "reconstruction_prediction",
            "prediction_weight": 1.0,
            "prediction_weight_schedule": "linear_decay",
            "prediction_min_weight": 0.0,
            "prediction_horizon_weight_schedule": "linear_decay",
        }
    return {
        **STAE_TRAINING_CONFIG,
        "training_objective": "reconstruction",
        "prediction_weight": 0.0,
        "prediction_weight_schedule": "constant",
        "prediction_min_weight": 0.0,
    }


def default_method_payloads() -> list[MethodConfigurationCreate]:
    channels = ", ".join(str(item) for item in ENCODER_CHANNELS)
    return [
        MethodConfigurationCreate(
            name="AEDense d64 default",
            description=f"Deployment default dense autoencoder, latent d=64, encoder channels [{channels}], input 256x256x1.",
            method_type="ae_dense",
            method_config={**_base_method_config(), "latent_dim": 64},
            training_config=DEFAULT_TRAINING_CONFIG,
            inference_config=DEFAULT_INFERENCE_CONFIG,
            method_graph=_dense_graph(64),
        ),
        MethodConfigurationCreate(
            name="AEDense d256 default",
            description=f"Deployment default dense autoencoder, latent d=256, encoder channels [{channels}], input 256x256x1.",
            method_type="ae_dense",
            method_config={**_base_method_config(), "latent_dim": 256},
            training_config=DEFAULT_TRAINING_CONFIG,
            inference_config=DEFAULT_INFERENCE_CONFIG,
            method_graph=_dense_graph(256),
        ),
        MethodConfigurationCreate(
            name="AESpatial c16 default",
            description=f"Deployment default spatial autoencoder, z=16x8x8, encoder channels [{channels}], input 256x256x1.",
            method_type="ae_spatial",
            method_config={**_base_method_config(), "bottleneck_channels": 16},
            training_config=DEFAULT_TRAINING_CONFIG,
            inference_config=DEFAULT_INFERENCE_CONFIG,
            method_graph=_spatial_graph(16),
        ),
        MethodConfigurationCreate(
            name="AESpatial c64 default",
            description=f"Deployment default spatial autoencoder, z=64x8x8, encoder channels [{channels}], input 256x256x1.",
            method_type="ae_spatial",
            method_config={**_base_method_config(), "bottleneck_channels": 64},
            training_config=DEFAULT_TRAINING_CONFIG,
            inference_config=DEFAULT_INFERENCE_CONFIG,
            method_graph=_spatial_graph(64),
        ),
        MethodConfigurationCreate(
            name="VAE Baur d128 default",
            description=(
                f"Baur-style variational autoencoder, latent mu/logvar d=128, KL weight 1.0, "
                f"L1 reconstruction loss, encoder channels [{channels}], input 256x256x1."
            ),
            method_type="cnn_vae",
            method_config={**_base_method_config(), "latent_dim": 128, "kl_weight": 1.0},
            training_config=VAE_TRAINING_CONFIG,
            inference_config=VAE_INFERENCE_CONFIG,
            method_graph=_vae_graph(128),
        ),
        MethodConfigurationCreate(
            name="STAE reconstruction prediction default",
            description=(
                "3D spatio-temporal autoencoder default with clip_length=8, future_length=1, "
                "reconstruction plus future prediction, Conv3d/BatchNorm3d/LeakyReLU/MaxPool3d encoder."
            ),
            method_type="spatiotemporal_autoencoder",
            method_config={
                **_base_method_config(),
                "clip_length": 8,
                "future_length": 1,
                "temporal_stride": 1,
                "future_stride": 1,
                "missing_frame_policy": "skip",
                "score_timestamp_mode": "last_input",
                "prediction_branch": True,
            },
            training_config=STAE_TRAINING_CONFIG,
            inference_config=STAE_INFERENCE_CONFIG,
            method_graph=_stae_graph(),
        ),
        MethodConfigurationCreate(
            name="STAE Reconstruction paper default",
            description=(
                "Paper-near 3D spatio-temporal autoencoder for reconstruction only: "
                "input clip 16x128x128, Conv3d/BatchNorm3d/LeakyReLU/MaxPool3d encoder "
                "with Zhao-style channels [32, 48, 64, 64] and bottleneck 64x2x16x16."
            ),
            method_type="spatiotemporal_autoencoder",
            method_config=_paper_stae_method_config(prediction_branch=False),
            training_config=_paper_stae_training_config(prediction_branch=False),
            inference_config=STAE_RECONSTRUCTION_INFERENCE_CONFIG,
            method_graph=_stae_paper_graph(prediction_branch=False),
        ),
        MethodConfigurationCreate(
            name="STAE Reconstruction + Future Prediction paper default",
            description=(
                "Zhao-near 3D spatio-temporal autoencoder with reconstruction and multi-frame future prediction: "
                "input clip 16x128x128, future_length=16, channels [32, 48, 64, 64], shared bottleneck 64x2x16x16."
            ),
            method_type="spatiotemporal_autoencoder",
            method_config=_paper_stae_method_config(prediction_branch=True),
            training_config=_paper_stae_training_config(prediction_branch=True),
            inference_config=STAE_INFERENCE_CONFIG,
            method_graph=_stae_paper_graph(prediction_branch=True),
        ),
        MethodConfigurationCreate(
            name="fastAnoGAN paper default",
            description=(
                "Paper-near fastAnoGAN default: 64x64x1 input, latent_dim=128, ResNet WGAN-GP generator/critic, "
                "LayerNorm critic blocks, izif encoder training, kappa=1.0."
            ),
            method_type="fast_anogan",
            method_config=_fast_anogan_method_config(),
            training_config=FAST_ANOGAN_TRAINING_CONFIG,
            inference_config=FAST_ANOGAN_INFERENCE_CONFIG,
            method_graph=_fast_anogan_graph(),
        ),
    ]


def _old_paper_stae_graph(*, prediction_branch: bool) -> dict:
    encoder: list[dict] = []
    for index, out_channels in enumerate([16, 32, 64], start=1):
        encoder.extend(
            [
                _layer(
                    f"paper-stae-enc-conv-{index}",
                    "Conv3d",
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                _layer(
                    f"paper-stae-enc-bn-{index}",
                    "BatchNorm3d",
                    num_features=out_channels,
                    eps=0.00001,
                    momentum=0.1,
                ),
                _layer(f"paper-stae-enc-act-{index}", "LeakyReLU", negative_slope=0.01, inplace=False),
                _layer(f"paper-stae-enc-pool-{index}", "MaxPool3d", kernel_size=2, stride=2, padding=0),
            ]
        )

    reconstruction_decoder: list[dict] = []
    for index, out_channels in enumerate([32, 16, INPUT_CHANNELS], start=1):
        reconstruction_decoder.append(
            _layer(
                f"paper-stae-rec-deconv-{index}",
                "ConvTranspose3d",
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0,
                stride_t=2,
                stride_xy=2,
                output_padding_t=1,
                output_padding_xy=1,
                bias=True,
            )
        )
        if index < 3:
            reconstruction_decoder.append(
                _layer(f"paper-stae-rec-act-{index}", "LeakyReLU", negative_slope=0.01, inplace=False)
            )

    prediction_decoder: list[dict] = []
    if prediction_branch:
        for index, out_channels in enumerate([32, 16], start=1):
            prediction_decoder.extend(
                [
                    _layer(
                        f"paper-stae-pred-deconv-{index}",
                        "ConvTranspose3d",
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        output_padding=0,
                        stride_t=1,
                        stride_xy=2,
                        output_padding_t=0,
                        output_padding_xy=1,
                        bias=True,
                    ),
                    _layer(f"paper-stae-pred-act-{index}", "LeakyReLU", negative_slope=0.01, inplace=False),
                ]
            )
        prediction_decoder.extend(
            [
                _layer(
                    "paper-stae-pred-deconv-3",
                    "ConvTranspose3d",
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=0,
                    stride_t=1,
                    stride_xy=2,
                    output_padding_t=0,
                    output_padding_xy=1,
                    bias=True,
                ),
                _layer(
                    "paper-stae-pred-out",
                    "Conv3d",
                    out_channels=INPUT_CHANNELS,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    kernel_size_t=2,
                    kernel_size_xy=3,
                    padding_t=0,
                    padding_xy=1,
                    bias=True,
                ),
            ]
        )

    return {
        "builder_kind": "spatiotemporal_autoencoder",
        "encoder": encoder,
        "latent": {"bottleneck_kind": "spatiotemporal", "shape": "64x2x16x16"},
        "decoder": reconstruction_decoder,
        "prediction_decoder": prediction_decoder,
    }


def _old_paper_stae_method_config(*, prediction_branch: bool) -> dict:
    return {
        "input_channels": INPUT_CHANNELS,
        "input_width": 128,
        "input_height": 128,
        "clip_length": 16,
        "future_length": 1 if prediction_branch else 0,
        "temporal_stride": 1,
        "future_stride": 1,
        "missing_frame_policy": "skip",
        "score_timestamp_mode": "last_input",
        "prediction_branch": prediction_branch,
        "output_activation": "sigmoid",
    }


def _old_paper_stae_training_config(*, prediction_branch: bool) -> dict:
    if prediction_branch:
        return {
            **STAE_TRAINING_CONFIG,
            "training_objective": "reconstruction_prediction",
            "prediction_weight": 1.0,
            "prediction_weight_schedule": "linear_decay",
            "prediction_min_weight": 0.2,
        }
    return {
        **STAE_TRAINING_CONFIG,
        "training_objective": "reconstruction",
        "prediction_weight": 0.0,
        "prediction_weight_schedule": "constant",
        "prediction_min_weight": 0.0,
    }


def _old_paper_payloads_by_name() -> dict[str, MethodConfigurationCreate]:
    return {
        "STAE Reconstruction paper default": MethodConfigurationCreate(
            name="STAE Reconstruction paper default",
            description=(
                "Paper-near 3D spatio-temporal autoencoder for reconstruction only: "
                "input clip 16x128x128, Conv3d/BatchNorm3d/LeakyReLU/MaxPool3d encoder "
                "with channels [16, 32, 64] and bottleneck 64x2x16x16."
            ),
            method_type="spatiotemporal_autoencoder",
            method_config=_old_paper_stae_method_config(prediction_branch=False),
            training_config=_old_paper_stae_training_config(prediction_branch=False),
            inference_config=STAE_RECONSTRUCTION_INFERENCE_CONFIG,
            method_graph=_old_paper_stae_graph(prediction_branch=False),
        ),
        "STAE Reconstruction + Future Prediction paper default": MethodConfigurationCreate(
            name="STAE Reconstruction + Future Prediction paper default",
            description=(
                "Paper-near 3D spatio-temporal autoencoder with reconstruction and one-step future prediction: "
                "input clip 16x128x128, future_length=1, shared bottleneck 64x2x16x16."
            ),
            method_type="spatiotemporal_autoencoder",
            method_config=_old_paper_stae_method_config(prediction_branch=True),
            training_config=_old_paper_stae_training_config(prediction_branch=True),
            inference_config=STAE_INFERENCE_CONFIG,
            method_graph=_old_paper_stae_graph(prediction_branch=True),
        ),
    }


def _configuration_matches_payload(configuration: models.MethodConfiguration, payload: MethodConfigurationCreate) -> bool:
    method, method_graph, method_config, training_config, inference_config, _, _ = _normalize_method_payload(payload)
    return (
        configuration.description == payload.description
        and configuration.method_type == method.type
        and configuration.method_graph == method_graph
        and configuration.method_config == method_config
        and configuration.training_config == training_config
        and configuration.inference_config == inference_config
    )


def _configuration_raw_matches_payload(configuration: models.MethodConfiguration, payload: MethodConfigurationCreate) -> bool:
    """Cheap exact match for old bundled payloads already stored in the DB.

    This avoids normalizing/validating old and new model graphs on every API
    start while still allowing one-time repair of defaults that are known to be
    unmodified copies of a previous MLTrace release.
    """
    return (
        configuration.description == payload.description
        and configuration.method_type == payload.method_type
        and configuration.method_graph == payload.method_graph
        and configuration.method_config == payload.method_config
        and configuration.training_config == payload.training_config
        and configuration.inference_config == payload.inference_config
    )


def _looks_like_old_paper_stae_default(
    configuration: models.MethodConfiguration,
    old_payload: MethodConfigurationCreate,
) -> bool:
    if configuration.method_type != "spatiotemporal_autoencoder":
        return False
    if configuration.description != old_payload.description:
        return False
    if configuration.name not in {
        "STAE Reconstruction paper default",
        "STAE Reconstruction + Future Prediction paper default",
    }:
        return False
    encoder = configuration.method_graph.get("encoder", []) if isinstance(configuration.method_graph, dict) else []
    conv_channels = [
        layer.get("config", {}).get("out_channels")
        for layer in encoder
        if isinstance(layer, dict) and layer.get("type") == "Conv3d"
    ]
    if conv_channels[:3] != [16, 32, 64]:
        return False
    if configuration.name == "STAE Reconstruction + Future Prediction paper default":
        return configuration.method_config.get("future_length") == 1
    return configuration.method_config.get("future_length") in {0, None}


def _apply_method_payload(db: Session, configuration: models.MethodConfiguration, payload: MethodConfigurationCreate) -> None:
    method, method_graph, method_config, training_config, inference_config, diagram, validation = _normalize_method_payload(payload)
    configuration.description = payload.description
    configuration.method_type = method.type
    configuration.method_family = method.method_family
    configuration.method_version = method.method_version
    configuration.training_mode = method.training_mode
    configuration.requires_training = method.requires_training
    configuration.supports_training_pipeline = method.supports_training_pipeline
    configuration.artifact_kind = method.artifact_kind
    configuration.builder_kind = method.builder_kind
    configuration.method_graph = method_graph
    configuration.method_config = method_config
    configuration.training_config = training_config
    configuration.inference_config = inference_config
    configuration.diagram = diagram
    configuration.validation = validation
    db.flush()
    _replace_method_parameter_index(db, configuration)


def ensure_default_method_configurations(db: Session) -> int:
    """Create missing built-in defaults without doing expensive startup checks.

    Existing defaults are only repaired when they are an exact raw copy of a
    known old bundled payload. We intentionally avoid normalizing historical and
    current payloads for every existing row because that validates large model
    graphs during API startup.
    """
    created = 0
    old_payloads = _old_paper_payloads_by_name()
    new_payloads = {payload.name: payload for payload in default_method_payloads()}
    payloads = default_method_payloads()
    for payload in payloads:
        existing = db.scalar(
            select(models.MethodConfiguration).where(func.lower(models.MethodConfiguration.name) == payload.name.lower())
        )
        if existing is not None:
            old_payload = old_payloads.get(payload.name)
            if old_payload is not None and _looks_like_old_paper_stae_default(existing, old_payload):
                _apply_method_payload(db, existing, new_payloads[payload.name])
                db.commit()
                logger.info("Updated bundled method configuration '%s' to current paper-near default", payload.name)
            continue
        create_method_configuration(db, payload)
        created += 1
        logger.info("Created default method configuration '%s'", payload.name)
    return created
