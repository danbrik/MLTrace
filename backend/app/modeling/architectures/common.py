from __future__ import annotations

from app.modeling.layers import validate_layer


def _validate_fast_anogan_block(block: dict, path: str, *, expected_direction: str, critic: bool = False) -> dict:
    if not isinstance(block, dict):
        raise ValueError(f"{path} must be an object.")
    out_channels = block.get("out_channels")
    if not isinstance(out_channels, int) or out_channels <= 0:
        raise ValueError(f"{path}.out_channels must be a positive integer.")
    block_type = str(block.get("block_type", "residual"))
    if block_type != "residual":
        raise ValueError(f"{path}.block_type must be 'residual'.")
    direction = str(block.get("direction", expected_direction))
    if direction != expected_direction:
        raise ValueError(f"{path}.direction must be '{expected_direction}'.")
    normalization = str(block.get("normalization", "layer_norm" if critic else "none"))
    if critic and normalization == "batch_norm":
        raise ValueError(f"{path}.normalization cannot be batch_norm for WGAN-GP critic blocks.")
    if normalization not in {"none", "layer_norm"}:
        raise ValueError(f"{path}.normalization must be 'none' or 'layer_norm'.")
    return {
        "id": str(block.get("id") or path.split(".")[-1]),
        "block_type": block_type,
        "direction": direction,
        "out_channels": int(out_channels),
        "normalization": normalization,
    }


def validate_fast_anogan_model_graph(model_graph: dict | None, expected_builder_kind: str) -> dict:
    if not isinstance(model_graph, dict):
        raise ValueError("model_graph must be an object.")
    builder_kind = model_graph.get("builder_kind")
    if builder_kind is not None and builder_kind != expected_builder_kind:
        raise ValueError(f"model_graph.builder_kind must be {expected_builder_kind}.")

    generator_blocks = model_graph.get("generator_blocks")
    critic_blocks = model_graph.get("critic_blocks")
    encoder_blocks = model_graph.get("encoder_blocks")
    if not isinstance(generator_blocks, list) or len(generator_blocks) == 0:
        raise ValueError("fastAnoGAN requires at least one generator block.")
    if not isinstance(critic_blocks, list) or len(critic_blocks) == 0:
        raise ValueError("fastAnoGAN requires at least one critic block.")
    if not isinstance(encoder_blocks, list) or len(encoder_blocks) == 0:
        raise ValueError("fastAnoGAN requires at least one encoder block.")

    normalized_generator = [
        _validate_fast_anogan_block(block, f"model_graph.generator_blocks[{index}]", expected_direction="up")
        for index, block in enumerate(generator_blocks)
    ]
    normalized_critic = [
        _validate_fast_anogan_block(block, f"model_graph.critic_blocks[{index}]", expected_direction="down", critic=True)
        for index, block in enumerate(critic_blocks)
    ]
    normalized_encoder = [
        _validate_fast_anogan_block(block, f"model_graph.encoder_blocks[{index}]", expected_direction="down")
        for index, block in enumerate(encoder_blocks)
    ]

    feature_layer = str(model_graph.get("feature_layer") or "critic_blocks")
    if feature_layer != "critic_blocks":
        raise ValueError("fastAnoGAN currently supports feature_layer='critic_blocks'.")

    return {
        "builder_kind": expected_builder_kind,
        "generator_blocks": normalized_generator,
        "critic_blocks": normalized_critic,
        "encoder_blocks": normalized_encoder,
        "feature_layer": feature_layer,
    }


def validate_sequential_model_graph(model_graph: dict | None, expected_builder_kind: str) -> dict:
    if not isinstance(model_graph, dict):
        raise ValueError("model_graph must be an object.")
    builder_kind = model_graph.get("builder_kind")
    if builder_kind is not None and builder_kind != expected_builder_kind:
        raise ValueError(f"model_graph.builder_kind must be {expected_builder_kind}.")

    encoder = model_graph.get("encoder")
    decoder = model_graph.get("decoder")
    if not isinstance(encoder, list) or len(encoder) == 0:
        raise ValueError("CNN architectures require at least one encoder layer.")
    if not isinstance(decoder, list) or len(decoder) == 0:
        raise ValueError("CNN architectures require at least one decoder layer.")

    normalized_encoder = [
        validate_layer(layer, f"model_graph.encoder[{index}]") for index, layer in enumerate(encoder)
    ]
    normalized_decoder = [
        validate_layer(layer, f"model_graph.decoder[{index}]") for index, layer in enumerate(decoder)
    ]
    latent = model_graph.get("latent") or {}
    if not isinstance(latent, dict):
        raise ValueError("model_graph.latent must be an object.")

    return {
        "builder_kind": expected_builder_kind,
        "encoder": normalized_encoder,
        "latent": latent,
        "decoder": normalized_decoder,
    }


def validate_spatiotemporal_model_graph(
    model_graph: dict | None,
    expected_builder_kind: str,
    method_config: dict,
) -> dict:
    if not isinstance(model_graph, dict):
        raise ValueError("model_graph must be an object.")
    builder_kind = model_graph.get("builder_kind")
    if builder_kind is not None and builder_kind != expected_builder_kind:
        raise ValueError(f"model_graph.builder_kind must be {expected_builder_kind}.")

    encoder = model_graph.get("encoder")
    decoder = model_graph.get("decoder")
    if not isinstance(encoder, list) or len(encoder) == 0:
        raise ValueError("SpatioTemporal AE requires at least one encoder layer.")
    if not isinstance(decoder, list) or len(decoder) == 0:
        raise ValueError("SpatioTemporal AE requires at least one reconstruction decoder layer.")

    normalized_encoder = [
        validate_layer(layer, f"model_graph.encoder[{index}]") for index, layer in enumerate(encoder)
    ]
    normalized_decoder = [
        validate_layer(layer, f"model_graph.decoder[{index}]") for index, layer in enumerate(decoder)
    ]
    prediction_decoder = model_graph.get("prediction_decoder") or []
    if prediction_decoder and not isinstance(prediction_decoder, list):
        raise ValueError("model_graph.prediction_decoder must be a list.")
    normalized_prediction_decoder = [
        validate_layer(layer, f"model_graph.prediction_decoder[{index}]")
        for index, layer in enumerate(prediction_decoder)
    ]
    if method_config.get("prediction_branch") and not normalized_prediction_decoder:
        raise ValueError("Prediction branch is enabled, but prediction_decoder is empty.")

    latent = model_graph.get("latent") or {}
    if not isinstance(latent, dict):
        raise ValueError("model_graph.latent must be an object.")

    return {
        "builder_kind": expected_builder_kind,
        "encoder": normalized_encoder,
        "latent": latent,
        "decoder": normalized_decoder,
        "prediction_decoder": normalized_prediction_decoder,
    }
