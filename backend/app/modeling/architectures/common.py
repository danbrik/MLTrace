from __future__ import annotations

from app.modeling.layers import validate_layer


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
