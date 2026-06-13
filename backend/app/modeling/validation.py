from __future__ import annotations

from dataclasses import dataclass
from math import floor
from time import perf_counter
from typing import Any

from app.modeling.forward import forward_through_graph


@dataclass(frozen=True)
class TensorSpec:
    rank: int
    channels: int | None = None
    height: int | None = None
    width: int | None = None
    features: int | None = None

    def label(self) -> str:
        if self.rank == 4:
            return f"N,{self.channels},{self.height},{self.width}"
        if self.rank == 2:
            return f"N,{self.features}"
        return f"rank {self.rank}"


@dataclass(frozen=True)
class LayerShapeSpec:
    section: str
    index: int
    layer_id: str | None
    layer_type: str
    input: TensorSpec
    output: TensorSpec

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "index": self.index,
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "input": self.input.__dict__,
            "output": self.output.__dict__,
            "input_label": self.input.label(),
            "output_label": self.output.label(),
        }


def validate_cnn_tensor_contract(method_graph: dict, method_config: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    layer_specs: list[LayerShapeSpec] = []
    builder_kind = method_graph.get("builder_kind")

    input_spec = TensorSpec(
        rank=4,
        channels=int(method_config["input_channels"]),
        height=int(method_config["input_height"]),
        width=int(method_config["input_width"]),
    )
    current = input_spec

    for index, layer in enumerate(method_graph.get("encoder", []), start=1):
        current = _infer_layer(layer, current, "encoder", index, errors, layer_specs)

    encoder_output = current
    if encoder_output.rank == 4:
        if encoder_output.channels is None or encoder_output.height is None or encoder_output.width is None:
            errors.append("Encoder output has unknown spatial shape.")
            flattened_features = None
        else:
            flattened_features = encoder_output.channels * encoder_output.height * encoder_output.width
    elif encoder_output.rank == 2:
        flattened_features = encoder_output.features
    else:
        flattened_features = None
        errors.append(f"Encoder output rank {encoder_output.rank} cannot be bridged to latent space.")

    latent_dim = int(method_config["latent_dim"])
    if flattened_features is not None and flattened_features <= 0:
        errors.append("Encoder flattened feature count must be positive.")
    if latent_dim <= 0:
        errors.append("latent_dim must be positive.")
    if (
        builder_kind == "sequential_autoencoder"
        and flattened_features is not None
        and flattened_features > 0
        and latent_dim > 0
        and flattened_features != latent_dim
    ):
        errors.append(
            "CNN Autoencoder latent_dim must match the final encoder output feature count. "
            f"Final encoder output is {encoder_output.label()} ({flattened_features} flattened features), "
            f"but latent_dim is {latent_dim}. Change latent_dim to {flattened_features} "
            f"or change the final encoder layer to output {latent_dim} features."
        )

    decoder_seed = _decoder_seed_from_encoder(encoder_output, latent_dim, errors)
    current = decoder_seed
    for index, layer in enumerate(method_graph.get("decoder", []), start=1):
        current = _infer_layer(layer, current, "decoder", index, errors, layer_specs)

    if current != input_spec:
        errors.append(
            "Decoder output must match input shape "
            f"{input_spec.label()}, got {current.label()}."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "layer_specs": [item.as_dict() for item in layer_specs],
        "torch_check": None,
    }


def _decoder_seed_from_encoder(encoder_output: TensorSpec, latent_dim: int, errors: list[str]) -> TensorSpec:
    if encoder_output.rank == 4:
        return encoder_output
    if encoder_output.rank == 2:
        if encoder_output.features is None or encoder_output.features <= 0:
            errors.append("Decoder seed cannot be inferred from encoder feature output.")
        return TensorSpec(rank=2, features=encoder_output.features or latent_dim)
    errors.append("Decoder seed cannot be inferred.")
    return TensorSpec(rank=2, features=latent_dim)


def _infer_layer(
    layer: dict,
    input_spec: TensorSpec,
    section: str,
    index: int,
    errors: list[str],
    layer_specs: list[LayerShapeSpec],
) -> TensorSpec:
    layer_type = layer.get("type")
    cfg = layer.get("config") or {}
    try:
        output = _infer_layer_output(layer_type, cfg, input_spec)
    except ValueError as exc:
        errors.append(f"{section}[{index}] {layer_type}: {exc}")
        output = input_spec
    if output.rank == 4 and (output.height is None or output.width is None or output.height <= 0 or output.width <= 0):
        errors.append(f"{section}[{index}] {layer_type}: output spatial dimensions must be positive.")
    if output.rank == 2 and (output.features is None or output.features <= 0):
        errors.append(f"{section}[{index}] {layer_type}: output feature count must be positive.")
    layer_specs.append(
        LayerShapeSpec(
            section=section,
            index=index,
            layer_id=layer.get("id"),
            layer_type=str(layer_type),
            input=input_spec,
            output=output,
        )
    )
    return output


def _infer_layer_output(layer_type: str, cfg: dict, spec: TensorSpec) -> TensorSpec:
    if layer_type in {"ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"}:
        return spec
    if layer_type == "Dropout2d":
        _require_rank(spec, 4, layer_type)
        return spec
    if layer_type == "Conv2d":
        _require_rank(spec, 4, layer_type)
        kernel = int(cfg["kernel_size"])
        stride = int(cfg.get("stride", 1))
        padding = int(cfg.get("padding", 0))
        return TensorSpec(
            rank=4,
            channels=int(cfg["out_channels"]),
            height=_conv2d_out(spec.height, kernel, stride, padding),
            width=_conv2d_out(spec.width, kernel, stride, padding),
        )
    if layer_type == "ConvTranspose2d":
        _require_rank(spec, 4, layer_type)
        kernel = int(cfg["kernel_size"])
        stride = int(cfg.get("stride", 1))
        padding = int(cfg.get("padding", 0))
        output_padding = int(cfg.get("output_padding", 0))
        if output_padding >= stride:
            raise ValueError("output_padding must be smaller than stride.")
        return TensorSpec(
            rank=4,
            channels=int(cfg["out_channels"]),
            height=_conv_transpose2d_out(spec.height, kernel, stride, padding, output_padding),
            width=_conv_transpose2d_out(spec.width, kernel, stride, padding, output_padding),
        )
    if layer_type == "BatchNorm2d":
        _require_rank(spec, 4, layer_type)
        num_features = int(cfg["num_features"])
        if spec.channels is not None and num_features != spec.channels:
            raise ValueError(f"num_features must match input channels {spec.channels}, got {num_features}.")
        return spec
    if layer_type == "MaxPool2d":
        _require_rank(spec, 4, layer_type)
        kernel = int(cfg["kernel_size"])
        stride = int(cfg.get("stride") or kernel)
        padding = int(cfg.get("padding", 0))
        return TensorSpec(
            rank=4,
            channels=spec.channels,
            height=_conv2d_out(spec.height, kernel, stride, padding),
            width=_conv2d_out(spec.width, kernel, stride, padding),
        )
    if layer_type == "Upsample":
        _require_rank(spec, 4, layer_type)
        scale = float(cfg["scale_factor"])
        return TensorSpec(
            rank=4,
            channels=spec.channels,
            height=floor((spec.height or 0) * scale),
            width=floor((spec.width or 0) * scale),
        )
    if layer_type == "Flatten":
        if spec.rank == 2:
            return spec
        _require_rank(spec, 4, layer_type)
        return TensorSpec(rank=2, features=(spec.channels or 0) * (spec.height or 0) * (spec.width or 0))
    if layer_type == "Unflatten":
        _require_rank(spec, 2, layer_type)
        channels = int(cfg["channels"])
        height = int(cfg["height"])
        width = int(cfg["width"])
        expected = channels * height * width
        if spec.features is not None and spec.features != expected:
            raise ValueError(f"features must equal channels*height*width ({expected}), got {spec.features}.")
        return TensorSpec(rank=4, channels=channels, height=height, width=width)
    if layer_type == "Linear":
        _require_rank(spec, 2, layer_type)
        return TensorSpec(rank=2, features=int(cfg["out_features"]))
    raise ValueError(f"Unknown layer type: {layer_type}")


def _require_rank(spec: TensorSpec, rank: int, layer_type: str) -> None:
    if spec.rank != rank:
        raise ValueError(f"{layer_type} requires rank {rank} input, got {spec.label()}.")


def _conv2d_out(size: int | None, kernel: int, stride: int, padding: int) -> int:
    return floor(((size or 0) + 2 * padding - (kernel - 1) - 1) / stride + 1)


def _conv_transpose2d_out(size: int | None, kernel: int, stride: int, padding: int, output_padding: int) -> int:
    return ((size or 0) - 1) * stride - 2 * padding + kernel + output_padding


def run_cnn_torch_dummy_forward(method_graph: dict, method_config: dict) -> dict:
    logs: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    started_at = perf_counter()

    try:
        input_spec = TensorSpec(
            rank=4,
            channels=int(method_config["input_channels"]),
            height=int(method_config["input_height"]),
            width=int(method_config["input_width"]),
        )
    except Exception as exc:
        message = f"Invalid input shape config: {exc}"
        return {
            "valid": False,
            "status": "failed",
            "errors": [message],
            "warnings": warnings,
            "logs": [message, "Failed"],
            "torch_check": {"status": "failed", "message": message},
        }

    logs.append("Building encoder")
    try:
        import torch
    except Exception:
        message = "Torch is not installed."
        logs.append(message)
        logs.append("Failed")
        return {
            "valid": False,
            "status": "missing",
            "errors": [],
            "warnings": [message],
            "logs": logs,
            "torch_check": {"status": "missing", "message": message},
        }

    try:
        torch_check = _run_torch_check(method_graph, method_config, input_spec, torch, logs, started_at)
        return {
            "valid": True,
            "status": "available",
            "errors": errors,
            "warnings": warnings,
            "logs": logs,
            "torch_check": torch_check,
        }
    except Exception as exc:
        message = str(exc)
        logs.append(f"Failed: {message}")
        return {
            "valid": False,
            "status": "failed",
            "errors": [message],
            "warnings": warnings,
            "logs": logs,
            "torch_check": {
                "status": "failed",
                "message": message,
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 1),
            },
        }


def _run_torch_check(
    method_graph: dict,
    method_config: dict,
    input_spec: TensorSpec,
    torch,
    logs: list[str],
    started_at: float,
) -> dict:
    expected_shape = (1, input_spec.channels, input_spec.height, input_spec.width)
    logs.append(f"Running dummy input {expected_shape}")
    x = torch.zeros(expected_shape, dtype=torch.float32)
    output = forward_through_graph(torch, method_graph, method_config, x, logs)
    output_shape = tuple(output.shape)
    logs.append("Checking decoder output")
    if output_shape != expected_shape:
        raise ValueError(f"dummy forward produced {output_shape}, expected {expected_shape}")

    logs.append("Passed")
    return {
        "status": "available",
        "message": "Torch dummy-forward validation passed.",
        "input_shape": list(expected_shape),
        "output_shape": list(output_shape),
        "elapsed_ms": round((perf_counter() - started_at) * 1000, 1),
    }


