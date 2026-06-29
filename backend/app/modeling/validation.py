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
    frames: int | None = None
    height: int | None = None
    width: int | None = None
    features: int | None = None

    def label(self) -> str:
        if self.rank == 5:
            return f"N,{self.channels},{self.frames},{self.height},{self.width}"
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
    if method_graph.get("builder_kind") == "fast_anogan":
        return validate_fast_anogan_tensor_contract(method_graph, method_config)
    if method_graph.get("builder_kind") == "spatiotemporal_autoencoder":
        return validate_spatiotemporal_tensor_contract(method_graph, method_config)

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

    latent_dim = int(method_config.get("latent_dim") or 0)
    if flattened_features is not None and flattened_features <= 0:
        errors.append("Encoder flattened feature count must be positive.")
    if builder_kind != "sequential_spatial_autoencoder" and latent_dim <= 0:
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
    if builder_kind == "sequential_spatial_autoencoder":
        bottleneck_channels = int(method_config.get("bottleneck_channels") or 0)
        if bottleneck_channels <= 0:
            errors.append("bottleneck_channels must be positive.")
        if encoder_output.rank != 4:
            errors.append(
                "AESpatial encoder output must remain spatial rank 4 "
                f"(N,C,H,W), got {encoder_output.label()}."
            )
        elif encoder_output.channels != bottleneck_channels:
            errors.append(
                "AESpatial bottleneck_channels must match the final encoder output channels. "
                f"Final encoder output is {encoder_output.label()}, but bottleneck_channels is {bottleneck_channels}."
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


def validate_fast_anogan_tensor_contract(method_graph: dict, method_config: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    layer_specs: list[LayerShapeSpec] = []
    input_channels = int(method_config["input_channels"])
    input_height = int(method_config["input_height"])
    input_width = int(method_config["input_width"])
    latent_dim = int(method_config["latent_dim"])
    seed_size = int(method_config.get("generator_seed_size") or 4)

    generator_blocks = method_graph.get("generator_blocks", [])
    critic_blocks = method_graph.get("critic_blocks", [])
    encoder_blocks = method_graph.get("encoder_blocks", [])
    if not generator_blocks or not critic_blocks or not encoder_blocks:
        errors.append("fastAnoGAN requires generator, critic, and encoder blocks.")

    current = TensorSpec(rank=2, features=latent_dim)
    generator_seed_channels = int(generator_blocks[0]["out_channels"]) if generator_blocks else 1
    seed = TensorSpec(rank=4, channels=generator_seed_channels, height=seed_size, width=seed_size)
    layer_specs.append(
        LayerShapeSpec("generator", 0, "generator-seed", "Linear+Unflatten", current, seed)
    )
    current = seed
    for index, block in enumerate(generator_blocks, start=1):
        output = TensorSpec(
            rank=4,
            channels=int(block["out_channels"]),
            height=(current.height or 0) * 2,
            width=(current.width or 0) * 2,
        )
        layer_specs.append(
            LayerShapeSpec("generator", index, block.get("id"), "ResidualUpBlock", current, output)
        )
        current = output
    generator_output = TensorSpec(rank=4, channels=input_channels, height=current.height, width=current.width)
    layer_specs.append(
        LayerShapeSpec("generator", len(generator_blocks) + 1, "generator-output", "Conv2d", current, generator_output)
    )
    if generator_output.channels != input_channels or generator_output.height != input_height or generator_output.width != input_width:
        errors.append(
            "Generator output must match input shape "
            f"N,{input_channels},{input_height},{input_width}, got {generator_output.label()}."
        )

    image_spec = TensorSpec(rank=4, channels=input_channels, height=input_height, width=input_width)
    current = image_spec
    for index, block in enumerate(critic_blocks, start=1):
        normalization = block.get("normalization")
        if normalization == "batch_norm":
            errors.append(f"critic[{index}] uses batch_norm, which is not allowed for WGAN-GP.")
        output = TensorSpec(
            rank=4,
            channels=int(block["out_channels"]),
            height=_conv2d_out(current.height, 3, 2, 1),
            width=_conv2d_out(current.width, 3, 2, 1),
        )
        layer_specs.append(
            LayerShapeSpec("critic", index, block.get("id"), "ResidualDownBlock", current, output)
        )
        current = output
    if current.height is None or current.width is None or current.height <= 0 or current.width <= 0:
        errors.append("Critic output spatial dimensions must be positive.")

    current = image_spec
    for index, block in enumerate(encoder_blocks, start=1):
        output = TensorSpec(
            rank=4,
            channels=int(block["out_channels"]),
            height=_conv2d_out(current.height, 3, 2, 1),
            width=_conv2d_out(current.width, 3, 2, 1),
        )
        layer_specs.append(
            LayerShapeSpec("encoder", index, block.get("id"), "ResidualDownBlock", current, output)
        )
        current = output
    encoder_output = TensorSpec(rank=2, features=latent_dim)
    layer_specs.append(
        LayerShapeSpec("encoder", len(encoder_blocks) + 1, "encoder-latent", "Flatten+Linear", current, encoder_output)
    )
    if current.height is None or current.width is None or current.height <= 0 or current.width <= 0:
        errors.append("Encoder output spatial dimensions must be positive before latent projection.")

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
    if output.rank == 5 and (
        output.frames is None
        or output.height is None
        or output.width is None
        or output.frames <= 0
        or output.height <= 0
        or output.width <= 0
    ):
        errors.append(f"{section}[{index}] {layer_type}: output temporal/spatial dimensions must be positive.")
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
    if layer_type == "Dropout3d":
        _require_rank(spec, 5, layer_type)
        return spec
    if layer_type == "Conv3d":
        _require_rank(spec, 5, layer_type)
        kernel_t, kernel_xy = _kernel_t_xy(cfg)
        stride_t, stride_xy = _stride_t_xy(cfg)
        padding_t, padding_xy = _padding_t_xy(cfg)
        return TensorSpec(
            rank=5,
            channels=int(cfg["out_channels"]),
            frames=_conv2d_out(spec.frames, kernel_t, stride_t, padding_t),
            height=_conv2d_out(spec.height, kernel_xy, stride_xy, padding_xy),
            width=_conv2d_out(spec.width, kernel_xy, stride_xy, padding_xy),
        )
    if layer_type == "ConvTranspose3d":
        _require_rank(spec, 5, layer_type)
        kernel_t, kernel_xy = _kernel_t_xy(cfg)
        stride_t, stride_xy = _stride_t_xy(cfg)
        padding_t, padding_xy = _padding_t_xy(cfg)
        output_padding_t, output_padding_xy = _output_padding_t_xy(cfg)
        if output_padding_t >= stride_t or output_padding_xy >= stride_xy:
            raise ValueError("output_padding must be smaller than stride.")
        return TensorSpec(
            rank=5,
            channels=int(cfg["out_channels"]),
            frames=_conv_transpose2d_out(spec.frames, kernel_t, stride_t, padding_t, output_padding_t),
            height=_conv_transpose2d_out(spec.height, kernel_xy, stride_xy, padding_xy, output_padding_xy),
            width=_conv_transpose2d_out(spec.width, kernel_xy, stride_xy, padding_xy, output_padding_xy),
        )
    if layer_type == "BatchNorm3d":
        _require_rank(spec, 5, layer_type)
        num_features = int(cfg["num_features"])
        if spec.channels is not None and num_features != spec.channels:
            raise ValueError(f"num_features must match input channels {spec.channels}, got {num_features}.")
        return spec
    if layer_type == "MaxPool3d":
        _require_rank(spec, 5, layer_type)
        kernel = int(cfg["kernel_size"])
        stride = int(cfg.get("stride") or kernel)
        padding = int(cfg.get("padding", 0))
        return TensorSpec(
            rank=5,
            channels=spec.channels,
            frames=_conv2d_out(spec.frames, kernel, stride, padding),
            height=_conv2d_out(spec.height, kernel, stride, padding),
            width=_conv2d_out(spec.width, kernel, stride, padding),
        )
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


def _kernel_t_xy(cfg: dict) -> tuple[int, int]:
    base = int(cfg.get("kernel_size", 1))
    return int(cfg.get("kernel_size_t", base)), int(cfg.get("kernel_size_xy", base))


def _stride_t_xy(cfg: dict) -> tuple[int, int]:
    base = int(cfg.get("stride", 1))
    return int(cfg.get("stride_t", base)), int(cfg.get("stride_xy", base))


def _padding_t_xy(cfg: dict) -> tuple[int, int]:
    base = int(cfg.get("padding", 0))
    return int(cfg.get("padding_t", base)), int(cfg.get("padding_xy", base))


def _output_padding_t_xy(cfg: dict) -> tuple[int, int]:
    base = int(cfg.get("output_padding", 0))
    return int(cfg.get("output_padding_t", base)), int(cfg.get("output_padding_xy", base))


def validate_spatiotemporal_tensor_contract(method_graph: dict, method_config: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    layer_specs: list[LayerShapeSpec] = []
    clip_length = int(method_config.get("clip_length") or 0)
    future_length = int(method_config.get("future_length") or 0)
    prediction_branch = bool(method_config.get("prediction_branch"))
    input_spec = TensorSpec(
        rank=5,
        channels=int(method_config["input_channels"]),
        frames=clip_length,
        height=int(method_config["input_height"]),
        width=int(method_config["input_width"]),
    )
    current = input_spec
    for index, layer in enumerate(method_graph.get("encoder", []), start=1):
        current = _infer_layer(layer, current, "encoder", index, errors, layer_specs)
    if current.rank != 5:
        errors.append(f"STAE encoder output must stay rank 5, got {current.label()}.")

    recon = current
    for index, layer in enumerate(method_graph.get("decoder", []), start=1):
        recon = _infer_layer(layer, recon, "decoder", index, errors, layer_specs)
    if recon != input_spec:
        errors.append(f"Reconstruction decoder output must match {input_spec.label()}, got {recon.label()}.")

    if prediction_branch:
        if future_length <= 0:
            errors.append("future_length must be positive when prediction_branch is enabled.")
        prediction = current
        for index, layer in enumerate(method_graph.get("prediction_decoder", []), start=1):
            prediction = _infer_layer(layer, prediction, "prediction_decoder", index, errors, layer_specs)
        expected_prediction = TensorSpec(
            rank=5,
            channels=input_spec.channels,
            frames=future_length,
            height=input_spec.height,
            width=input_spec.width,
        )
        if prediction != expected_prediction:
            errors.append(
                "Prediction decoder output must match future target shape "
                f"{expected_prediction.label()}, got {prediction.label()}."
            )
    elif method_graph.get("prediction_decoder"):
        warnings.append("prediction_decoder is stored but prediction_branch is disabled.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "layer_specs": [item.as_dict() for item in layer_specs],
        "torch_check": None,
    }


def run_cnn_torch_dummy_forward(method_graph: dict, method_config: dict) -> dict:
    logs: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    started_at = perf_counter()

    try:
        if method_graph.get("builder_kind") == "spatiotemporal_autoencoder":
            input_spec = TensorSpec(
                rank=5,
                channels=int(method_config["input_channels"]),
                frames=int(method_config["clip_length"]),
                height=int(method_config["input_height"]),
                width=int(method_config["input_width"]),
            )
        else:
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
    expected_shape = (
        (1, input_spec.channels, input_spec.frames, input_spec.height, input_spec.width)
        if input_spec.rank == 5
        else (1, input_spec.channels, input_spec.height, input_spec.width)
    )
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
