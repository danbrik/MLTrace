"""Torch module construction and real-image forward passes for sequential methods.

This module owns the single source of truth for turning a stored method graph
into torch modules. ``validation.py`` uses it for the zero-tensor dummy check;
the training-pipeline dry run uses :func:`run_image_forward_pass` to push an
actual preprocessed image through a randomly initialized model.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np


def torch_layer_from_node(nn, layer: dict):
    """Instantiate a torch module from a stored method-graph layer node.

    Lazy modules are used where available so the layer does not need to know
    its input shape up front; the first forward pass materializes the weights.
    """
    layer_type = layer["type"]
    cfg = layer.get("config") or {}
    if layer_type == "Conv2d":
        return nn.LazyConv2d(
            out_channels=int(cfg["out_channels"]),
            kernel_size=int(cfg["kernel_size"]),
            stride=int(cfg.get("stride", 1)),
            padding=int(cfg.get("padding", 0)),
            bias=bool(cfg.get("bias", True)),
        )
    if layer_type == "ConvTranspose2d":
        lazy_conv_transpose = getattr(nn, "LazyConvTranspose2d", None)
        if lazy_conv_transpose is None:
            raise ValueError("torch.nn.LazyConvTranspose2d is unavailable.")
        return lazy_conv_transpose(
            out_channels=int(cfg["out_channels"]),
            kernel_size=int(cfg["kernel_size"]),
            stride=int(cfg.get("stride", 1)),
            padding=int(cfg.get("padding", 0)),
            output_padding=int(cfg.get("output_padding", 0)),
            bias=bool(cfg.get("bias", True)),
        )
    if layer_type == "BatchNorm2d":
        return nn.BatchNorm2d(num_features=int(cfg["num_features"]), eps=float(cfg.get("eps", 0.00001)), momentum=float(cfg.get("momentum", 0.1)))
    if layer_type == "MaxPool2d":
        return nn.MaxPool2d(kernel_size=int(cfg["kernel_size"]), stride=int(cfg.get("stride") or cfg["kernel_size"]), padding=int(cfg.get("padding", 0)))
    if layer_type == "Upsample":
        return nn.Upsample(scale_factor=float(cfg["scale_factor"]), mode=str(cfg["mode"]))
    if layer_type == "Dropout2d":
        return nn.Dropout2d(p=float(cfg.get("p", 0.1)))
    if layer_type == "Flatten":
        return nn.Flatten(start_dim=int(cfg.get("start_dim", 1)), end_dim=int(cfg.get("end_dim", -1)))
    if layer_type == "Unflatten":
        return nn.Unflatten(1, (int(cfg["channels"]), int(cfg["height"]), int(cfg["width"])))
    if layer_type == "Linear":
        return nn.LazyLinear(out_features=int(cfg["out_features"]), bias=bool(cfg.get("bias", True)))
    if layer_type == "ReLU":
        return nn.ReLU(inplace=bool(cfg.get("inplace", False)))
    if layer_type == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=float(cfg.get("negative_slope", 0.01)), inplace=bool(cfg.get("inplace", False)))
    if layer_type == "GELU":
        return nn.GELU()
    if layer_type == "Sigmoid":
        return nn.Sigmoid()
    if layer_type == "Tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown Torch layer: {layer_type}")


def build_sequential_modules(torch, method_graph: dict):
    """Build randomly initialized encoder/decoder Sequentials from a method graph."""
    nn = torch.nn
    encoder = nn.Sequential(*[torch_layer_from_node(nn, layer) for layer in method_graph.get("encoder", [])])
    decoder = nn.Sequential(*[torch_layer_from_node(nn, layer) for layer in method_graph.get("decoder", [])])
    return encoder, decoder


def forward_through_graph(torch, method_graph: dict, method_config: dict, x, logs: list[str]):
    """Run an input batch through encoder -> latent bridge -> decoder.

    For variational autoencoders the mu projection and decoder-seed projection
    are created on the fly (matching the saved torch-check behaviour); the
    stochastic sampling is skipped, so z is the deterministic mu.
    """
    nn = torch.nn
    encoder, decoder = build_sequential_modules(torch, method_graph)
    encoder.eval()
    decoder.eval()
    builder_kind = method_graph.get("builder_kind")

    with torch.no_grad():
        logs.append("Running encoder")
        encoded = encoder(x)
        if builder_kind == "sequential_variational_autoencoder":
            encoded_shape = tuple(encoded.shape[1:])
            flat = encoded.flatten(1)
            latent_dim = int(method_config["latent_dim"])
            logs.append(f"Projecting encoder output to mu/logvar latent dim {latent_dim}")
            to_mu = nn.Linear(flat.shape[1], latent_dim)
            z = to_mu(flat)
            to_seed = nn.Linear(latent_dim, flat.shape[1])
            seed = to_seed(z).reshape((x.shape[0], *encoded_shape))
        else:
            logs.append("Using encoder output as AE latent tensor")
            seed = encoded
        logs.append("Running decoder")
        output = decoder(seed)
    return output


def _image_to_nchw(image: np.ndarray, warnings: list[str], logs: list[str]) -> np.ndarray:
    """Normalize a preprocessed numpy image to a float32 NCHW batch of size 1."""
    if image.ndim == 2:
        array = image[np.newaxis, np.newaxis, :, :]
    elif image.ndim == 3:
        array = np.transpose(image, (2, 0, 1))[np.newaxis, :, :, :]
    else:
        raise ValueError(f"Preprocessed image must be 2D or 3D, got shape {tuple(image.shape)}.")

    if array.dtype == np.uint8:
        warnings.append("uint8 preprocessing output was scaled to [0, 1] for the forward pass.")
        return array.astype(np.float32) / 255.0
    array = array.astype(np.float32)
    logs.append(f"Forward input value range: [{float(array.min()):.4f}, {float(array.max()):.4f}]")
    return array


def run_image_forward_pass(method_graph: dict, method_config: dict, image: np.ndarray) -> dict:
    """Run one real preprocessed image through a randomly initialized model.

    Returns the model output as a numpy image plus shape/timing metadata.
    Raises ValueError with an actionable message on shape incompatibility or
    when torch is missing, so callers can report the problem in-band.
    """
    warnings: list[str] = []
    logs: list[str] = []
    started_at = perf_counter()

    batch = _image_to_nchw(image, warnings, logs)
    _, channels, height, width = batch.shape
    expected = (
        int(method_config["input_channels"]),
        int(method_config["input_height"]),
        int(method_config["input_width"]),
    )
    if (channels, height, width) != expected:
        raise ValueError(
            f"Preprocessing output is {channels}x{height}x{width} (CxHxW) but the method expects "
            f"{expected[0]}x{expected[1]}x{expected[2]}. Adjust the preprocessing resize/crop steps "
            "or the method's input shape."
        )

    try:
        import torch
    except Exception as exc:
        raise ValueError("Torch is not installed.") from exc

    logs.append(f"Running forward pass with random weights, input shape (1, {channels}, {height}, {width})")
    x = torch.from_numpy(batch)
    output = forward_through_graph(torch, method_graph, method_config, x, logs)
    output_shape = tuple(output.shape)
    logs.append(f"Forward pass produced output shape {output_shape}")

    output_array = output.squeeze(0).numpy()
    if output_array.shape[0] == 1:
        output_image = output_array[0]
    else:
        output_image = np.transpose(output_array, (1, 2, 0))

    return {
        "output": output_image,
        "input_shape": [1, channels, height, width],
        "output_shape": list(output_shape),
        "warnings": warnings,
        "logs": logs,
        "elapsed_ms": round((perf_counter() - started_at) * 1000, 1),
    }
