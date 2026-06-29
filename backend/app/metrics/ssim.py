from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SsimSettings:
    window_size: int = 11
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    k1: float = 0.01
    k2: float = 0.03
    data_range: float = 1.0

    @property
    def c1(self) -> float:
        return float((self.k1 * self.data_range) ** 2)

    @property
    def c2(self) -> float:
        return float((self.k2 * self.data_range) ** 2)

    def metadata(self) -> dict[str, float | int]:
        return {
            "ssim_window_size": self.window_size,
            "ssim_alpha": self.alpha,
            "ssim_beta": self.beta,
            "ssim_gamma": self.gamma,
            "ssim_k1": self.k1,
            "ssim_k2": self.k2,
            "ssim_data_range": self.data_range,
            "ssim_c1": self.c1,
            "ssim_c2": self.c2,
        }


def ssim_settings_from_config(config: dict[str, Any] | None) -> SsimSettings:
    config = config or {}
    window_size = int(config.get("ssim_window_size", 11) or 11)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    data_range = float(config.get("ssim_data_range", 1.0))
    return SsimSettings(
        window_size=window_size,
        alpha=float(config.get("ssim_alpha", 1.0)),
        beta=float(config.get("ssim_beta", 1.0)),
        gamma=float(config.get("ssim_gamma", 1.0)),
        k1=float(config.get("ssim_k1", 0.01)),
        k2=float(config.get("ssim_k2", 0.03)),
        data_range=data_range if data_range > 0 else 1.0,
    )


def _pow_component_np(component: np.ndarray, exponent: float) -> np.ndarray:
    if exponent == 1.0:
        return component
    return np.power(np.clip(component, 0.0, None), exponent)


def _as_hwc(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return array[..., None]
    if array.ndim == 3:
        return array
    raise ValueError(f"SSIM expects HxW or HxWxC arrays, got {array.shape}.")


def ssim_map_np(left: np.ndarray, right: np.ndarray, config: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
    if left.shape != right.shape:
        raise ValueError(f"Cannot compute SSIM for different shapes: {left.shape} vs {right.shape}.")
    settings = ssim_settings_from_config(config)
    x = _as_hwc(left)
    y = _as_hwc(right)
    kernel = (settings.window_size, settings.window_size)
    c1 = settings.c1
    c2 = settings.c2
    c3 = c2 / 2.0

    channel_maps: list[np.ndarray] = []
    for channel in range(x.shape[2]):
        xc = x[..., channel]
        yc = y[..., channel]
        mu_x = cv2.blur(xc, kernel, borderType=cv2.BORDER_REFLECT)
        mu_y = cv2.blur(yc, kernel, borderType=cv2.BORDER_REFLECT)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        var_x = np.maximum(cv2.blur(xc * xc, kernel, borderType=cv2.BORDER_REFLECT) - mu_x_sq, 0.0)
        var_y = np.maximum(cv2.blur(yc * yc, kernel, borderType=cv2.BORDER_REFLECT) - mu_y_sq, 0.0)
        cov_xy = cv2.blur(xc * yc, kernel, borderType=cv2.BORDER_REFLECT) - mu_xy
        sigma_x = np.sqrt(var_x)
        sigma_y = np.sqrt(var_y)

        luminance = (2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)
        contrast = (2.0 * sigma_x * sigma_y + c2) / (var_x + var_y + c2)
        structure = (cov_xy + c3) / (sigma_x * sigma_y + c3)
        channel_maps.append(
            _pow_component_np(luminance, settings.alpha)
            * _pow_component_np(contrast, settings.beta)
            * _pow_component_np(structure, settings.gamma)
        )
    return np.mean(np.stack(channel_maps, axis=-1), axis=-1), settings.metadata()


def ssim_distance_map_np(left: np.ndarray, right: np.ndarray, config: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
    similarity, metadata = ssim_map_np(left, right, config)
    return 1.0 - similarity, metadata


def _pow_component_torch(torch, component, exponent: float):
    if exponent == 1.0:
        return component
    return torch.clamp(component, min=0.0).pow(exponent)


def _flatten_to_nchw(torch, tensor):
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 5:
        return tensor.permute(0, 2, 1, 3, 4).reshape(tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3], tensor.shape[4])
    raise ValueError(f"SSIM expects N,C,H,W or N,C,T,H,W tensors, got {tuple(tensor.shape)}.")


def ssim_loss_torch(torch, left, right, config: dict[str, Any] | None = None):
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Cannot compute SSIM for different shapes: {tuple(left.shape)} vs {tuple(right.shape)}.")
    settings = ssim_settings_from_config(config)
    x = _flatten_to_nchw(torch, left).float()
    y = _flatten_to_nchw(torch, right).float()
    channels = int(x.shape[1])
    kernel = torch.ones((channels, 1, settings.window_size, settings.window_size), dtype=x.dtype, device=x.device)
    kernel = kernel / float(settings.window_size * settings.window_size)
    pad = settings.window_size // 2
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    y_pad = torch.nn.functional.pad(y, (pad, pad, pad, pad), mode="replicate")

    mu_x = torch.nn.functional.conv2d(x_pad, kernel, groups=channels)
    mu_y = torch.nn.functional.conv2d(y_pad, kernel, groups=channels)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    var_x = torch.clamp(
        torch.nn.functional.conv2d(x_pad * x_pad, kernel, groups=channels) - mu_x_sq,
        min=0.0,
    )
    var_y = torch.clamp(
        torch.nn.functional.conv2d(y_pad * y_pad, kernel, groups=channels) - mu_y_sq,
        min=0.0,
    )
    cov_xy = torch.nn.functional.conv2d(x_pad * y_pad, kernel, groups=channels) - mu_xy
    sigma_x = torch.sqrt(var_x + 1e-12)
    sigma_y = torch.sqrt(var_y + 1e-12)

    c1 = settings.c1
    c2 = settings.c2
    c3 = c2 / 2.0
    luminance = (2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)
    contrast = (2.0 * sigma_x * sigma_y + c2) / (var_x + var_y + c2)
    structure = (cov_xy + c3) / (sigma_x * sigma_y + c3)
    ssim_map = (
        _pow_component_torch(torch, luminance, settings.alpha)
        * _pow_component_torch(torch, contrast, settings.beta)
        * _pow_component_torch(torch, structure, settings.gamma)
    )
    return 1.0 - ssim_map.mean()
