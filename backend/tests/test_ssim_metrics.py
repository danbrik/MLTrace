import numpy as np
import pytest

from app.metrics.ssim import ssim_distance_map_np, ssim_loss_torch, ssim_settings_from_config
from app.schemas import HeatmapVisualizationConfig
from app.testing.service import _pixel_error_map
from app.training.engine import _loss_fn


def test_ssim_settings_use_standard_k_constants_for_c_values() -> None:
    settings = ssim_settings_from_config({"ssim_k1": 0.01, "ssim_k2": 0.03, "ssim_data_range": 1.0})

    assert settings.c1 == pytest.approx(0.0001)
    assert settings.c2 == pytest.approx(0.0009)
    assert settings.metadata()["ssim_c1"] == pytest.approx((0.01 * 1.0) ** 2)
    assert settings.metadata()["ssim_c2"] == pytest.approx((0.03 * 1.0) ** 2)


def test_ssim_distance_is_near_zero_for_identical_images_and_higher_for_changed_images() -> None:
    image = np.zeros((24, 24), dtype=np.float32)
    changed = image.copy()
    changed[8:16, 8:16] = 1.0

    identical_distance, metadata = ssim_distance_map_np(image, image)
    changed_distance, _ = ssim_distance_map_np(image, changed)

    assert metadata["ssim_k1"] == pytest.approx(0.01)
    assert float(np.mean(identical_distance)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.mean(changed_distance)) > float(np.mean(identical_distance))


def test_heatmap_can_use_ssim_residual_source() -> None:
    source = np.zeros((24, 24), dtype=np.float32)
    reconstruction = source.copy()
    reconstruction[8:16, 8:16] = 1.0

    error = _pixel_error_map(
        source,
        reconstruction,
        HeatmapVisualizationConfig(residual_source="ssim_residual", ssim_window_size=11),
    )

    assert error.shape == source.shape
    assert float(np.max(error)) > 0.0


def test_torch_ssim_and_combined_losses_are_available() -> None:
    torch = pytest.importorskip("torch")
    source = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
    same = source.clone()
    changed = source.clone()
    changed[:, :, 8:16, 8:16] = 1.0

    assert float(ssim_loss_torch(torch, source, same)) == pytest.approx(0.0, abs=1e-6)
    assert float(ssim_loss_torch(torch, source, changed)) > 0.0

    combined = _loss_fn(torch, "mse_ssim", {"ssim_weight": 0.5})
    assert float(combined(source, changed)) > 0.0
