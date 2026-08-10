import numpy as np
import pytest

from app.metrics.aggregation import aggregate_score, normalize_aggregation
from app.testing.service import _score_array, _score_masked


def test_normalize_aggregation_accepts_percentiles_and_falls_back_to_mean() -> None:
    assert normalize_aggregation("p99") == "p99"
    assert normalize_aggregation("P99.9") == "p99.9"
    assert normalize_aggregation("max") == "max"
    assert normalize_aggregation(None) == "mean"
    assert normalize_aggregation("p150") == "mean"
    assert normalize_aggregation("nonsense") == "mean"


def test_aggregate_score_matches_numpy_reference() -> None:
    values = np.arange(100, dtype=np.float64)

    assert aggregate_score(values, "mean") == pytest.approx(values.mean())
    assert aggregate_score(values, "p99") == pytest.approx(np.percentile(values, 99))
    assert aggregate_score(values, "max") == pytest.approx(99.0)
    assert aggregate_score(np.array([]), "p99") == 0.0


def test_percentile_separates_a_local_blob_that_the_mean_dilutes() -> None:
    rng = np.random.default_rng(0)
    normal = np.abs(rng.normal(0.0, 0.01, size=(240, 384)))
    anomalous = normal.copy()
    anomalous[100:140, 80:120] += 0.5  # 1600 Pixel, 1.7 % der Flaeche und damit ueber der p99-Grenze

    mean_ratio = aggregate_score(anomalous, "mean") / aggregate_score(normal, "mean")
    p99_ratio = aggregate_score(anomalous, "p99") / aggregate_score(normal, "p99")

    assert p99_ratio > 5.0 * mean_ratio


def test_score_array_honours_frame_score_aggregation() -> None:
    source = np.zeros((32, 32), dtype=np.float32)
    reconstruction = source.copy()
    reconstruction[0, 0] = 1.0

    mean_score, _ = _score_array(source, reconstruction, "mae", {"frame_score_aggregation": "mean"})
    max_score, _ = _score_array(source, reconstruction, "mae", {"frame_score_aggregation": "max"})

    assert mean_score == pytest.approx(1.0 / source.size)
    assert max_score == pytest.approx(1.0)


def test_score_masked_honours_frame_score_aggregation() -> None:
    source = np.zeros((32, 32), dtype=np.float32)
    reconstruction = source.copy()
    reconstruction[0, 0] = 1.0
    mask = np.zeros(source.shape, dtype=bool)
    mask[0:4, 0:4] = True

    mean_score, _, warning = _score_masked(source, reconstruction, mask, "mae", {"frame_score_aggregation": "mean"})
    max_score, _, _ = _score_masked(source, reconstruction, mask, "mae", {"frame_score_aggregation": "max"})

    assert warning is None
    assert mean_score == pytest.approx(1.0 / 16)
    assert max_score == pytest.approx(1.0)
