"""Opt-in checkpoint integration test; no user images or labels are transmitted."""
import os
import numpy as np
import pytest

from app.analysis.dinov3_engine import DinoEncoder, analyze_features
from app.analysis.dinov3_schemas import RepresentationConfig


@pytest.mark.skipif(os.environ.get("MLTRACE_TEST_DINOV3") != "1", reason="Opt-in public checkpoint download")
def test_real_dinov3_checkpoint(monkeypatch):
    encoder = DinoEncoder("cpu")
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, (96, 128), dtype=np.uint8) for _ in range(4)]
    features = encoder.encode(images)
    assert features.shape == (4, 384)
    assert features.dtype == np.float32 and np.isfinite(features).all()
    assert encoder.snapshot["revision"]
    assert encoder.snapshot["download_authentication"] == "anonymous"
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    cfg = RepresentationConfig(training_dataset_id=1, preprocessing_pipeline_id=1, cluster_count=2, intervals=[{
        "id": "sample", "name": "Sample", "label": "normal", "start": "2026-01-01T00:00:00", "end": "2026-01-01T00:01:00",
    }])
    pca, umap, clusters, metrics = analyze_features(features, cfg, lambda *args: None)
    assert pca.shape == umap.shape == (4, 2)
    assert len(clusters) == 4
    assert metrics["feature_dimension"] == 384
    # A populated cache must work without any network access.
    import huggingface_hub
    original_download = huggingface_hub.hf_hub_download
    def offline_download(**kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["token"] is False
        return original_download(**kwargs)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline_download)
    reloaded = DinoEncoder("cpu")
    np.testing.assert_array_equal(reloaded.encode(images), features)
