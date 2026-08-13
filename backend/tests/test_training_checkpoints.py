from pathlib import Path

import pytest

from app.training.checkpoints import atomic_torch_save, load_checkpoint


def test_atomic_checkpoint_replaces_only_after_complete_save(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    target = tmp_path / "runs" / "1" / "checkpoint.pt"
    atomic_torch_save(torch, {"version": 1, "signature": "one", "value": 1}, target, serialize=True)
    assert load_checkpoint(torch, target, "one")["value"] == 1

    original_save = torch.save

    def broken_save(payload, path):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    torch.save = broken_save
    try:
        with pytest.raises(OSError, match="disk full"):
            atomic_torch_save(torch, {"version": 1, "signature": "two"}, target, serialize=True)
    finally:
        torch.save = original_save

    assert load_checkpoint(torch, target, "one")["value"] == 1
    assert list(target.parent.glob("*.tmp")) == []


def test_checkpoint_rejects_changed_signature(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    target = tmp_path / "checkpoint.pt"
    atomic_torch_save(torch, {"version": 1, "signature": "original"}, target)
    with pytest.raises(ValueError, match="changed"):
        load_checkpoint(torch, target, "different")
