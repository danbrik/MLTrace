from pathlib import Path

from app.artifact_signatures import artifact_signature


def test_artifact_signature_is_content_sensitive_and_stable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"model-v1")

    first = artifact_signature(artifact)
    assert first is not None
    assert artifact_signature(artifact) == first

    artifact.write_bytes(b"model-v2")
    assert artifact_signature(artifact) != first


def test_directory_signature_uses_sorted_relative_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "bundle"
    artifact.mkdir()
    (artifact / "b.bin").write_bytes(b"b")
    (artifact / "a.bin").write_bytes(b"a")
    first = artifact_signature(artifact)

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "a.bin").write_bytes(b"a")
    (replacement / "b.bin").write_bytes(b"b")

    assert artifact_signature(replacement) == first


def test_missing_artifact_has_no_signature(tmp_path: Path) -> None:
    assert artifact_signature(None) is None
    assert artifact_signature(tmp_path / "missing.pt") is None
