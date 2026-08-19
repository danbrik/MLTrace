"""Stable content signatures for trained model artifacts.

Evaluation may combine score series produced by several inference runs.  A
training run row is reused when training is restarted, so its numeric id alone
does not identify the exact model generation.  These helpers provide a small,
streaming SHA-256 fingerprint that can be snapshotted onto every inference run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


_SIGNATURE_VERSION = b"mltrace-artifact-signature-v1\0"
_CHUNK_SIZE = 1024 * 1024


def artifact_signature(path_value: str | Path | None) -> str | None:
    """Return a deterministic SHA-256 for a file or directory artifact.

    Directory entries are ordered by their POSIX relative path and both the
    path and file bytes participate in the digest.  Missing paths deliberately
    return ``None`` so callers can preserve legacy/unverifiable state instead
    of manufacturing a misleading identity.
    """

    if not path_value:
        return None
    root = Path(path_value)
    if not root.exists():
        return None

    digest = hashlib.sha256()
    digest.update(_SIGNATURE_VERSION)
    if root.is_file():
        digest.update(b"file\0")
        digest.update(root.name.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        _update_file(digest, root)
        return digest.hexdigest()

    digest.update(b"directory\0")
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        _update_file(digest, path)
        digest.update(b"\0")
    return digest.hexdigest()


def _update_file(digest: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
