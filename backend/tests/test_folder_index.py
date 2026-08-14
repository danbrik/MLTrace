from datetime import datetime, timedelta
from pathlib import Path

from app import models
from app.training import data as data_module

from tests.test_testing_service import make_db, write_tiff


def _seed_folder(db, root: Path, count: int = 5):
    for index in range(count):
        ts = datetime(2026, 4, 1, 12, 0, 0) + timedelta(seconds=index * 10)
        write_tiff(root / f"frame_{ts:%Y%m%d_%H%M%S}.tiff", 100, size=(8, 6))
    dataset = models.Dataset(
        name="Root",
        root_path=str(root),
        status="ready",
        timestamp_regex=r"(?P<timestamp>\d{8}_\d{6})",
        timestamp_format="%Y%m%d_%H%M%S",
    )
    db.add(dataset)
    db.flush()
    folder = models.DatasetFolder(dataset_id=dataset.id, relative_path=".", image_count=count)
    db.add(folder)
    db.commit()
    return folder


def test_folder_index_caches_and_invalidates(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        root = tmp_path / "images"
        folder = _seed_folder(db, root, count=5)
        monkeypatch.setattr(data_module, "data_dir", lambda: tmp_path / "data")

        # Count how often filenames get re-parsed.
        calls = {"n": 0}
        real_extract = data_module.extract_timestamp
        real_direct_tiff_files = data_module.direct_tiff_files

        def counting_direct_tiff_files(path):
            calls["scans"] = calls.get("scans", 0) + 1
            return real_direct_tiff_files(path)

        def counting_extract(name, regex, fmt):
            calls["n"] += 1
            return real_extract(name, regex, fmt)

        monkeypatch.setattr(data_module, "direct_tiff_files", counting_direct_tiff_files)
        monkeypatch.setattr(data_module, "extract_timestamp", counting_extract)

        first = data_module._folder_timestamp_index(folder)
        assert len(first) == 5
        assert calls["scans"] == 1
        assert calls["n"] == 5  # parsed every file once

        # Second call loads from disk without listing or re-parsing the folder.
        calls["n"] = 0
        second = data_module._folder_timestamp_index(folder)
        assert calls["scans"] == 1
        assert calls["n"] == 0
        assert [(ts, name) for ts, name, _ in second] == [(ts, name) for ts, name, _ in first]
        # Absolute paths are reconstructed correctly.
        assert all(Path(path).exists() for _, _, path in second)

        # Adding a file changes the signature → rebuild (re-parse).
        new_ts = datetime(2026, 4, 1, 12, 0, 50)
        write_tiff(root / f"frame_{new_ts:%Y%m%d_%H%M%S}.tiff", 100, size=(8, 6))
        calls["n"] = 0
        third = data_module._folder_timestamp_index(folder)
        assert len(third) == 6
        assert calls["scans"] == 2
        assert calls["n"] == 6  # re-parsed after invalidation

        # A malformed cache falls back to one clean rebuild.
        data_module._folder_index_cache_path(folder.id).write_text("not json", encoding="utf-8")
        calls["n"] = 0
        fourth = data_module._folder_timestamp_index(folder)
        assert len(fourth) == 6
        assert calls["scans"] == 3
        assert calls["n"] == 6
    finally:
        db.close()
