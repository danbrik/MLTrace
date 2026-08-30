from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import threading
import tracemalloc

import numpy as np
import pytest

from app.analysis import image_distribution_runtime as runtime


GRAPH = {
    "nodes": [{"id": "load", "type": "load_image", "config": {"mode": "unchanged", "dtype": "source"}}],
    "edges": [],
}


class _Compiled:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    def run(self, _path: str) -> np.ndarray:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("temporary NAS error")
        return np.asarray([[0, 2], [4, 6]], dtype=np.float32)


def _manifest(path: Path, count: int) -> None:
    connection = runtime._create_manifest(path, "test-cache-key")
    try:
        base = datetime(2026, 1, 1)
        connection.executemany(
            "INSERT INTO selected(file_path, timestamp, relative_path, file_size) VALUES (?, ?, ?, ?)",
            (
                (
                    f"/nas/image-{index}.tif",
                    (base + timedelta(seconds=index)).isoformat(),
                    f"image-{index}.tif",
                    2_000_000,
                )
                for index in range(count)
            ),
        )
        connection.execute("UPDATE metadata SET value='1' WHERE key='selection_complete'")
        connection.commit()
    finally:
        connection.close()


def test_choose_worker_count_requires_fifteen_percent_gain() -> None:
    assert runtime.choose_worker_count([
        {"workers": 1, "images_per_second": 100},
        {"workers": 2, "images_per_second": 110},
        {"workers": 4, "images_per_second": 130},
    ]) == 4
    assert runtime.choose_worker_count([
        {"workers": 1, "images_per_second": 100},
        {"workers": 2, "images_per_second": 120},
        {"workers": 4, "images_per_second": 130},
    ]) == 2


def test_transient_nas_reads_retry_three_times_without_losing_metrics() -> None:
    compiled = _Compiled(failures=3)
    delays: list[float] = []
    mean, std, q95, error, transient = runtime.process_image_with_retries(
        compiled,
        "/nas/image.tif",
        retry_delays=(1.0, 3.0, 10.0),
        sleep=delays.append,
    )

    assert compiled.calls == 4
    assert delays == [1.0, 3.0, 10.0]
    assert error == ""
    assert transient is False
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(np.std([0, 2, 4, 6]))
    assert q95 == pytest.approx(5.7)


def test_defective_single_image_is_recorded_without_retrying() -> None:
    class Broken:
        calls = 0

        def run(self, _path: str) -> np.ndarray:
            self.calls += 1
            raise ValueError("corrupt TIFF")

    compiled = Broken()
    mean, std, q95, error, transient = runtime.process_image_with_retries(
        compiled,
        "/nas/corrupt.tif",
        retry_delays=(0.0, 0.0, 0.0),
    )

    assert compiled.calls == 1
    assert (mean, std, q95) == (None, None, None)
    assert error == "ValueError: corrupt TIFF"
    assert transient is False


def test_pending_manifest_reads_are_memory_bounded_at_100k_rows(tmp_path) -> None:
    path = tmp_path / "manifest.sqlite"
    _manifest(path, 100_000)
    connection = sqlite3.connect(path)
    try:
        tracemalloc.start()
        rows = runtime._fetch_pending(connection, 128)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        connection.close()

    assert len(rows) == 128
    assert peak < 2 * 1024 * 1024


def test_partial_manifest_resumes_and_csv_has_no_duplicate_rows(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manifest.sqlite"
    _manifest(path, 3)
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE selected SET status=1, mean_intensity=3, spatial_std_intensity=?, q95_intensity=5.7 WHERE id=1",
        (float(np.std([0, 2, 4, 6])),),
    )
    connection.commit()
    connection.close()

    monkeypatch.setattr(runtime, "compile_pipeline", lambda _graph: _Compiled())
    monkeypatch.setattr(runtime, "CALIBRATION_CANDIDATES", (1,))
    monkeypatch.setattr(runtime, "CALIBRATION_IMAGES_PER_STAGE", 1)
    reports: list[tuple[str, dict]] = []
    processed, successful, failed, processed_bytes, _, _, resumed = runtime.process_manifest(
        path,
        GRAPH,
        3,
        6_000_000,
        threading.Event(),
        lambda step, payload: reports.append((step, payload)),
    )

    csv_path = tmp_path / "result.csv"
    points = runtime.export_and_aggregate(path, csv_path, threading.Event(), lambda *_: None)
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert resumed is True
    assert (processed, successful, failed, processed_bytes) == (3, 3, 0, 6_000_000)
    assert len(rows) == 4
    assert len({row.split(",")[2] for row in rows[1:]}) == 3
    assert points[0].image_count == 3


def test_massive_transient_failure_stops_processing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manifest.sqlite"
    _manifest(path, 8)
    monkeypatch.setattr(runtime, "compile_pipeline", lambda _graph: _Compiled())
    monkeypatch.setattr(runtime, "CALIBRATION_CANDIDATES", (1,))
    monkeypatch.setattr(runtime, "CALIBRATION_IMAGES_PER_STAGE", 1)
    monkeypatch.setattr(runtime, "PROCESS_BATCH_PER_WORKER", 8)
    monkeypatch.setattr(runtime, "FAILURE_WINDOW", 4)
    monkeypatch.setattr(runtime, "FAILURE_FRACTION_LIMIT", 0.75)
    monkeypatch.setattr(
        runtime,
        "process_image_with_retries",
        lambda *_args, **_kwargs: (None, None, None, "OSError: NAS unavailable", True),
    )

    with pytest.raises(runtime.SourceUnavailableError, match="NAS appears unavailable"):
        runtime.process_manifest(
            path,
            GRAPH,
            8,
            16_000_000,
            threading.Event(),
            lambda *_: None,
        )
