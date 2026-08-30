from __future__ import annotations

import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, Iterator

import numpy as np

from app import models
from app.database import data_dir
from app.preprocessing.base import ImageLoadError
from app.preprocessing.pipeline import CompiledPreprocessingPipeline, compile_pipeline
from app.scanner import TIFF_EXTENSIONS, extract_timestamp
from app.schemas import (
    ImageDistributionHourlyPoint,
    ImageDistributionMetricSummary,
    ImageDistributionPeriod,
    ImageDistributionResponse,
    PreprocessingGraph,
)


RUNTIME_VERSION = "image-distribution-stream-v2"
CALIBRATION_CANDIDATES = (1, 2, 4)
CALIBRATION_IMAGES_PER_STAGE = 256
PROCESS_BATCH_PER_WORKER = 32
PROGRESS_INTERVAL_SECONDS = 0.5
TRANSIENT_RETRY_DELAYS = (1.0, 3.0, 10.0)
FAILURE_WINDOW = 100
FAILURE_FRACTION_LIMIT = 0.8
CSV_FIELDS = [
    "image_index",
    "timestamp",
    "relative_path",
    "mean_intensity",
    "spatial_std_intensity",
    "q95_intensity",
    "error",
]

ProgressReporter = Callable[[str, dict], None]


class RuntimeAbortedError(Exception):
    pass


class SourceUnavailableError(RuntimeError):
    pass


def _folder_path(folder: models.DatasetFolder) -> Path:
    root = Path(folder.dataset.root_path).expanduser()
    return root if folder.relative_path == "." else root / folder.relative_path


def _folder_signature(folder: models.DatasetFolder) -> str:
    path = _folder_path(folder)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    raw = json.dumps({
        "folder_id": folder.id,
        "path": str(path),
        "mtime_ns": mtime_ns,
        "image_count": folder.image_count,
        "regex": folder.dataset.timestamp_regex,
        "format": folder.dataset.timestamp_format,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def configuration_key(training_dataset: models.TrainingDataset, pipeline: models.PreprocessingPipeline) -> str:
    folders = {rule.folder.id: rule.folder for rule in training_dataset.rules}
    payload = {
        "version": RUNTIME_VERSION,
        "training_dataset_id": training_dataset.id,
        "training_dataset_updated_at": training_dataset.updated_at.isoformat() if training_dataset.updated_at else None,
        "pipeline_id": pipeline.id,
        "pipeline_graph": pipeline.graph,
        "rules": sorted([
            [rule.id, rule.folder_id, rule.start_timestamp.isoformat(), rule.end_timestamp.isoformat(), rule.stride]
            for rule in training_dataset.rules
        ]),
        "folders": sorted((folder_id, _folder_signature(folder)) for folder_id, folder in folders.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def folder_index_path(folder_id: int) -> Path:
    return data_dir() / "image_distribution" / "folder_indexes" / f"{folder_id}.sqlite"


def run_manifest_path(run_id: int) -> Path:
    return data_dir() / "image_distribution_runs" / str(run_id) / "manifest.sqlite"


def cache_csv_path(cache_key: str) -> Path:
    return data_dir() / "image_distribution" / f"{cache_key}.csv"


def cache_result_path(cache_key: str) -> Path:
    return data_dir() / "image_distribution" / f"{cache_key}.json"


def _open_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


@contextmanager
def _index_lock(path: Path, abort_event: threading.Event, report: ProgressReporter) -> Iterator[None]:
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            break
        except FileExistsError:
            if abort_event.is_set():
                raise RuntimeAbortedError()
            try:
                if time.time() - lock.stat().st_mtime > 24 * 60 * 60:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            report("waiting_for_folder_index", {"detail": f"Waiting for index lock: {path.name}"})
            abort_event.wait(2.0)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _valid_folder_index(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        with _open_sqlite(path) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            complete = connection.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
            return row is not None and row[0] == signature and complete is not None and complete[0] == "1"
    except (OSError, sqlite3.Error):
        return False


def ensure_folder_index(
    folder: models.DatasetFolder,
    abort_event: threading.Event,
    report: ProgressReporter,
) -> Path:
    path = folder_index_path(folder.id)
    signature = _folder_signature(folder)
    if _valid_folder_index(path, signature):
        report("loading_folder_index", {"phase_processed": folder.image_count, "phase_total": folder.image_count})
        return path
    with _index_lock(path, abort_event, report):
        if _valid_folder_index(path, signature):
            report("loading_folder_index", {"phase_processed": folder.image_count, "phase_total": folder.image_count})
            return path
        source = _folder_path(folder)
        dataset = folder.dataset
        if not dataset.timestamp_regex or not dataset.timestamp_format:
            raise ValueError(f"Dataset '{dataset.name}' has no confirmed timestamp parser.")
        temporary = path.with_suffix(f".sqlite.{os.getpid()}.part")
        temporary.unlink(missing_ok=True)
        connection = _open_sqlite(temporary)
        try:
            report("indexing_folders", {"phase_processed": 0, "phase_total": folder.image_count})
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE images (timestamp TEXT NOT NULL, file_name TEXT PRIMARY KEY, file_size INTEGER NOT NULL) WITHOUT ROWID"
            )
            batch: list[tuple[str, str, int]] = []
            indexed = 0
            for entry in os.scandir(source):
                if abort_event.is_set():
                    raise RuntimeAbortedError()
                if not entry.is_file(follow_symlinks=False) or Path(entry.name).suffix.lower() not in TIFF_EXTENSIONS:
                    continue
                try:
                    _, timestamp = extract_timestamp(entry.name, dataset.timestamp_regex, dataset.timestamp_format)
                    file_size = int(entry.stat(follow_symlinks=False).st_size)
                except ValueError as exc:
                    raise ValueError(
                        f"File '{entry.name}' in dataset '{dataset.name}' does not match the confirmed timestamp parser."
                    ) from exc
                batch.append((timestamp.isoformat(), entry.name, file_size))
                indexed += 1
                if len(batch) >= 5000:
                    connection.executemany("INSERT OR REPLACE INTO images VALUES (?, ?, ?)", batch)
                    connection.commit()
                    batch.clear()
                    Path(path.with_suffix(path.suffix + ".lock")).touch(exist_ok=True)
                    report("indexing_folders", {"phase_processed": indexed, "phase_total": folder.image_count})
            if batch:
                connection.executemany("INSERT OR REPLACE INTO images VALUES (?, ?, ?)", batch)
            connection.execute("CREATE INDEX ix_images_timestamp ON images(timestamp, file_name)")
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
                ("signature", signature),
                ("complete", "1"),
                ("row_count", str(indexed)),
            ])
            connection.commit()
        finally:
            connection.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(path)
    return path


def _manifest_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(connection.execute("SELECT key, value FROM metadata"))
    except sqlite3.OperationalError:
        return {}


def _create_manifest(path: Path, cache_key: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)
    connection = _open_sqlite(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE selected (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            status INTEGER NOT NULL DEFAULT 0,
            mean_intensity REAL,
            spatial_std_intensity REAL,
            q95_intensity REAL,
            error TEXT,
            processed_at TEXT
        )"""
    )
    connection.execute("CREATE INDEX ix_selected_status_id ON selected(status, id)")
    connection.execute("CREATE INDEX ix_selected_timestamp ON selected(timestamp, id)")
    connection.execute("INSERT INTO metadata VALUES ('cache_key', ?)", (cache_key,))
    connection.execute("INSERT INTO metadata VALUES ('selection_complete', '0')")
    connection.commit()
    return connection


def prepare_manifest(
    run_id: int,
    training_dataset: models.TrainingDataset,
    pipeline: models.PreprocessingPipeline,
    abort_event: threading.Event,
    report: ProgressReporter,
) -> tuple[Path, str, int, int, bool]:
    cache_key = configuration_key(training_dataset, pipeline)
    path = run_manifest_path(run_id)
    resumed = False
    if path.is_file():
        connection = _open_sqlite(path)
        metadata = _manifest_metadata(connection)
        if metadata.get("cache_key") == cache_key and metadata.get("selection_complete") == "1":
            resumed = connection.execute("SELECT COUNT(*) FROM selected WHERE status != 0").fetchone()[0] > 0
            total, total_bytes = connection.execute("SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM selected").fetchone()
            connection.close()
            return path, cache_key, int(total), int(total_bytes), resumed
        connection.close()
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            candidate.unlink(missing_ok=True)

    connection = _create_manifest(path, cache_key)
    try:
        rules = sorted(training_dataset.rules, key=lambda rule: (rule.start_timestamp, rule.folder_id, rule.id))
        phase_total = sum(int(rule.matching_images or 0) for rule in rules) or None
        examined = 0
        selected_count = 0
        report("selecting_images", {
            "phase_processed": 0,
            "phase_total": phase_total,
            "selected_images": 0,
        })
        for rule in rules:
            index_path = ensure_folder_index(rule.folder, abort_event, report)
            source_folder = _folder_path(rule.folder)
            relative_folder = rule.folder.relative_path
            with _open_sqlite(index_path) as index:
                cursor = index.execute(
                    "SELECT timestamp, file_name, file_size FROM images WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp, file_name",
                    (rule.start_timestamp.isoformat(), rule.end_timestamp.isoformat()),
                )
                batch: list[tuple[str, str, str, int]] = []
                for position, (timestamp, file_name, file_size) in enumerate(cursor):
                    if abort_event.is_set():
                        raise RuntimeAbortedError()
                    examined += 1
                    if position % max(1, rule.stride) != 0:
                        continue
                    relative = str(Path(relative_folder) / file_name)
                    batch.append((str(source_folder / file_name), timestamp, relative, int(file_size)))
                    if len(batch) >= 5000:
                        before = connection.total_changes
                        connection.executemany(
                            "INSERT OR IGNORE INTO selected(file_path, timestamp, relative_path, file_size) VALUES (?, ?, ?, ?)",
                            batch,
                        )
                        selected_count += connection.total_changes - before
                        connection.commit()
                        batch.clear()
                        report("selecting_images", {
                            "phase_processed": examined,
                            "phase_total": phase_total,
                            "selected_images": selected_count,
                        })
                if batch:
                    before = connection.total_changes
                    connection.executemany(
                        "INSERT OR IGNORE INTO selected(file_path, timestamp, relative_path, file_size) VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    selected_count += connection.total_changes - before
                    connection.commit()
        connection.execute("UPDATE metadata SET value='1' WHERE key='selection_complete'")
        connection.commit()
        total, total_bytes = connection.execute("SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM selected").fetchone()
        if not total:
            raise ValueError("Train/Test dataset selects no images.")
        report("selecting_images", {
            "phase_processed": examined,
            "phase_total": phase_total or examined,
            "selected_images": int(total),
        })
        return path, cache_key, int(total), int(total_bytes), False
    finally:
        connection.close()


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, ImageLoadError):
        error = error.original
    return isinstance(error, (OSError, TimeoutError, ConnectionError))


def process_image_with_retries(
    compiled: CompiledPreprocessingPipeline,
    file_path: str,
    *,
    retry_delays: tuple[float, ...] = TRANSIENT_RETRY_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[float | None, float | None, float | None, str, bool]:
    last_error: BaseException | None = None
    for attempt in range(len(retry_delays) + 1):
        try:
            values = np.asarray(compiled.run(file_path), dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError("Preprocessing produced no finite pixels")
            return (
                float(np.mean(finite)),
                float(np.std(finite, ddof=0)),
                float(np.quantile(finite, 0.95)),
                "",
                False,
            )
        except Exception as exc:  # noqa: BLE001 - result rows preserve all image-level failures
            last_error = exc
            if not _is_transient(exc) or attempt >= len(retry_delays):
                break
            sleep(retry_delays[attempt])
    assert last_error is not None
    return None, None, None, f"{type(last_error).__name__}: {last_error}", _is_transient(last_error)


def _fetch_pending(connection: sqlite3.Connection, limit: int) -> list[tuple[int, str, int]]:
    return [
        (int(row[0]), str(row[1]), int(row[2]))
        for row in connection.execute(
            "SELECT id, file_path, file_size FROM selected WHERE status=0 ORDER BY id LIMIT ?", (limit,)
        )
    ]


def _run_batch(
    compiled: CompiledPreprocessingPipeline,
    rows: list[tuple[int, str, int]],
    workers: int,
) -> tuple[list[tuple], float]:
    started = time.perf_counter()
    def calculate(row: tuple[int, str, int]) -> tuple:
        item_id, file_path, file_size = row
        mean, std, q95, error, transient = process_image_with_retries(compiled, file_path)
        return item_id, file_size, mean, std, q95, error, transient
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-distribution") as executor:
        results = list(executor.map(calculate, rows))
    return results, max(1e-9, time.perf_counter() - started)


def choose_worker_count(calibration: list[dict]) -> int:
    if not calibration:
        return 1
    best_workers = int(calibration[0]["workers"])
    best_rate = float(calibration[0]["images_per_second"])
    for candidate in calibration[1:]:
        rate = float(candidate["images_per_second"])
        if rate >= best_rate * 1.15:
            best_workers = int(candidate["workers"])
            best_rate = rate
    return best_workers


def _persist_results(connection: sqlite3.Connection, results: list[tuple]) -> tuple[int, int, int, int, list[bool]]:
    successful = 0
    failed = 0
    processed_bytes = 0
    transient_failures: list[bool] = []
    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    updates = []
    for item_id, file_size, mean, std, q95, error, transient in results:
        status = 2 if error else 1
        successful += status == 1
        failed += status == 2
        processed_bytes += file_size
        transient_failures.append(bool(error and transient))
        updates.append((status, mean, std, q95, error or None, now, item_id))
    connection.executemany(
        "UPDATE selected SET status=?, mean_intensity=?, spatial_std_intensity=?, q95_intensity=?, error=?, processed_at=? WHERE id=?",
        updates,
    )
    connection.commit()
    return len(results), successful, failed, processed_bytes, transient_failures


def process_manifest(
    manifest_path: Path,
    graph: dict,
    total_images: int,
    total_bytes: int,
    abort_event: threading.Event,
    report: ProgressReporter,
) -> tuple[int, int, int, int, list[dict], list[dict], bool]:
    connection = _open_sqlite(manifest_path)
    compiled = compile_pipeline(PreprocessingGraph.model_validate(graph))
    existing = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(status=1),0), COALESCE(SUM(status=2),0), COALESCE(SUM(CASE WHEN status != 0 THEN file_size ELSE 0 END),0) FROM selected WHERE status != 0"
    ).fetchone()
    processed, successful, failed, processed_bytes = map(int, existing)
    resumed = processed > 0
    rate_samples: deque[tuple[float, int, int]] = deque()
    recent_failures: deque[bool] = deque(maxlen=FAILURE_WINDOW)
    calibration: list[dict] = []
    hourly_counts = [int(row[0]) for row in connection.execute(
        "SELECT COUNT(*) FROM selected GROUP BY substr(timestamp, 1, 13)"
    )]
    try:
        if processed < total_images:
            for workers in CALIBRATION_CANDIDATES:
                if abort_event.is_set():
                    raise RuntimeAbortedError()
                rows = _fetch_pending(connection, CALIBRATION_IMAGES_PER_STAGE)
                if not rows:
                    break
                report("calibrating_workers", {
                    "effective_worker_count": workers,
                    "processed_images": processed,
                    "total_images": total_images,
                })
                results, elapsed = _run_batch(compiled, rows, workers)
                count, good, bad, byte_count, transient = _persist_results(connection, results)
                processed += count
                successful += good
                failed += bad
                processed_bytes += byte_count
                recent_failures.extend(transient)
                calibration.append({
                    "workers": workers,
                    "images": count,
                    "seconds": elapsed,
                    "images_per_second": count / elapsed,
                    "mb_per_second": byte_count / elapsed / (1024 * 1024),
                })
                rate_samples.append((time.monotonic() - elapsed, count, byte_count))
                if len(recent_failures) == FAILURE_WINDOW and sum(recent_failures) >= math.ceil(FAILURE_WINDOW * FAILURE_FRACTION_LIMIT):
                    raise SourceUnavailableError(
                        "The NAS appears unavailable: at least 80% of the most recent image reads failed after retries."
                    )

        workers = choose_worker_count(calibration)
        selected_calibration = next(
            (entry for entry in calibration if entry["workers"] == workers),
            calibration[-1] if calibration else None,
        )
        image_rate = float(selected_calibration["images_per_second"]) if selected_calibration else 0.0
        report("processing_images", {
            "processed_images": processed,
            "total_images": total_images,
            "successful_images": successful,
            "failed_images": failed,
            "processed_bytes": processed_bytes,
            "total_bytes": total_bytes,
            "throughput_images_per_second": image_rate or None,
            "throughput_mb_per_second": (
                float(selected_calibration["mb_per_second"]) if selected_calibration else None
            ),
            "eta_seconds": (total_images - processed) / image_rate if image_rate > 0 else None,
            "effective_worker_count": workers,
            "calibration_results": calibration,
            "stride_projections": stride_projections(total_images, image_rate, hourly_counts),
            "resumed": resumed,
        })
        while processed < total_images:
            if abort_event.is_set():
                raise RuntimeAbortedError()
            rows = _fetch_pending(connection, max(1, workers * PROCESS_BATCH_PER_WORKER))
            if not rows:
                break
            results, elapsed = _run_batch(compiled, rows, workers)
            count, good, bad, byte_count, transient = _persist_results(connection, results)
            processed += count
            successful += good
            failed += bad
            processed_bytes += byte_count
            recent_failures.extend(transient)
            now = time.monotonic()
            rate_samples.append((now - elapsed, count, byte_count))
            while rate_samples and now - rate_samples[0][0] > 60.0:
                rate_samples.popleft()
            window_seconds = max(elapsed, now - rate_samples[0][0] if len(rate_samples) > 1 else elapsed)
            window_images = sum(item[1] for item in rate_samples)
            window_bytes = sum(item[2] for item in rate_samples)
            image_rate = window_images / max(window_seconds, 1e-9)
            byte_rate = window_bytes / max(window_seconds, 1e-9)
            eta = (total_images - processed) / image_rate if image_rate > 0 else None
            projections = stride_projections(total_images, image_rate, hourly_counts)
            report("processing_images", {
                "processed_images": processed,
                "total_images": total_images,
                "successful_images": successful,
                "failed_images": failed,
                "processed_bytes": processed_bytes,
                "total_bytes": total_bytes,
                "throughput_images_per_second": image_rate,
                "throughput_mb_per_second": byte_rate / (1024 * 1024),
                "eta_seconds": eta,
                "effective_worker_count": workers,
                "calibration_results": calibration,
                "stride_projections": projections,
                "resumed": resumed,
            })
            if len(recent_failures) == FAILURE_WINDOW and sum(recent_failures) >= math.ceil(FAILURE_WINDOW * FAILURE_FRACTION_LIMIT):
                raise SourceUnavailableError(
                    "The NAS appears unavailable: at least 80% of the most recent image reads failed after retries."
                )
        return processed, successful, failed, processed_bytes, calibration, stride_projections(total_images, image_rate, hourly_counts), resumed
    finally:
        connection.close()


def stride_projections(total_images: int, image_rate: float, hourly_counts: list[int]) -> list[dict]:
    median_per_hour = float(np.median(hourly_counts)) if hourly_counts else 0.0
    minimum_per_hour = min(hourly_counts) if hourly_counts else 0
    return [{
        "factor": factor,
        "estimated_images": math.ceil(total_images / factor),
        "estimated_seconds": (total_images / factor / image_rate) if image_rate > 0 else None,
        "estimated_median_images_per_hour": median_per_hour / factor,
        "estimated_min_images_per_hour": minimum_per_hour / factor,
    } for factor in (2, 5, 10)]


def _summary(values: list[float]) -> ImageDistributionMetricSummary:
    return ImageDistributionMetricSummary(
        median=float(np.quantile(values, 0.5)),
        q25=float(np.quantile(values, 0.25)),
        q75=float(np.quantile(values, 0.75)),
    )


def export_and_aggregate(
    manifest_path: Path,
    csv_path: Path,
    abort_event: threading.Event,
    report: ProgressReporter,
) -> list[ImageDistributionHourlyPoint]:
    connection = _open_sqlite(manifest_path)
    part_path = csv_path.with_suffix(csv_path.suffix + ".part")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with part_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            cursor = connection.execute(
                "SELECT timestamp, relative_path, status, mean_intensity, spatial_std_intensity, q95_intensity, COALESCE(error,'') FROM selected ORDER BY timestamp, id"
            )
            for timestamp, relative_path, status, mean, std, q95, error in cursor:
                if abort_event.is_set():
                    raise RuntimeAbortedError()
                writer.writerow({
                    "image_index": written,
                    "timestamp": timestamp,
                    "relative_path": relative_path,
                    "mean_intensity": "" if mean is None else repr(float(mean)),
                    "spatial_std_intensity": "" if std is None else repr(float(std)),
                    "q95_intensity": "" if q95 is None else repr(float(q95)),
                    "error": error,
                })
                written += 1
                if written % 10000 == 0:
                    report("writing_csv", {"phase_processed": written})
        part_path.replace(csv_path)
        hour_total = int(connection.execute(
            "SELECT COUNT(DISTINCT substr(timestamp, 1, 13)) FROM selected WHERE status=1"
        ).fetchone()[0])
        report("aggregating_hourly", {"phase_processed": 0, "phase_total": hour_total})
        points: list[ImageDistributionHourlyPoint] = []
        current_hour: datetime | None = None
        metrics = {"mean": [], "std": [], "q95": []}

        def finish_hour() -> None:
            if current_hour is None or not metrics["mean"]:
                return
            points.append(ImageDistributionHourlyPoint(
                hour=current_hour,
                image_count=len(metrics["mean"]),
                mean_intensity=_summary(metrics["mean"]),
                spatial_std_intensity=_summary(metrics["std"]),
                q95_intensity=_summary(metrics["q95"]),
            ))
            report("aggregating_hourly", {
                "phase_processed": len(points),
                "phase_total": hour_total,
            })

        cursor = connection.execute(
            "SELECT timestamp, mean_intensity, spatial_std_intensity, q95_intensity "
            "FROM selected WHERE status=1 ORDER BY timestamp, id"
        )
        for timestamp, mean, std, q95 in cursor:
            if abort_event.is_set():
                raise RuntimeAbortedError()
            parsed = datetime.fromisoformat(timestamp)
            hour = parsed.replace(minute=0, second=0, microsecond=0)
            if current_hour is not None and hour != current_hour:
                finish_hour()
                metrics = {"mean": [], "std": [], "q95": []}
            current_hour = hour
            metrics["mean"].append(float(mean))
            metrics["std"].append(float(std))
            metrics["q95"].append(float(q95))
        finish_hour()
        return points
    finally:
        connection.close()


def training_periods(training_dataset: models.TrainingDataset) -> list[ImageDistributionPeriod]:
    return [ImageDistributionPeriod(
        name=training_dataset.name,
        usage_label=training_dataset.usage_label,
        start=rule.start_timestamp,
        end=rule.end_timestamp,
    ) for rule in sorted(training_dataset.rules, key=lambda item: item.start_timestamp)]


def load_cached_result(cache_key: str) -> ImageDistributionResponse | None:
    csv_path = cache_csv_path(cache_key)
    result_path = cache_result_path(cache_key)
    if not csv_path.is_file() or not result_path.is_file():
        return None
    try:
        result = ImageDistributionResponse.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return result.model_copy(update={"cache_hit": True})


def save_cached_result(result: ImageDistributionResponse) -> None:
    path = cache_result_path(result.cache_key)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(result.model_dump_json(), encoding="utf-8")
    temporary.replace(path)
