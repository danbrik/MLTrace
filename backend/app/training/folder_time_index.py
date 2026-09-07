"""Disk-backed timestamp indexes for very large TIFF folders.

The regular dataset scanner intentionally stores only representative images in
the application database.  This sidecar index keeps range resolution bounded in
RAM even when a folder contains millions of files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
import fcntl
import heapq
import json
import os
from pathlib import Path
import sqlite3
import uuid

from app import models
from app.database import data_dir
from app.scanner import TIFF_EXTENSIONS, extract_timestamp
from app.training.data import ResolvedDatasetImage


_EPOCH = datetime(1970, 1, 1)


def _timestamp_us(value: datetime) -> int:
    return int((value.replace(tzinfo=None) - _EPOCH).total_seconds() * 1_000_000)


def _folder_path(folder: models.DatasetFolder) -> Path:
    root = Path(folder.dataset.root_path).expanduser()
    return root if folder.relative_path == "." else root / folder.relative_path


def _index_path(folder_id: int) -> Path:
    return data_dir() / "folder_time_index" / f"{folder_id}.sqlite3"


def _signature(folder: models.DatasetFolder, path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "schema_version": 2,
        "folder_id": folder.id,
        "folder_path": str(path.resolve()),
        "directory_mtime_ns": stat.st_mtime_ns,
        "timestamp_regex": folder.dataset.timestamp_regex,
        "timestamp_format": folder.dataset.timestamp_format,
    }


def _is_current(path: Path, signature: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = 'signature'").fetchone()
        return bool(row and json.loads(row[0]) == signature)
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return False


def ensure_folder_time_index(
    folder: models.DatasetFolder,
    *,
    abort_check: Callable[[], None] | None = None,
) -> Path:
    """Build a reusable SQLite range index without collecting paths in RAM."""

    folder_path = _folder_path(folder)
    signature = _signature(folder, folder_path)
    target = _index_path(folder.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_current(target, signature):
        return target

    lock_path = target.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _is_current(target, signature):
            return target
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(temporary)
            if abort_check:
                def sqlite_progress() -> int:
                    try:
                        abort_check()
                    except Exception:
                        return 1
                    return 0
                connection.set_progress_handler(sqlite_progress, 10_000)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE unsorted_images (timestamp_us INTEGER NOT NULL, file_name TEXT NOT NULL)")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            batch: list[tuple[int, str]] = []
            regex = folder.dataset.timestamp_regex
            timestamp_format = folder.dataset.timestamp_format
            if not regex or not timestamp_format:
                raise ValueError(f"Dataset '{folder.dataset.name}' has no confirmed timestamp parser.")
            with os.scandir(folder_path) as entries:
                for number, entry in enumerate(entries, 1):
                    if abort_check and number % 1000 == 0:
                        abort_check()
                    if not entry.is_file() or Path(entry.name).suffix.lower() not in TIFF_EXTENSIONS:
                        continue
                    try:
                        _, timestamp = extract_timestamp(entry.name, regex, timestamp_format)
                    except ValueError as exc:
                        raise ValueError(
                            f"File '{entry.name}' in dataset '{folder.dataset.name}' does not match the confirmed timestamp parser."
                        ) from exc
                    batch.append((_timestamp_us(timestamp), entry.name))
                    if len(batch) >= 5000:
                        connection.executemany("INSERT INTO unsorted_images VALUES (?, ?)", batch)
                        connection.commit()
                        batch.clear()
            if batch:
                connection.executemany("INSERT INTO unsorted_images VALUES (?, ?)", batch)
            try:
                connection.execute("CREATE TABLE images (ordinal INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_us INTEGER NOT NULL, file_name TEXT NOT NULL)")
                connection.execute("INSERT INTO images(timestamp_us, file_name) SELECT timestamp_us, file_name FROM unsorted_images ORDER BY timestamp_us, file_name")
                connection.execute("DROP TABLE unsorted_images")
                connection.execute("CREATE UNIQUE INDEX ix_images_time_name ON images(timestamp_us, file_name)")
            except sqlite3.OperationalError:
                if abort_check:
                    abort_check()
                raise
            connection.execute(
                "INSERT INTO metadata VALUES ('signature', ?)",
                (json.dumps(signature, sort_keys=True, separators=(",", ":")),),
            )
            connection.commit()
            connection.close()
            os.replace(temporary, target)
        finally:
            if connection is not None:
                connection.close()
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return target


def _iter_rule_range(
    rule: models.TrainingDatasetRule,
    start: datetime,
    end: datetime,
    *,
    end_inclusive: bool,
    abort_check: Callable[[], None] | None,
) -> Iterator[ResolvedDatasetImage]:
    rule_start = max(rule.start_timestamp, start)
    rule_end = min(rule.end_timestamp, end)
    if rule_end < rule_start or (rule_end == rule_start and not end_inclusive):
        return
    index_path = ensure_folder_time_index(rule.folder, abort_check=abort_check)
    lower = _timestamp_us(rule_start)
    upper = _timestamp_us(rule_end)
    original_start = _timestamp_us(rule.start_timestamp)
    comparator = "<=" if end_inclusive or rule_end < end else "<"
    folder_path = _folder_path(rule.folder)
    with sqlite3.connect(index_path) as connection:
        if abort_check:
            def query_progress() -> int:
                try:
                    abort_check()
                except Exception:
                    return 1
                return 0
            connection.set_progress_handler(query_progress, 10_000)
        base_row = connection.execute(
            "SELECT ordinal FROM images WHERE timestamp_us >= ? ORDER BY timestamp_us, file_name LIMIT 1", (original_start,),
        ).fetchone()
        if base_row is None:
            return
        base_ordinal = int(base_row[0])
        cursor = connection.execute(
            f"SELECT timestamp_us, file_name FROM images WHERE timestamp_us >= ? AND timestamp_us {comparator} ? "
            "AND ((ordinal - ?) % ?) = 0 ORDER BY timestamp_us, file_name",
            (lower, upper, base_ordinal, max(1, int(rule.stride))),
        )
        for local_index, (timestamp_us, file_name) in enumerate(cursor):
            if abort_check and local_index % 1000 == 0:
                abort_check()
            timestamp = _EPOCH + timedelta(microseconds=timestamp_us)
            yield ResolvedDatasetImage(
                file_path=str(folder_path / file_name),
                timestamp_parsed=timestamp,
                dataset_name=rule.folder.dataset.name,
                dataset_root_path=rule.folder.dataset.root_path,
                folder_id=rule.folder.id,
                folder_relative_path=rule.folder.relative_path,
                file_name=file_name,
            )


def iter_training_dataset_range_records(
    training_dataset: models.TrainingDataset,
    start: datetime,
    end: datetime,
    *,
    end_inclusive: bool,
    abort_check: Callable[[], None] | None = None,
) -> Iterator[ResolvedDatasetImage]:
    """Yield a globally ordered, de-duplicated range with saved rule strides."""

    streams = [
        _iter_rule_range(rule, start, end, end_inclusive=end_inclusive, abort_check=abort_check)
        for rule in sorted(
            training_dataset.rules,
            key=lambda item: (item.start_timestamp, item.end_timestamp, item.folder_id, item.id),
        )
        if rule.end_timestamp >= start and rule.start_timestamp <= end
    ]
    merged = heapq.merge(*streams, key=lambda item: (item.timestamp_parsed, item.file_path))
    previous_path: str | None = None
    for record in merged:
        if record.file_path == previous_path:
            continue
        previous_path = record.file_path
        yield record


def nearest_training_dataset_records(
    training_dataset: models.TrainingDataset,
    target: datetime,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    limit_per_rule: int = 32,
) -> list[ResolvedDatasetImage]:
    """Resolve nearby stride-selected paths using indexed lookups only."""

    target_us = _timestamp_us(target)
    found: list[ResolvedDatasetImage] = []
    for rule in sorted(training_dataset.rules, key=lambda item: (item.start_timestamp, item.end_timestamp, item.folder_id, item.id)):
        lower_dt = max(rule.start_timestamp, start) if start is not None else rule.start_timestamp
        upper_dt = min(rule.end_timestamp, end) if end is not None else rule.end_timestamp
        if upper_dt < lower_dt:
            continue
        index_path = ensure_folder_time_index(rule.folder)
        lower, upper = _timestamp_us(lower_dt), _timestamp_us(upper_dt)
        with sqlite3.connect(index_path) as connection:
            base = connection.execute(
                "SELECT ordinal FROM images WHERE timestamp_us >= ? ORDER BY timestamp_us, file_name LIMIT 1",
                (_timestamp_us(rule.start_timestamp),),
            ).fetchone()
            if base is None:
                continue
            parameters = (lower, upper, int(base[0]), max(1, int(rule.stride)), target_us, limit_per_rule)
            before = connection.execute(
                "SELECT timestamp_us, file_name FROM images WHERE timestamp_us >= ? AND timestamp_us <= ? "
                "AND ((ordinal - ?) % ?) = 0 AND timestamp_us <= ? ORDER BY timestamp_us DESC, file_name DESC LIMIT ?",
                parameters,
            ).fetchall()
            after = connection.execute(
                "SELECT timestamp_us, file_name FROM images WHERE timestamp_us >= ? AND timestamp_us <= ? "
                "AND ((ordinal - ?) % ?) = 0 AND timestamp_us > ? ORDER BY timestamp_us, file_name LIMIT ?",
                parameters,
            ).fetchall()
        folder_path = _folder_path(rule.folder)
        for timestamp_us, file_name in before + after:
            found.append(ResolvedDatasetImage(
                file_path=str(folder_path / file_name), timestamp_parsed=_EPOCH + timedelta(microseconds=timestamp_us),
                dataset_name=rule.folder.dataset.name, dataset_root_path=rule.folder.dataset.root_path,
                folder_id=rule.folder.id, folder_relative_path=rule.folder.relative_path, file_name=file_name,
            ))
    unique = {record.file_path: record for record in found}
    return sorted(unique.values(), key=lambda item: (abs((item.timestamp_parsed - target).total_seconds()), item.timestamp_parsed, item.file_path))
