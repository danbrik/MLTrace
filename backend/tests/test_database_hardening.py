import sqlite3
import threading
import time
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from app.database import (
    SQLITE_BUSY_TIMEOUT_MS,
    configure_sqlite_engine,
    is_sqlite_lock_error,
)
from app.projects import _catalog_engine


def _pragma_snapshot(engine) -> tuple[str, int, int, int]:
    with engine.connect() as connection:
        return (
            connection.execute(text("PRAGMA journal_mode")).scalar_one().lower(),
            connection.execute(text("PRAGMA busy_timeout")).scalar_one(),
            connection.execute(text("PRAGMA synchronous")).scalar_one(),
            connection.execute(text("PRAGMA foreign_keys")).scalar_one(),
        )


def test_sqlite_engine_uses_wal_timeout_normal_sync_and_foreign_keys(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'hardened.db'}",
        connect_args={"timeout": SQLITE_BUSY_TIMEOUT_MS / 1000},
    )
    configure_sqlite_engine(engine)

    assert _pragma_snapshot(engine) == ("wal", SQLITE_BUSY_TIMEOUT_MS, 1, 1)


def test_catalog_database_uses_the_same_sqlite_hardening() -> None:
    assert _pragma_snapshot(_catalog_engine) == ("wal", SQLITE_BUSY_TIMEOUT_MS, 1, 1)


def test_only_real_sqlite_lock_errors_are_classified_for_retry() -> None:
    busy = sqlite3.OperationalError("database is locked")
    assert is_sqlite_lock_error(OperationalError("write", {}, busy))
    assert not is_sqlite_lock_error(OperationalError("select", {}, sqlite3.OperationalError("no such table: missing")))
    assert not is_sqlite_lock_error(ValueError("model failed"))


def test_sqlite_writer_waits_for_short_competing_transaction(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'writers.db'}", connect_args={"timeout": 1})
    configure_sqlite_engine(engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)"))
        connection.execute(text("INSERT INTO values_table (id, value) VALUES (1, 0)"))

    first = engine.raw_connection()
    first.execute("BEGIN IMMEDIATE")
    first.execute("UPDATE values_table SET value = 1 WHERE id = 1")

    completed = threading.Event()

    def competing_writer() -> None:
        with engine.begin() as connection:
            connection.execute(text("UPDATE values_table SET value = 2 WHERE id = 1"))
        completed.set()

    thread = threading.Thread(target=competing_writer)
    thread.start()
    time.sleep(0.1)
    assert not completed.is_set()
    first.commit()
    first.close()
    thread.join(timeout=2)

    assert completed.is_set()
    with engine.connect() as connection:
        assert connection.execute(text("SELECT value FROM values_table WHERE id = 1")).scalar_one() == 2
