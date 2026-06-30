from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()


def engine_options(database_url: str) -> dict:
    url = make_url(database_url)
    options: dict = {"pool_pre_ping": True}
    if url.drivername.startswith("sqlite"):
        if url.database and url.database != ":memory:":
            database_path = Path(url.database).expanduser()
            database_path.parent.mkdir(parents=True, exist_ok=True)
        options["connect_args"] = {"check_same_thread": False}
    return options


engine = create_engine(settings.database_url, **engine_options(settings.database_url))


if make_url(settings.database_url).drivername.startswith("sqlite"):
    # Training runs execute in separate worker processes that write metrics
    # concurrently with the API. WAL lets readers and a single writer coexist
    # without "database is locked" errors; the short busy_timeout absorbs the
    # brief contention when a worker commits per-epoch metrics.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def data_dir() -> Path:
    """Directory for run artifacts and logs.

    For SQLite this is the folder that holds the database (the conventional
    ``.mltrace/`` directory); otherwise it falls back to ``./.mltrace``.
    """
    url = make_url(settings.database_url)
    if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
        return Path(url.database).expanduser().resolve().parent
    return Path(".mltrace").resolve()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_database_schema() -> None:
    Base.metadata.create_all(bind=engine)


def vacuum_database() -> None:
    """Reclaim space and truncate the WAL. Useful after deleting large rows
    (e.g. cleared heatmaps). VACUUM must run outside a transaction."""
    is_sqlite = make_url(settings.database_url).drivername.startswith("sqlite")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql("VACUUM")
        if is_sqlite:
            conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
