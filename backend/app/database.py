from collections.abc import Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()


def _url_data_dir(database_url: str) -> Path:
    url = make_url(database_url)
    if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
        return Path(url.database).expanduser().resolve().parent
    return Path(".mltrace").resolve()


_database_url: ContextVar[str] = ContextVar("mltrace_database_url", default=settings.database_url)
_artifact_dir: ContextVar[Path] = ContextVar(
    "mltrace_artifact_dir", default=_url_data_dir(settings.database_url)
)


def engine_options(database_url: str) -> dict:
    url = make_url(database_url)
    options: dict = {"pool_pre_ping": True}
    if url.drivername.startswith("sqlite"):
        if url.database and url.database != ":memory:":
            database_path = Path(url.database).expanduser()
            database_path.parent.mkdir(parents=True, exist_ok=True)
        options["connect_args"] = {"check_same_thread": False}
    return options


def create_project_engine(database_url: str):
    project_engine = create_engine(database_url, **engine_options(database_url))
    if make_url(database_url).drivername.startswith("sqlite"):
        @event.listens_for(project_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()
    return project_engine


engine = create_project_engine(settings.database_url)


_sessionmakers: dict[str, sessionmaker] = {
    settings.database_url: sessionmaker(autocommit=False, autoflush=False, bind=engine)
}


def session_factory(database_url: str) -> sessionmaker:
    factory = _sessionmakers.get(database_url)
    if factory is None:
        factory = sessionmaker(autocommit=False, autoflush=False, bind=create_project_engine(database_url))
        _sessionmakers[database_url] = factory
    return factory


def SessionLocal() -> Session:  # noqa: N802 - compatibility with the former sessionmaker
    db = session_factory(_database_url.get())()
    db.info["database_url"] = _database_url.get()
    db.info["data_dir"] = str(_artifact_dir.get())
    return db


@contextmanager
def project_context(database_url: str, artifact_dir: str | Path) -> Iterator[None]:
    url_token = _database_url.set(database_url)
    dir_token = _artifact_dir.set(Path(artifact_dir).expanduser().resolve())
    try:
        yield
    finally:
        _artifact_dir.reset(dir_token)
        _database_url.reset(url_token)


def data_dir() -> Path:
    """Directory for run artifacts and logs.

    For SQLite this is the folder that holds the database (the conventional
    ``.mltrace/`` directory); otherwise it falls back to ``./.mltrace``.
    """
    return _artifact_dir.get()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_database_schema() -> None:
    current_engine = session_factory(_database_url.get()).kw["bind"]
    Base.metadata.create_all(bind=current_engine)


def vacuum_database() -> None:
    """Reclaim space and truncate the WAL. Useful after deleting large rows
    (e.g. cleared heatmaps). VACUUM must run outside a transaction."""
    database_url = _database_url.get()
    current_engine = session_factory(database_url).kw["bind"]
    is_sqlite = make_url(database_url).drivername.startswith("sqlite")
    with current_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql("VACUUM")
        if is_sqlite:
            conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
