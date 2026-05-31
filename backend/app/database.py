from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_database_schema() -> None:
    Base.metadata.create_all(bind=engine)
