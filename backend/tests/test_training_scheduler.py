from pathlib import Path

from sqlalchemy.engine import make_url

from app.training.scheduler import _worker_database_url


def test_worker_database_url_resolves_relative_sqlite_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = _worker_database_url("sqlite:///./.mltrace/mltrace.db")
    url = make_url(resolved)

    assert url.drivername == "sqlite"
    assert Path(url.database).is_absolute()
    assert Path(url.database) == tmp_path / ".mltrace" / "mltrace.db"


def test_worker_database_url_leaves_non_sqlite_urls_unchanged() -> None:
    url = "postgresql+psycopg://user:password@localhost:5432/mltrace"

    assert _worker_database_url(url) == url
