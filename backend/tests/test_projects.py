from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import projects


def configure_catalog(monkeypatch, tmp_path: Path, legacy_db: Path | None = None) -> None:
    root = tmp_path / ".mltrace"
    engine = create_engine(f"sqlite:///{root / 'catalog.db'}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(projects, "ROOT_DIR", root)
    monkeypatch.setattr(projects, "CATALOG_PATH", root / "catalog.db")
    monkeypatch.setattr(projects, "_catalog_engine", engine)
    monkeypatch.setattr(projects, "CatalogSession", sessionmaker(autocommit=False, autoflush=False, bind=engine))
    database = legacy_db or root / "missing-legacy.db"
    monkeypatch.setattr(projects, "get_settings", lambda: SimpleNamespace(database_url=f"sqlite:///{database}"))


def test_existing_database_is_registered_without_moving(monkeypatch, tmp_path: Path) -> None:
    legacy = tmp_path / ".mltrace" / "mltrace.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"")
    configure_catalog(monkeypatch, tmp_path, legacy)

    projects.initialize_catalog()
    registered = projects.list_projects()

    assert len(registered) == 1
    assert registered[0].name == "Default Project"
    assert Path(registered[0].artifact_dir) == legacy.parent
    assert legacy.exists()
    projects.initialize_catalog()
    assert len(projects.list_projects()) == 1


def test_create_project_provisions_isolated_database(monkeypatch, tmp_path: Path) -> None:
    configure_catalog(monkeypatch, tmp_path)
    projects.initialize_catalog()

    created = projects.create_project("Line A", "An isolated experiment")

    database = Path(created.artifact_dir) / "mltrace.db"
    assert database.exists()
    assert created.name == "Line A"
    assert len(projects.list_projects()) == 1

    try:
        projects.create_project("line a", "Duplicate with different casing")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("case-insensitive duplicate project name was accepted")


def test_project_context_uses_separate_sqlite_files(monkeypatch, tmp_path: Path) -> None:
    configure_catalog(monkeypatch, tmp_path)
    projects.initialize_catalog()
    first = projects.create_project("First", "First database")
    second = projects.create_project("Second", "Second database")

    assert first.database_url != second.database_url
    assert first.artifact_dir != second.artifact_dir

