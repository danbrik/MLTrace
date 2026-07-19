from __future__ import annotations

import shutil
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings
from app.database import project_context


class CatalogBase(DeclarativeBase):
    pass


class Project(CatalogBase):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    name_key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    database_url: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_dir: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC).replace(tzinfo=None))
    last_opened_at: Mapped[datetime | None] = mapped_column(DateTime)


class SchedulerQueueEntry(CatalogBase):
    __tablename__ = "scheduler_queue"
    __table_args__ = (UniqueConstraint("project_id", "kind", "run_id", name="uq_catalog_scheduler_job"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    queue_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())


def _root_dir() -> Path:
    url = make_url(get_settings().database_url)
    if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
        return Path(url.database).expanduser().resolve().parent
    return Path(".mltrace").resolve()


ROOT_DIR = _root_dir()
CATALOG_PATH = ROOT_DIR / "catalog.db"
_catalog_engine = create_engine(f"sqlite:///{CATALOG_PATH}", connect_args={"check_same_thread": False})
CatalogSession = sessionmaker(autocommit=False, autoflush=False, bind=_catalog_engine)
_migration_lock = threading.Lock()


def serialize_project(project: Project) -> dict:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at,
        "last_opened_at": project.last_opened_at,
    }


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve()}"


def migrate_project(database_url: str) -> None:
    with _migration_lock:
        config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
        config.set_main_option("script_location", str(Path(__file__).resolve().parents[1] / "alembic"))
        config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
        command.upgrade(config, "head")


def initialize_catalog() -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    CatalogBase.metadata.create_all(_catalog_engine)
    with CatalogSession() as db:
        if (db.scalar(select(func.count()).select_from(Project)) or 0) == 0:
            legacy_url = get_settings().database_url
            legacy = make_url(legacy_url)
            if legacy.drivername.startswith("sqlite") and legacy.database and Path(legacy.database).expanduser().exists():
                db.add(Project(
                    id=str(uuid.uuid4()),
                    name="Default Project",
                    name_key="default project",
                    description="Automatically imported existing MLTrace project.",
                    database_url=_sqlite_url(Path(legacy.database).expanduser()),
                    artifact_dir=str(Path(legacy.database).expanduser().resolve().parent),
                ))
                db.commit()


def list_projects() -> list[Project]:
    with CatalogSession() as db:
        return list(db.scalars(select(Project).order_by(Project.last_opened_at.desc(), Project.created_at.desc())))


def get_project(project_id: str) -> Project | None:
    with CatalogSession() as db:
        project = db.get(Project, project_id)
        if project is not None:
            db.expunge(project)
        return project


def create_project(name: str, description: str) -> Project:
    clean_name = name.strip()
    clean_description = description.strip()
    if not clean_name:
        raise ValueError("Project name is required.")
    if len(clean_name) > 100:
        raise ValueError("Project name must not exceed 100 characters.")
    if not clean_description:
        raise ValueError("Project description is required.")
    if len(clean_description) > 500:
        raise ValueError("Project description must not exceed 500 characters.")
    with CatalogSession() as db:
        if db.scalar(select(Project.id).where(Project.name_key == clean_name.casefold())) is not None:
            raise ValueError("A project with this name already exists.")
    project_id = str(uuid.uuid4())
    project_dir = ROOT_DIR / "projects" / project_id
    database_url = _sqlite_url(project_dir / "mltrace.db")
    try:
        project_dir.mkdir(parents=True, exist_ok=False)
        migrate_project(database_url)
        from app.modeling.defaults import ensure_default_method_configurations
        from app.database import SessionLocal
        with project_context(database_url, project_dir):
            db = SessionLocal()
            try:
                ensure_default_method_configurations(db)
            finally:
                db.close()
        project = Project(
            id=project_id,
            name=clean_name,
            name_key=clean_name.casefold(),
            description=clean_description,
            database_url=database_url,
            artifact_dir=str(project_dir.resolve()),
        )
        with CatalogSession() as db:
            db.add(project)
            db.commit()
            db.refresh(project)
            db.expunge(project)
        return project
    except IntegrityError as exc:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise ValueError("A project with this name already exists.") from exc
    except Exception:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise


def mark_project_opened(project_id: str) -> Project | None:
    with CatalogSession() as db:
        project = db.get(Project, project_id)
        if project is None:
            return None
        project.last_opened_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        db.refresh(project)
        db.expunge(project)
        return project


def migrate_all_projects() -> None:
    for project in list_projects():
        migrate_project(project.database_url)


def ensure_queue_entry(project_id: str, kind: str, run_id: int, enqueued_at: datetime | None = None) -> None:
    with CatalogSession() as db:
        existing = db.scalar(select(SchedulerQueueEntry).where(
            SchedulerQueueEntry.project_id == project_id,
            SchedulerQueueEntry.kind == kind,
            SchedulerQueueEntry.run_id == run_id,
        ))
        if existing is not None:
            return
        next_rank = (db.scalar(select(func.max(SchedulerQueueEntry.queue_rank))) or 0) + 1
        db.add(SchedulerQueueEntry(
            project_id=project_id,
            kind=kind,
            run_id=run_id,
            queue_rank=next_rank,
            enqueued_at=enqueued_at or datetime.now(UTC).replace(tzinfo=None),
        ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()


def list_queue_entries() -> list[SchedulerQueueEntry]:
    with CatalogSession() as db:
        entries = list(db.scalars(select(SchedulerQueueEntry).order_by(
            SchedulerQueueEntry.queue_rank, SchedulerQueueEntry.enqueued_at, SchedulerQueueEntry.id
        )))
        for entry in entries:
            db.expunge(entry)
        return entries


def remove_queue_entry(project_id: str, kind: str, run_id: int) -> None:
    with CatalogSession() as db:
        entry = db.scalar(select(SchedulerQueueEntry).where(
            SchedulerQueueEntry.project_id == project_id,
            SchedulerQueueEntry.kind == kind,
            SchedulerQueueEntry.run_id == run_id,
        ))
        if entry is not None:
            db.delete(entry)
            db.commit()


def move_queue_entry(project_id: str, kind: str, run_id: int, direction: str) -> int:
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'.")
    with CatalogSession() as db:
        entries = list(db.scalars(select(SchedulerQueueEntry).order_by(
            SchedulerQueueEntry.queue_rank, SchedulerQueueEntry.enqueued_at, SchedulerQueueEntry.id
        )))
        index = next((i for i, item in enumerate(entries) if (
            item.project_id, item.kind, item.run_id
        ) == (project_id, kind, run_id)), -1)
        if index < 0:
            raise ValueError("Queued scheduler job was not found in the global queue.")
        target = index - 1 if direction == "up" else index + 1
        if target < 0 or target >= len(entries):
            raise ValueError("Scheduler job is already at the queue boundary.")
        entries[index].queue_rank, entries[target].queue_rank = entries[target].queue_rank, entries[index].queue_rank
        db.commit()
        return entries[index].queue_rank
