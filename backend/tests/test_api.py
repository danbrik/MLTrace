from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


def write_tiff(path: Path, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=127).save(path)


def make_client():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db: Session = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def add_scanned_dataset(client: TestClient, root: Path, name: str, folder: str, count: int) -> dict:
    for index in range(count):
        timestamp = datetime(2026, 2, 4, 15, 30, 0) + timedelta(seconds=index * 10)
        write_tiff(root / folder / f"frame_{timestamp:%Y%m%d_%H%M%S}.tif")

    created = client.post(
        "/api/datasets",
        json={"name": name, "root_path": str(root)},
    )
    assert created.status_code == 200
    dataset = created.json()
    assert dataset["timestamp_format"] == "%Y%m%d_%H%M%S"

    scanned = client.post(
        f"/api/datasets/{dataset['id']}/confirm-timestamp-format",
        json={
            "timestamp_regex": r"(?P<timestamp>\d{8}_\d{6})",
            "timestamp_format": "%Y%m%d_%H%M%S",
        },
    )
    assert scanned.status_code == 200
    dataset = scanned.json()
    assert dataset["status"] == "ready"
    assert dataset["folders"][0]["image_count"] == count
    return dataset


def test_dataset_create_confirm_and_training_dataset_preview(tmp_path: Path) -> None:
    root = tmp_path / "dataset_a"

    client_iter = make_client()
    client = next(client_iter)
    try:
        dataset = add_scanned_dataset(client, root, "February", "0226", 5)
        folder_id = dataset["folders"][0]["id"]
        preview = client.post(
            "/api/training-datasets/preview",
            json={
                "rules": [
                    {
                        "folder_id": folder_id,
                        "start_timestamp": "2026-02-04T15:30:00",
                        "end_timestamp": "2026-02-04T15:30:40",
                        "stride": 2,
                    }
                ],
            },
        )
        assert preview.status_code == 200
        assert preview.json()["total_matching_images"] == 5
        assert preview.json()["total_selected_images"] == 3
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_training_dataset_can_span_datasets_and_be_deleted(tmp_path: Path) -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        dataset_a = add_scanned_dataset(client, tmp_path / "dataset_a", "A", "0226", 5)
        dataset_b = add_scanned_dataset(client, tmp_path / "dataset_b", "B", "line_01", 4)

        created = client.post(
            "/api/training-datasets",
            json={
                "name": "Combined",
                "notes": "Two roots",
                "rules": [
                    {
                        "folder_id": dataset_a["folders"][0]["id"],
                        "start_timestamp": "2026-02-04T15:30:00",
                        "end_timestamp": "2026-02-04T15:30:40",
                        "stride": 2,
                    },
                    {
                        "folder_id": dataset_b["folders"][0]["id"],
                        "start_timestamp": "2026-02-04T15:30:00",
                        "end_timestamp": "2026-02-04T15:30:30",
                        "stride": 1,
                    },
                ],
            },
        )
        assert created.status_code == 200
        training_dataset = created.json()
        assert training_dataset["dataset_names"] == ["A", "B"]
        assert training_dataset["total_matching_images"] == 9
        assert training_dataset["total_selected_images"] == 7
        assert len(training_dataset["rules"]) == 2

        details = client.get(f"/api/training-datasets/{training_dataset['id']}")
        assert details.status_code == 200
        assert len(details.json()["rules"]) == 2
        assert {
            (rule["dataset_name"], rule["folder_relative_path"], rule["start_timestamp"], rule["end_timestamp"], rule["stride"])
            for rule in details.json()["rules"]
        } == {
            ("A", "0226", "2026-02-04T15:30:00", "2026-02-04T15:30:40", 2),
            ("B", "line_01", "2026-02-04T15:30:00", "2026-02-04T15:30:30", 1),
        }

        deleted = client.delete(f"/api/training-datasets/{training_dataset['id']}")
        assert deleted.status_code == 204
        listed = client.get("/api/training-datasets")
        assert listed.status_code == 200
        assert listed.json() == []
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass
