from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import services
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


def add_scanned_dataset(
    client: TestClient,
    root: Path,
    name: str,
    folder: str,
    count: int,
    size: tuple[int, int] = (12, 8),
) -> dict:
    for index in range(count):
        timestamp = datetime(2026, 2, 4, 15, 30, 0) + timedelta(seconds=index * 10)
        write_tiff(root / f"{folder}_frame_{timestamp:%Y%m%d_%H%M%S}.tif", size=size)

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
    assert dataset["folders"][0]["filename_template"]["prefix"] == f"{folder}_frame_"
    assert dataset["folders"][0]["filename_template"]["suffix"] == ".tif"
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


def test_dataset_connection_probe_and_delete(tmp_path: Path) -> None:
    probe_root = tmp_path / "probe_dataset"
    write_tiff(probe_root / "frame_without_timestamp.tif")

    client_iter = make_client()
    client = next(client_iter)
    try:
        probe = client.post("/api/datasets/test-connection", json={"root_path": str(probe_root)})
        assert probe.status_code == 200
        probe_body = probe.json()
        assert probe_body["exists"] is True
        assert probe_body["is_directory"] is True
        assert probe_body["supported_file_found"] is True
        assert probe_body["sample_file_path"].endswith("frame_without_timestamp.tif")

        missing = client.post("/api/datasets/test-connection", json={"root_path": str(tmp_path / "missing")})
        assert missing.status_code == 200
        assert missing.json()["supported_file_found"] is False
        assert missing.json()["exists"] is False

        dataset = add_scanned_dataset(client, tmp_path / "dataset_to_delete", "Delete me", "line_01", 2)
        deleted = client.delete(f"/api/datasets/{dataset['id']}")
        assert deleted.status_code == 204

        listed = client.get("/api/datasets")
        assert listed.status_code == 200
        assert listed.json() == []
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_dataset_connection_probe_stops_at_first_supported_file(tmp_path: Path) -> None:
    probe_root = tmp_path / "probe_fast"
    write_tiff(probe_root / "first.tif")
    for index in range(200):
        (probe_root / f"line_{index + 1:03d}").mkdir(parents=True)

    client_iter = make_client()
    client = next(client_iter)
    try:
        probe = client.post("/api/datasets/test-connection", json={"root_path": str(probe_root)})
        assert probe.status_code == 200
        assert probe.json()["sample_file_path"].endswith("first.tif")
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_dataset_connection_probe_reports_limit(tmp_path: Path) -> None:
    probe_root = tmp_path / "probe_limited"
    for index in range(550):
        (probe_root / f"note_{index:03d}.txt").parent.mkdir(parents=True, exist_ok=True)
        (probe_root / f"note_{index:03d}.txt").write_text("not an image")

    client_iter = make_client()
    client = next(client_iter)
    try:
        probe = client.post("/api/datasets/test-connection", json={"root_path": str(probe_root)})
        assert probe.status_code == 200
        body = probe.json()
        assert body["supported_file_found"] is False
        assert "fast probe limit" in body["message"]
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_training_dataset_can_span_datasets_and_be_deleted(tmp_path: Path, monkeypatch) -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        dataset_a = add_scanned_dataset(client, tmp_path / "dataset_a", "A", "0226", 5)
        dataset_b = add_scanned_dataset(client, tmp_path / "dataset_b", "B", "line_01", 4)

        created = client.post(
            "/api/training-datasets",
            json={
                "name": "Combined",
                "usage_label": "test",
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
        assert training_dataset["usage_label"] == "test"
        assert training_dataset["dataset_names"] == ["A", "B"]
        assert training_dataset["image_signatures"] == ["12x8 | .tif | uint8 | 1ch | L"]
        assert training_dataset["total_matching_images"] == 9
        assert training_dataset["total_selected_images"] == 7
        assert training_dataset["start_timestamp"] == "2026-02-04T15:30:00"
        assert training_dataset["end_timestamp"] == "2026-02-04T15:30:40"
        assert len(training_dataset["rules"]) == 2
        assert sorted((rule["matching_images"], rule["selected_images"]) for rule in training_dataset["rules"]) == [
            (4, 4),
            (5, 3),
        ]

        def fail_if_counting_from_files(*_args, **_kwargs):
            raise AssertionError("Saved Train/Test Dataset reads must use stored counts.")

        monkeypatch.setattr(services, "count_folder_range_images", fail_if_counting_from_files)

        details = client.get(f"/api/training-datasets/{training_dataset['id']}")
        assert details.status_code == 200
        assert len(details.json()["rules"]) == 2
        assert details.json()["counts_missing"] is False
        assert {
            (rule["dataset_name"], rule["folder_relative_path"], rule["start_timestamp"], rule["end_timestamp"], rule["stride"])
            for rule in details.json()["rules"]
        } == {
            ("A", ".", "2026-02-04T15:30:00", "2026-02-04T15:30:40", 2),
            ("B", ".", "2026-02-04T15:30:00", "2026-02-04T15:30:30", 1),
        }

        listed_before_delete = client.get("/api/training-datasets")
        assert listed_before_delete.status_code == 200
        assert listed_before_delete.json()[0]["total_selected_images"] == 7

        blocked_dataset_delete = client.delete(f"/api/datasets/{dataset_a['id']}")
        assert blocked_dataset_delete.status_code == 409
        assert "saved train/test datasets" in blocked_dataset_delete.json()["detail"]

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


def test_training_dataset_rejects_mixed_image_signatures(tmp_path: Path) -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        dataset_a = add_scanned_dataset(client, tmp_path / "dataset_a", "A", "0226", 3, size=(12, 8))
        dataset_b = add_scanned_dataset(client, tmp_path / "dataset_b", "B", "line_01", 3, size=(16, 8))

        response = client.post(
            "/api/training-datasets",
            json={
                "name": "Mixed sizes",
                "usage_label": "mixed",
                "rules": [
                    {
                        "folder_id": dataset_a["folders"][0]["id"],
                        "start_timestamp": "2026-02-04T15:30:00",
                        "end_timestamp": "2026-02-04T15:30:20",
                        "stride": 1,
                    },
                    {
                        "folder_id": dataset_b["folders"][0]["id"],
                        "start_timestamp": "2026-02-04T15:30:00",
                        "end_timestamp": "2026-02-04T15:30:20",
                        "stride": 1,
                    },
                ],
            },
        )
        assert response.status_code == 400
        assert "same image data signature" in response.json()["detail"]
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_preprocessing_pipeline_crud_and_preview(tmp_path: Path) -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        dataset = add_scanned_dataset(client, tmp_path / "dataset_a", "A", "0226", 3)
        folder_id = dataset["folders"][0]["id"]
        graph = {
            "nodes": [
                {"id": "load", "type": "load_image", "config": {}, "position": {"x": 0, "y": 0}},
                {
                    "id": "resize",
                    "type": "resize",
                    "config": {"width": 8, "height": 4},
                    "position": {"x": 220, "y": 0},
                },
            ],
            "edges": [{"id": "load-resize", "source": "load", "target": "resize"}],
        }

        steps = client.get("/api/preprocessing/steps")
        assert steps.status_code == 200
        assert "warp_perspective" in {step["type"] for step in steps.json()}

        created = client.post(
            "/api/preprocessing/pipelines",
            json={
                "name": "Resize preview",
                "description": "test",
                "graph": graph,
                "preview_folder_id": folder_id,
                "input_width": 20,
                "input_height": 10,
                "output_width": 8,
                "output_height": 4,
            },
        )
        assert created.status_code == 200
        pipeline = created.json()
        assert pipeline["name"] == "Resize preview"
        assert pipeline["preview_folder_id"] == folder_id
        assert pipeline["input_width"] == 20
        assert pipeline["input_height"] == 10
        assert pipeline["output_width"] == 8
        assert pipeline["output_height"] == 4

        listed = client.get("/api/preprocessing/pipelines")
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        assert listed.json()[0]["input_width"] == 20
        assert listed.json()[0]["output_height"] == 4

        loaded = client.get(f"/api/preprocessing/pipelines/{pipeline['id']}")
        assert loaded.status_code == 200
        assert loaded.json()["graph"]["nodes"][0]["type"] == "load_image"
        assert loaded.json()["preview_folder_id"] == folder_id
        assert loaded.json()["input_width"] == 20
        assert loaded.json()["output_height"] == 4

        preview = client.post(
            "/api/preprocessing/pipelines/preview",
            json={"folder_id": folder_id, "graph": graph},
        )
        assert preview.status_code == 200
        body = preview.json()
        assert len(body["previews"]) == 2
        assert body["previews"][1]["width"] == 8
        assert body["previews"][1]["height"] == 4

        invalid = client.post(
            "/api/preprocessing/pipelines",
            json={
                "name": "Invalid",
                "graph": {
                    "nodes": [{"id": "resize", "type": "resize", "config": {}}],
                    "edges": [],
                },
            },
        )
        assert invalid.status_code == 400

        deleted = client.delete(f"/api/preprocessing/pipelines/{pipeline['id']}")
        assert deleted.status_code == 204
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_preprocessing_pipeline_update_and_unique_name() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        graph = {
            "nodes": [
                {"id": "load", "type": "load_image", "config": {}, "position": {"x": 0, "y": 0}},
                {"id": "resize", "type": "resize", "config": {"width": 8, "height": 4}, "position": {"x": 220, "y": 0}},
            ],
            "edges": [{"id": "load-resize", "source": "load", "target": "resize"}],
        }

        created = client.post(
            "/api/preprocessing/pipelines",
            json={
                "name": "Pipe one",
                "graph": graph,
                "input_width": 20,
                "input_height": 10,
                "output_width": 8,
                "output_height": 4,
            },
        )
        assert created.status_code == 200
        pid = created.json()["id"]

        # Duplicate name (case-insensitive) is rejected.
        duplicate = client.post("/api/preprocessing/pipelines", json={"name": "pipe ONE", "graph": graph})
        assert duplicate.status_code == 400

        second = client.post("/api/preprocessing/pipelines", json={"name": "Pipe two", "graph": graph})
        assert second.status_code == 200

        # Update name + graph of the first pipeline.
        updated = client.put(
            f"/api/preprocessing/pipelines/{pid}",
            json={
                "name": "Pipe renamed",
                "graph": graph,
                "input_width": 40,
                "input_height": 30,
                "output_width": 8,
                "output_height": 4,
            },
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "Pipe renamed"
        assert updated.json()["input_width"] == 40
        assert updated.json()["input_height"] == 30
        assert updated.json()["output_width"] == 8
        assert updated.json()["output_height"] == 4

        # Renaming onto an existing name is rejected.
        clash = client.put(f"/api/preprocessing/pipelines/{pid}", json={"name": "Pipe two", "graph": graph})
        assert clash.status_code == 400

        # Updating a missing pipeline returns 404.
        missing = client.put("/api/preprocessing/pipelines/99999", json={"name": "X", "graph": graph})
        assert missing.status_code == 404
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_analysis_layout_crud_and_duplicate_name() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        payload = {
            "name": "Baseline board",
            "description": "AE comparison",
            "layout": {
                "version": 1,
                "plots": [{"id": "plot-1", "plotType": "timeseries", "sources": []}],
                "selectedPipelineId": "1",
            },
        }
        created = client.post("/api/analysis/layouts", json=payload)
        assert created.status_code == 200
        layout = created.json()
        assert layout["name"] == "Baseline board"
        assert layout["layout"]["plots"][0]["id"] == "plot-1"

        duplicate = client.post("/api/analysis/layouts", json={**payload, "name": "baseline BOARD"})
        assert duplicate.status_code == 409

        listed = client.get("/api/analysis/layouts")
        assert listed.status_code == 200
        assert [item["name"] for item in listed.json()] == ["Baseline board"]

        fetched = client.get(f"/api/analysis/layouts/{layout['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["description"] == "AE comparison"

        updated = client.put(
            f"/api/analysis/layouts/{layout['id']}",
            json={"name": "Updated board", "description": None, "layout": {"version": 1, "plots": []}},
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "Updated board"
        assert updated.json()["layout"]["plots"] == []

        deleted = client.delete(f"/api/analysis/layouts/{layout['id']}")
        assert deleted.status_code == 204
        assert client.get(f"/api/analysis/layouts/{layout['id']}").status_code == 404
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass
