from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base, get_db
from app.main import app

BASE = datetime(2026, 2, 1)


def setup():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine, autoflush=False)
    Base.metadata.create_all(engine)
    def override():
        with sessions() as db:
            yield db
    app.dependency_overrides[get_db] = override
    with sessions() as db:
        dataset = models.Dataset(name="raw", root_path="/workspace-eval", status="ready")
        db.add(dataset); db.flush()
        folder = models.DatasetFolder(dataset_id=dataset.id, relative_path=".", image_count=120)
        db.add(folder); db.flush()
        testing_dataset = models.TrainingDataset(dataset_id=dataset.id, name="test-layout-dataset", usage_label="test")
        db.add(testing_dataset); db.flush()
        db.add(models.TrainingDatasetRule(training_dataset_id=testing_dataset.id, folder_id=folder.id, start_timestamp=BASE, end_timestamp=BASE + timedelta(seconds=119), stride=1))
        training = models.TrainingRun(
            training_pipeline_id=999, status="finished", ended_at=BASE,
            artifact_kind="mean_image", artifact_path="/tmp/workspace.npy", artifact_signature="b" * 64,
            training_pipeline_name="Model B", method_type="mean_image", method_family="baseline",
            training_mode="fit", builder_kind="mean_image", preprocessing_pipeline_name="identity",
            dataset_names=["train"], dataset_names_text="train", training_parameters={},
        )
        db.add(training); db.flush()
        run = models.TestingRun(
            name="Inference", training_run_id=training.id, training_dataset_id=testing_dataset.id,
            status="finished", ended_at=BASE + timedelta(minutes=3), training_run_name="Model B",
            training_pipeline_name="Model B", training_dataset_name=testing_dataset.name,
            preprocessing_pipeline_name="identity", method_type="mean_image", method_family="baseline",
            training_mode="fit", artifact_kind="mean_image", artifact_path="/tmp/workspace.npy",
            artifact_signature="b" * 64, inference_config={}, result_revision=1,
        )
        db.add(run); db.flush()
        for position in range(120):
            value = float(position % 5) / 10.0
            if 70 <= position <= 80:
                value += 5.0
            db.add(models.TestingRunResult(testing_run_id=run.id, position=position,
                image_path=f"/workspace/{position}.tif", timestamp=BASE + timedelta(seconds=position),
                score=value, full_mse=value, width=8, height=8))
        ids = training.id, testing_dataset.id, run.id
        db.commit()
    return TestClient(app), sessions, ids


def iso(seconds: int) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def test_model_summary_and_separation_lifecycle_are_model_centred_and_idempotent():
    client, _sessions, (training_id, dataset_id, run_id) = setup()
    try:
        models_response = client.get("/api/evaluation-workspaces/models")
        assert models_response.status_code == 200, models_response.text
        summary = models_response.json()[0]
        assert summary["training_run_id"] == training_id
        assert [summary[key] for key in ("sep_median", "sep_min", "d_mean", "d_max")] == [None] * 4

        layout = client.post("/api/evaluation-workspaces/separation-layouts", json={
            "training_dataset_id": dataset_id, "name": "Known event",
            "pairs": [{"pair_key": "event-1", "name": "Fault", "normal_start": iso(0),
                       "normal_end": iso(20), "anomaly_start": iso(70), "anomaly_end": iso(80)}],
        })
        assert layout.status_code == 200, layout.text
        payload = {"testing_run_id": run_id, "layout_id": layout.json()["id"], "pair_keys": ["event-1"], "score_series": "score"}
        first = client.post(f"/api/evaluation-workspaces/models/{training_id}/separation/calculate", json=payload)
        assert first.status_code == 200, first.text
        assert first.json()["sep_median"] is not None
        second = client.post(f"/api/evaluation-workspaces/models/{training_id}/separation/calculate", json=payload)
        assert second.status_code == 200, second.text
        results = client.get(f"/api/evaluation-workspaces/models/{training_id}/separation/results").json()
        assert len(results) == 1
        assert results[0]["normal_point_count"] == 20
        assert results[0]["anomaly_point_count"] == 11  # inclusive anomaly end
        excluded = client.patch(f"/api/evaluation-workspaces/models/{training_id}/separation/results/{results[0]['id']}", json={"included": False})
        assert excluded.json()["sep_median"] is None
        deleted = client.delete(f"/api/evaluation-workspaces/models/{training_id}/separation/results/{results[0]['id']}")
        assert deleted.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_drift_requires_conflict_decisions_and_keeps_activatable_history():
    client, _sessions, (training_id, dataset_id, run_id) = setup()
    try:
        layout_payload = {
            "training_dataset_id": dataset_id, "name": "Daily stability",
            "reference_start": iso(0), "reference_end": iso(20),
            "analysis_start": iso(20), "analysis_end": iso(105), "bucket_seconds": 20,
            "reference_exclusion_action": "filter_points",
            "exclusions": [{"exclusion_key": "maintenance", "name": "Maintenance", "start_timestamp": iso(42), "end_timestamp": iso(44)}],
            "buckets": [],
        }
        preview = client.post(f"/api/evaluation-workspaces/models/{training_id}/drift/preview", json={"testing_run_id": run_id, "score_series": "score", "layout": layout_payload})
        assert preview.status_code == 200, preview.text
        rows = preview.json()["buckets"]
        assert any(row["status"] == "conflict" for row in rows)
        assert rows[-1]["reason"] == "incomplete remainder"
        layout_payload["buckets"] = [{"bucket_key": row["bucket_key"], "start_timestamp": row["start_timestamp"], "end_timestamp": row["end_timestamp"], "decision": "filter_points" if row["status"] == "conflict" else row["decision"]} for row in rows]
        layout = client.post("/api/evaluation-workspaces/drift-layouts", json=layout_payload)
        assert layout.status_code == 200, layout.text
        calculated = client.post(f"/api/evaluation-workspaces/models/{training_id}/drift/calculate", json={"testing_run_id": run_id, "layout_id": layout.json()["id"], "score_series": "score"})
        assert calculated.status_code == 200, calculated.text
        assert calculated.json()["d_mean"] is not None
        client.post(f"/api/evaluation-workspaces/models/{training_id}/drift/calculate", json={"testing_run_id": run_id, "layout_id": layout.json()["id"], "score_series": "score"})
        history = client.get(f"/api/evaluation-workspaces/models/{training_id}/drift/calculations").json()
        assert len(history) == 2
        assert sum(item["active"] for item in history) == 1
        activated = client.post(f"/api/evaluation-workspaces/models/{training_id}/drift/calculations/{history[1]['id']}/activate")
        assert activated.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_result_revision_marks_workspace_results_stale_without_raw_table_scan():
    client, sessions, (training_id, dataset_id, run_id) = setup()
    try:
        layout = client.post("/api/evaluation-workspaces/separation-layouts", json={
            "training_dataset_id": dataset_id, "name": "Stale", "pairs": [{"pair_key": "x", "name": "x",
            "normal_start": iso(0), "normal_end": iso(20), "anomaly_start": iso(70), "anomaly_end": iso(80)}]}).json()
        client.post(f"/api/evaluation-workspaces/models/{training_id}/separation/calculate", json={"testing_run_id": run_id, "layout_id": layout["id"], "pair_keys": ["x"], "score_series": "score"})
        with sessions() as db:
            run = db.get(models.TestingRun, run_id); run.result_revision += 1; db.commit()
        summary = client.get(f"/api/evaluation-workspaces/models/{training_id}").json()
        assert summary["sep_median"] is None
        assert client.get(f"/api/evaluation-workspaces/models/{training_id}/separation/results").json()[0]["stale"] is True
    finally:
        app.dependency_overrides.clear()
