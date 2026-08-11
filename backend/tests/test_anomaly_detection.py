from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.anomaly_detection.service import SignalPoint, detect
from app.anomaly_detection import service as anomaly_service
from app.database import Base, get_db
from app.main import app
from app.schemas import AnomalyDetectionConfig
from app.schemas import AnomalyDetectionRunCreate
from app.testing import service as testing_service


BASE = datetime(2026, 1, 1)


def _points(values: list[float], *, seconds: int = 60) -> list[SignalPoint]:
    return [SignalPoint(BASE + timedelta(seconds=index * seconds), value) for index, value in enumerate(values)]


def _fast_config(**overrides) -> AnomalyDetectionConfig:
    values = {
        "smoothing_half_life_minutes": 0.1,
        "baseline_window_minutes": 10,
        "warmup_minutes": 2,
        "minimum_warmup_points": 3,
        "warning_z": 3,
        "high_z": 5,
        "cusum_drift": 1,
        "cusum_threshold": 10,
        "confirmation_minutes": 2,
        "recovery_z": 1,
        "recovery_minutes": 2,
        "preroll_minutes": 10,
        "minimum_gap_minutes": 15,
    }
    values.update(overrides)
    return AnomalyDetectionConfig(**values)


def test_constant_signal_is_stable_when_mad_is_zero() -> None:
    output = detect(_points([1.0] * 40), _fast_config())
    assert output.events == []
    ready = [point for point in output.series if point.state != "warmup"]
    assert ready
    assert all(point.robust_z == 0 for point in ready)
    assert all(point.warning_threshold is not None for point in ready)


def test_brief_peak_remains_unconfirmed() -> None:
    values = [1.0] * 12 + [3.0] + [1.0] * 12
    output = detect(_points(values), _fast_config(confirmation_minutes=3))
    assert len(output.events) == 1
    assert output.events[0].confirmed_at is None
    assert output.events[0].end_reason == "recovered"


def test_robust_zscore_does_not_accumulate_cusum() -> None:
    values = [1.0] * 12 + [3.0] * 10 + [1.0] * 5
    output = detect(_points(values), _fast_config(algorithm="robust_zscore"))
    assert len(output.events) == 1
    assert output.events[0].confirmed_at is not None
    assert all(point.cusum == 0 for point in output.series)


def test_sustained_shift_confirms_and_frozen_baseline_does_not_follow() -> None:
    values = [1.0] * 12 + [3.0] * 10 + [1.0] * 5
    output = detect(_points(values), _fast_config())
    assert len(output.events) == 1
    event = output.events[0]
    assert event.confirmed_at is not None
    assert event.confirmed_at >= event.warning_start + timedelta(minutes=2)
    event_points = [point for point in output.series if event.warning_start <= point.timestamp <= event.end_timestamp]
    baselines = {point.baseline for point in event_points if point.baseline is not None}
    assert baselines == {1.0}
    assert event.end_reason == "recovered"


def test_large_gap_closes_active_event() -> None:
    points = _points([1.0] * 8 + [3.0] * 4)
    points.append(SignalPoint(points[-1].timestamp + timedelta(hours=1), 1.0))
    output = detect(points, _fast_config(confirmation_minutes=1))
    assert output.events
    assert output.events[0].end_reason == "data_gap"


def _seed_testing_run(db: Session, count: int = 80) -> models.TestingRun:
    run = models.TestingRun(
        name="Crucible inference",
        training_run_id=1,
        training_dataset_id=1,
        status="finished",
        training_run_name="model",
        training_pipeline_name="pipeline",
        training_dataset_name="dataset",
        preprocessing_pipeline_name="preprocessing",
        method_type="autoencoder",
        method_family="autoencoder",
        training_mode="gradient",
        artifact_kind="weights",
        artifact_path="/tmp/model.pt",
    )
    db.add(run)
    db.flush()
    rows = []
    for index in range(count):
        value = 3.0 if 35 <= index < 55 else 1.0
        rows.append({
            "testing_run_id": run.id,
            "position": index,
            "image_path": f"/tmp/{index}.tif",
            "timestamp": BASE + timedelta(minutes=index),
            "score": value,
            "full_mse": value,
            "roi_mse": None,
            "width": 8,
            "height": 8,
        })
    db.bulk_insert_mappings(models.TestingRunResult, rows)
    db.commit()
    return run


def test_anomaly_detection_api_crud_and_decimation() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        run = _seed_testing_run(db)
        testing_run_id = run.id

    def override_get_db():
        with SessionLocal() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            created = client.post("/api/anomaly-detection-runs", json={
                "name": "Sensitive detector",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _fast_config().model_dump(),
            })
            assert created.status_code == 200, created.text
            body = created.json()
            assert body["point_count"] == 80
            assert body["anomaly_count"] == 1
            assert body["algorithm_version"] == "robust_cusum_v1"
            assert body["config"]["algorithm"] == "robust_cusum"
            run_id = body["id"]

            listed = client.get("/api/anomaly-detection-runs")
            assert listed.status_code == 200
            assert [item["id"] for item in listed.json()] == [run_id]

            detail = client.get(f"/api/anomaly-detection-runs/{run_id}?max_points=20")
            assert detail.status_code == 200
            assert detail.json()["decimated"] is True
            assert len(detail.json()["series"]) <= 20
            assert detail.json()["events"][0]["confirmed_at"] is not None

            zscore = client.post("/api/anomaly-detection-runs", json={
                "name": "Robust z-score detector",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": BASE.isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _fast_config(algorithm="robust_zscore").model_dump(),
            })
            assert zscore.status_code == 200, zscore.text
            assert zscore.json()["algorithm_version"] == "robust_zscore_v1"
            assert zscore.json()["config"]["algorithm"] == "robust_zscore"
            assert all(point["cusum"] == 0 for point in zscore.json()["series"])
            assert client.delete(f"/api/anomaly-detection-runs/{zscore.json()['id']}").status_code == 204

            with_preroll = client.post("/api/anomaly-detection-runs", json={
                "name": "Selected range with pre-roll",
                "testing_run_id": testing_run_id,
                "score_series": "score",
                "start_timestamp": (BASE + timedelta(minutes=30)).isoformat(),
                "end_timestamp": (BASE + timedelta(minutes=79)).isoformat(),
                "config": _fast_config(preroll_minutes=30).model_dump(),
            })
            assert with_preroll.status_code == 200, with_preroll.text
            preroll_body = with_preroll.json()
            assert preroll_body["series"][0]["timestamp"].startswith("2026-01-01T00:30")
            assert preroll_body["series"][0]["state"] == "normal"
            assert all(not point["timestamp"].startswith("2026-01-01T00:29") for point in preroll_body["series"])
            assert client.delete(f"/api/anomaly-detection-runs/{preroll_body['id']}").status_code == 204

            deleted = client.delete(f"/api/anomaly-detection-runs/{run_id}")
            assert deleted.status_code == 204
            assert client.get(f"/api/anomaly-detection-runs/{run_id}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_restarting_inference_invalidates_saved_detections(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        testing_run = _seed_testing_run(db)
        detection = anomaly_service.create_run(db, AnomalyDetectionRunCreate(
            name="Derived detector",
            testing_run_id=testing_run.id,
            start_timestamp=BASE,
            end_timestamp=BASE + timedelta(minutes=79),
            config=_fast_config(),
        ))
        assert db.get(models.AnomalyDetectionRun, detection.id) is not None
        monkeypatch.setattr(testing_service.scheduler, "wake", lambda: None)
        restarted = testing_service.restart_testing_run(db, testing_run.id)
        assert restarted is not None
        assert restarted.status == "queued"
        assert db.get(models.AnomalyDetectionRun, detection.id) is None
