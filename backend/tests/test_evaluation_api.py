from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base, get_db
from app.main import app


BASE = datetime(2026, 1, 1)


def _client_and_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app), session_factory


def _seed_sources(session_factory) -> tuple[int, int, int, int]:
    with session_factory() as db:
        dataset = models.Dataset(
            name="D",
            root_path="/evaluation-api",
            status="ready",
        )
        db.add(dataset)
        db.flush()
        folder = models.DatasetFolder(
            dataset_id=dataset.id,
            relative_path=".",
            image_count=100,
        )
        db.add(folder)
        db.flush()
        training_datasets = []
        for name in ("evaluation", "reference", "calibration"):
            training_dataset = models.TrainingDataset(
                dataset_id=dataset.id,
                name=name,
                usage_label="test",
            )
            db.add(training_dataset)
            db.flush()
            db.add(
                models.TrainingDatasetRule(
                    training_dataset_id=training_dataset.id,
                    folder_id=folder.id,
                    start_timestamp=BASE,
                    end_timestamp=BASE + timedelta(seconds=99),
                    stride=1,
                )
            )
            training_datasets.append(training_dataset)

        run_ids = []
        for role_index, training_dataset in enumerate(training_datasets):
            run = models.TestingRun(
                name=training_dataset.name,
                training_run_id=999,
                training_dataset_id=training_dataset.id,
                status="finished",
                training_run_name="model",
                training_pipeline_name="pipeline",
                training_dataset_name=training_dataset.name,
                preprocessing_pipeline_name="preprocessing",
                method_type="mean_image",
                method_family="baseline",
                training_mode="fit",
                artifact_kind="mean_image",
                artifact_path="/tmp/model.npy",
                artifact_signature="a" * 64,
                inference_config={"score_aggregation": "mean"},
            )
            db.add(run)
            db.flush()
            run_ids.append(run.id)
            mappings = []
            for position in range(100):
                if role_index == 0:
                    value = 10.0 if 70 <= position <= 80 else 0.0
                elif role_index == 1:
                    value = (position % 5) / 100.0
                else:
                    value = (position % 10) / 100.0
                mappings.append(
                    {
                        "testing_run_id": run.id,
                        "position": position,
                        "image_path": f"/{training_dataset.name}/{position}.tif",
                        "timestamp": BASE + timedelta(seconds=position),
                        "score": value,
                        "full_mse": value,
                        "roi_mse": value,
                        "width": 8,
                        "height": 8,
                    }
                )
            db.bulk_insert_mappings(models.TestingRunResult, mappings)
        db.commit()
        return training_datasets[0].id, run_ids[0], run_ids[1], run_ids[2]


def _profile(client: TestClient) -> dict:
    response = client.post(
        "/api/evaluation-profiles",
        json={
            "name": "Standard",
            "normal_window_duration_seconds": 20,
            "normal_window_buffer_seconds": 0,
            "drift_window_seconds": 20,
            "false_alarm_horizon_seconds": 60,
            "anticipation_seconds": 0,
            "epsilon": 1e-12,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _labels(client: TestClient, training_dataset_id: int) -> dict:
    response = client.post(
        "/api/evaluation-label-sets",
        json={
            "training_dataset_id": training_dataset_id,
            "name": "Ground truth",
            "events": [
                {
                    "event_id": "event-1",
                    "type": "target",
                    "name": "Event",
                    "category": "fault",
                    "start_timestamp": (BASE + timedelta(seconds=70)).isoformat(),
                    "end_timestamp": (BASE + timedelta(seconds=80)).isoformat(),
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _evaluation(
    client: TestClient,
    profile_id: int,
    label_set_id: int,
    evaluation_run_id: int,
    reference_run_id: int,
    calibration_run_id: int,
) -> dict:
    response = client.post(
        "/api/evaluations",
        json={
            "name": "Single model",
            "evaluation_testing_run_id": evaluation_run_id,
            "reference_testing_run_id": reference_run_id,
            "calibration_testing_run_id": calibration_run_id,
            "profile_id": profile_id,
            "label_set_id": label_set_id,
            "score_series": "score",
            "evaluation_start_timestamp": BASE.isoformat(),
            "evaluation_end_timestamp": (BASE + timedelta(seconds=99)).isoformat(),
            "reference_start_timestamp": BASE.isoformat(),
            "reference_end_timestamp": (BASE + timedelta(seconds=99)).isoformat(),
            "calibration_start_timestamp": BASE.isoformat(),
            "calibration_end_timestamp": (BASE + timedelta(seconds=99)).isoformat(),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_evaluation_api_calculates_stages_independently_and_finalizes() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)
        evaluation = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )

        # Stages are intentionally callable in any order.
        for stage in ("detection", "separation", "drift"):
            response = client.post(
                f"/api/evaluations/{evaluation['id']}/calculate/{stage}"
            )
            assert response.status_code == 200, response.text
            evaluation = response.json()
            assert evaluation[f"{stage}_status"] == "current"

        assert evaluation["sep_median"] is not None
        assert evaluation["sep_min"] is not None
        assert evaluation["drift_mean"] is not None
        assert evaluation["drift_max"] is not None
        assert evaluation["event_recall"] == 1.0
        assert evaluation["median_delay_seconds"] == 0.0
        assert evaluation["frame_fpr"] is not None
        assert evaluation["false_alarm_rate_t0"] is not None
        assert len(evaluation["detection_result"]["operating_points"]) == 5

        changed = client.patch(
            f"/api/evaluations/{evaluation['id']}",
            json={"reference_start_timestamp": (BASE + timedelta(seconds=1)).isoformat()},
        )
        assert changed.status_code == 200, changed.text
        changed_body = changed.json()
        assert changed_body["separation_status"] == "current"
        assert changed_body["drift_status"] == "stale"
        assert changed_body["detection_status"] == "current"

        recalculated = client.post(
            f"/api/evaluations/{evaluation['id']}/calculate/drift"
        )
        assert recalculated.status_code == 200, recalculated.text
        finalized = client.post(f"/api/evaluations/{evaluation['id']}/finalize")
        assert finalized.status_code == 200, finalized.text
        assert finalized.json()["status"] == "finalized"
        assert client.patch(
            f"/api/evaluations/{evaluation['id']}", json={"name": "Changed"}
        ).status_code == 409
        assert client.delete(f"/api/evaluation-profiles/{profile['id']}").status_code == 409
        assert client.delete(f"/api/evaluation-label-sets/{labels['id']}").status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_label_csv_preview_overlap_and_score_preview_contract() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(session_factory)
        aware_labels = client.post(
            "/api/evaluation-label-sets",
            json={
                "training_dataset_id": dataset_id,
                "name": "Aware wall clock",
                "events": [
                    {
                        "event_id": "aware-event",
                        "type": "target",
                        "name": "Aware event",
                        "category": "fault",
                        "start_timestamp": "2026-01-01T00:01:10+02:00",
                        "end_timestamp": "2026-01-01T00:01:20",
                    }
                ],
            },
        )
        assert aware_labels.status_code == 200, aware_labels.text
        assert aware_labels.json()["events"][0]["start_timestamp"] == "2026-01-01T00:01:10"
        assert aware_labels.json()["events"][0]["end_timestamp"] == "2026-01-01T00:01:20"
        invalid_mixed_event = client.post(
            "/api/evaluation-label-sets",
            json={
                "training_dataset_id": dataset_id,
                "name": "Invalid mixed wall clock",
                "events": [
                    {
                        "event_id": "invalid-mixed",
                        "type": "target",
                        "name": "Invalid",
                        "category": "fault",
                        "start_timestamp": "2026-01-01T00:00:20+02:00",
                        "end_timestamp": "2026-01-01T00:00:10",
                    }
                ],
            },
        )
        assert invalid_mixed_event.status_code == 422, invalid_mixed_event.text
        blank_event_id = client.post(
            "/api/evaluation-label-sets",
            json={
                "training_dataset_id": dataset_id,
                "name": "Blank id",
                "events": [
                    {
                        "event_id": "   ",
                        "type": "target",
                        "name": "Blank id",
                        "category": "fault",
                        "start_timestamp": "2026-01-01T00:00:10",
                        "end_timestamp": "2026-01-01T00:00:20",
                    }
                ],
            },
        )
        assert blank_event_id.status_code == 400, blank_event_id.text
        assert "event_id must not be blank" in blank_event_id.json()["detail"]

        profile = _profile(client)
        touching = client.post(
            "/api/evaluations",
            json={
                "name": "Touching same-run roles",
                "evaluation_testing_run_id": evaluation_run,
                "reference_testing_run_id": evaluation_run,
                "calibration_testing_run_id": calibration_run,
                "profile_id": profile["id"],
                "label_set_id": aware_labels.json()["id"],
                "evaluation_start_timestamp": BASE.isoformat(),
                "evaluation_end_timestamp": (BASE + timedelta(seconds=50)).isoformat(),
                "reference_start_timestamp": (BASE + timedelta(seconds=50)).isoformat(),
                "reference_end_timestamp": (BASE + timedelta(seconds=99)).isoformat(),
                "calibration_start_timestamp": BASE.isoformat(),
                "calibration_end_timestamp": (BASE + timedelta(seconds=99)).isoformat(),
            },
        )
        assert touching.status_code == 400
        assert "must be disjoint" in touching.json()["detail"]
        normalized_evaluation = client.post(
            "/api/evaluations",
            json={
                "name": "Aware ranges",
                "evaluation_testing_run_id": evaluation_run,
                "reference_testing_run_id": reference_run,
                "calibration_testing_run_id": calibration_run,
                "profile_id": profile["id"],
                "label_set_id": aware_labels.json()["id"],
                "evaluation_start_timestamp": "2026-01-01T00:00:00+02:00",
                "evaluation_end_timestamp": "2026-01-01T00:01:39",
                "reference_start_timestamp": "2026-01-01T00:00:00+02:00",
                "reference_end_timestamp": "2026-01-01T00:01:39+02:00",
                "calibration_start_timestamp": "2026-01-01T00:00:00",
                "calibration_end_timestamp": "2026-01-01T00:01:39+02:00",
            },
        )
        assert normalized_evaluation.status_code == 200, normalized_evaluation.text
        assert normalized_evaluation.json()["evaluation_start_timestamp"] == BASE.isoformat()
        assert normalized_evaluation.json()["evaluation_end_timestamp"] == (
            BASE + timedelta(seconds=99)
        ).isoformat()
        invalid_mixed_range = client.post(
            "/api/evaluations",
            json={
                "name": "Invalid mixed range",
                "evaluation_start_timestamp": "2026-01-01T00:00:20+02:00",
                "evaluation_end_timestamp": "2026-01-01T00:00:10",
            },
        )
        assert invalid_mixed_range.status_code == 422, invalid_mixed_range.text
        csv_text = (
            "event_id,type,name,category,start_timestamp,end_timestamp,notes\n"
            "a,target,A,fault,2026-01-01T00:00:10,2026-01-01T00:00:20,\n"
            "b,target,B,fault,2026-01-01T00:00:20,2026-01-01T00:00:30,\n"
        )
        invalid = client.post(
            "/api/evaluation-label-sets/csv-preview",
            json={"training_dataset_id": dataset_id, "csv_text": csv_text},
        )
        assert invalid.status_code == 200
        assert invalid.json()["valid"] is False
        assert "overlap" in invalid.json()["errors"][0]["message"].lower()

        preview = client.get(
            f"/api/testing-runs/{evaluation_run}/evaluation-score-preview",
            params={"score_series": "score", "max_points": 100},
        )
        assert preview.status_code == 200, preview.text
        body = preview.json()
        assert body["start_timestamp"] == BASE.isoformat()
        assert body["end_timestamp"] == (BASE + timedelta(seconds=99)).isoformat()
        assert body["total"] == 100
        assert body["points"][0] == {
            "result_id": body["points"][0]["result_id"],
            "position": 0,
            "timestamp": BASE.isoformat(),
            "value": 0.0,
            "continuity_segment": 0,
        }
        aware = client.get(
            f"/api/testing-runs/{evaluation_run}/evaluation-score-preview",
            params={
                "score_series": "score",
                "start_timestamp": "2026-01-01T00:00:00+02:00",
                "end_timestamp": "2026-01-01T00:01:39+02:00",
            },
        )
        assert aware.status_code == 200, aware.text
        assert aware.json()["start_timestamp"] == BASE.isoformat()
        assert aware.json()["end_timestamp"] == (BASE + timedelta(seconds=99)).isoformat()
    finally:
        app.dependency_overrides.clear()


def test_testing_run_revision_marks_only_dependent_stages_stale() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)
        evaluation = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )
        for stage in ("separation", "drift", "detection"):
            response = client.post(
                f"/api/evaluations/{evaluation['id']}/calculate/{stage}"
            )
            assert response.status_code == 200, response.text

        with session_factory() as db:
            reference = db.get(models.TestingRun, reference_run)
            assert reference is not None
            reference.updated_at = reference.updated_at + timedelta(seconds=1)
            db.commit()

        refreshed = client.get(f"/api/evaluations/{evaluation['id']}")
        assert refreshed.status_code == 200
        assert refreshed.json()["separation_status"] == "current"
        assert refreshed.json()["drift_status"] == "stale"
        assert refreshed.json()["detection_status"] == "current"
    finally:
        app.dependency_overrides.clear()


def test_evaluation_cache_revision_tracks_testing_runs_and_results() -> None:
    client, session_factory = _client_and_session()
    try:
        _, evaluation_run, reference_run, _ = _seed_sources(session_factory)
        before = client.get("/api/cache/revisions").json()["revisions"]["evaluations"]

        with session_factory() as db:
            reference = db.get(models.TestingRun, reference_run)
            assert reference is not None
            reference.updated_at = reference.updated_at + timedelta(seconds=1)
            db.commit()
        after_run = client.get("/api/cache/revisions").json()["revisions"]["evaluations"]
        assert after_run != before

        with session_factory() as db:
            db.add(
                models.TestingRunResult(
                    testing_run_id=evaluation_run,
                    position=100,
                    image_path="/evaluation/100.tif",
                    timestamp=BASE + timedelta(seconds=100),
                    score=0.0,
                    full_mse=0.0,
                    roi_mse=0.0,
                    width=8,
                    height=8,
                )
            )
            source = db.get(models.TestingRun, evaluation_run)
            assert source is not None
            source.result_revision += 1
            db.commit()
        after_result = client.get("/api/cache/revisions").json()["revisions"]["evaluations"]
        assert after_result != after_run
    finally:
        app.dependency_overrides.clear()


def test_csv_round_trip_and_stage_error_does_not_clear_other_metrics() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)

        exported = client.get(
            f"/api/evaluation-label-sets/{labels['id']}/csv-export"
        )
        assert exported.status_code == 200
        assert exported.text.startswith(
            "event_id,type,name,category,start_timestamp,end_timestamp,notes\n"
        )
        empty = client.post(
            "/api/evaluation-label-sets",
            json={
                "training_dataset_id": dataset_id,
                "name": "Imported",
                "events": [],
            },
        )
        assert empty.status_code == 200
        imported = client.post(
            f"/api/evaluation-label-sets/{empty.json()['id']}/csv-import",
            json={
                "training_dataset_id": dataset_id,
                "csv_text": exported.text,
                "mode": "replace",
            },
        )
        assert imported.status_code == 200, imported.text
        assert imported.json()["events"][0]["event_id"] == "event-1"

        evaluation = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )
        separation = client.post(
            f"/api/evaluations/{evaluation['id']}/calculate/separation"
        )
        assert separation.status_code == 200, separation.text
        previous_sep = separation.json()["sep_median"]

        with session_factory() as db:
            calibration = db.get(models.TestingRun, calibration_run)
            assert calibration is not None
            calibration.artifact_signature = "b" * 64
            db.commit()

        failed = client.post(
            f"/api/evaluations/{evaluation['id']}/calculate/detection"
        )
        assert failed.status_code == 400
        current = client.get(f"/api/evaluations/{evaluation['id']}").json()
        assert current["separation_status"] == "current"
        assert current["sep_median"] == previous_sep
        assert current["detection_status"] == "error"
        assert current["event_recall"] is None
        assert current["detection_result"]["error"]["code"] == "calculation_error"
    finally:
        app.dependency_overrides.clear()


def _profile_update_payload(profile: dict, **overrides) -> dict:
    fields = {
        "name",
        "description",
        "normal_window_duration_seconds",
        "normal_window_buffer_seconds",
        "drift_window_seconds",
        "false_alarm_horizon_seconds",
        "anticipation_seconds",
        "epsilon",
    }
    payload = {field: profile.get(field) for field in fields}
    payload.update(overrides)
    return payload


def _label_update_payload(labels: dict, *, notes: str | None = None) -> dict:
    return {
        "training_dataset_id": labels["training_dataset_id"],
        "name": labels["name"],
        "description": labels.get("description"),
        "events": [
            {
                "event_id": event["event_id"],
                "type": event["type"],
                "name": event.get("name"),
                "category": event.get("category"),
                "start_timestamp": event["start_timestamp"],
                "end_timestamp": event["end_timestamp"],
                "notes": notes if notes is not None else event.get("notes"),
            }
            for event in labels["events"]
        ],
    }


def test_stage_failure_duplicate_and_resource_snapshots_follow_lifecycle_rules() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)
        assert client.post(
            "/api/evaluation-profiles",
            json=_profile_update_payload(profile),
        ).status_code == 409
        assert client.post(
            "/api/evaluation-label-sets",
            json=_label_update_payload(labels),
        ).status_code == 409
        finalized = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )
        for stage in ("separation", "drift", "detection"):
            response = client.post(
                f"/api/evaluations/{finalized['id']}/calculate/{stage}"
            )
            assert response.status_code == 200, response.text
        finalized = client.post(
            f"/api/evaluations/{finalized['id']}/finalize"
        ).json()
        profile_snapshot = finalized["profile_snapshot"]
        label_snapshot = finalized["label_snapshot"]

        duplicate = client.post(
            f"/api/evaluations/{finalized['id']}/duplicate", json={}
        )
        assert duplicate.status_code == 200, duplicate.text
        draft = duplicate.json()
        assert draft["status"] == "draft"
        assert draft["evaluation_testing_run_id"] == evaluation_run
        for stage in ("separation", "drift", "detection"):
            assert draft[f"{stage}_status"] == "not_calculated"
            assert draft[f"{stage}_result"] is None
        assert all(
            draft[field] is None
            for field in (
                "sep_median",
                "sep_min",
                "drift_mean",
                "drift_max",
                "event_recall",
                "median_delay_seconds",
                "frame_fpr",
                "false_alarm_rate_t0",
            )
        )

        for stage in ("separation", "detection"):
            response = client.post(
                f"/api/evaluations/{draft['id']}/calculate/{stage}"
            )
            assert response.status_code == 200, response.text
            draft = response.json()
        old_a = (draft["sep_median"], draft["sep_min"])
        old_c = (
            draft["event_recall"],
            draft["median_delay_seconds"],
            draft["frame_fpr"],
            draft["false_alarm_rate_t0"],
        )

        profile_changed = client.put(
            f"/api/evaluation-profiles/{profile['id']}",
            json=_profile_update_payload(profile, drift_window_seconds=200),
        )
        assert profile_changed.status_code == 200, profile_changed.text
        after_profile = client.get(f"/api/evaluations/{draft['id']}").json()
        assert after_profile["separation_status"] == "current"
        assert after_profile["detection_status"] == "current"

        drift_failed = client.post(
            f"/api/evaluations/{draft['id']}/calculate/drift"
        )
        assert drift_failed.status_code == 400
        after_failure = client.get(f"/api/evaluations/{draft['id']}").json()
        assert after_failure["drift_status"] == "error"
        assert (after_failure["sep_median"], after_failure["sep_min"]) == old_a
        assert (
            after_failure["event_recall"],
            after_failure["median_delay_seconds"],
            after_failure["frame_fpr"],
            after_failure["false_alarm_rate_t0"],
        ) == old_c
        assert after_failure["drift_result"]["error"]["code"] == "no_valid_drift_windows"
        assert after_failure["drift_result"]["diagnostics"]["windows"]

        profile_restored = client.put(
            f"/api/evaluation-profiles/{profile['id']}",
            json=_profile_update_payload(
                profile_changed.json(), drift_window_seconds=20
            ),
        )
        assert profile_restored.status_code == 200
        drift_ok = client.post(
            f"/api/evaluations/{draft['id']}/calculate/drift"
        )
        assert drift_ok.status_code == 200, drift_ok.text
        assert drift_ok.json()["separation_status"] == "current"
        assert drift_ok.json()["detection_status"] == "current"

        normal_changed = client.put(
            f"/api/evaluation-profiles/{profile['id']}",
            json=_profile_update_payload(
                profile_restored.json(), normal_window_duration_seconds=21
            ),
        )
        assert normal_changed.status_code == 200
        selectively_stale = client.get(f"/api/evaluations/{draft['id']}").json()
        assert selectively_stale["separation_status"] == "stale"
        assert selectively_stale["drift_status"] == "current"
        assert selectively_stale["detection_status"] == "current"

        label_changed = client.put(
            f"/api/evaluation-label-sets/{labels['id']}",
            json=_label_update_payload(labels, notes="Updated after finalization"),
        )
        assert label_changed.status_code == 200, label_changed.text
        all_stale = client.get(f"/api/evaluations/{draft['id']}").json()
        assert all_stale["separation_status"] == "stale"
        assert all_stale["drift_status"] == "stale"
        assert all_stale["detection_status"] == "stale"

        finalized_after_updates = client.get(
            f"/api/evaluations/{finalized['id']}"
        ).json()
        assert finalized_after_updates["profile_snapshot"] == profile_snapshot
        assert finalized_after_updates["label_snapshot"] == label_snapshot
        assert finalized_after_updates["status"] == "finalized"
        assert all(
            finalized_after_updates[f"{stage}_status"] == "current"
            for stage in ("separation", "drift", "detection")
        )
    finally:
        app.dependency_overrides.clear()


def test_profile_update_respects_effective_local_stage_overrides() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)
        evaluation = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )
        overridden = client.patch(
            f"/api/evaluations/{evaluation['id']}",
            json={"profile_overrides": {"drift_window_seconds": 20}},
        )
        assert overridden.status_code == 200, overridden.text
        drift = client.post(f"/api/evaluations/{evaluation['id']}/calculate/drift")
        assert drift.status_code == 200, drift.text
        original_values = (drift.json()["drift_mean"], drift.json()["drift_max"])

        changed_profile = client.put(
            f"/api/evaluation-profiles/{profile['id']}",
            json=_profile_update_payload(profile, drift_window_seconds=40),
        )
        assert changed_profile.status_code == 200, changed_profile.text
        unchanged_effective_stage = client.get(
            f"/api/evaluations/{evaluation['id']}"
        ).json()
        assert unchanged_effective_stage["profile_snapshot"]["drift_window_seconds"] == 40
        assert unchanged_effective_stage["profile_overrides"]["drift_window_seconds"] == 20
        assert unchanged_effective_stage["drift_status"] == "current"
        assert (
            unchanged_effective_stage["drift_mean"],
            unchanged_effective_stage["drift_max"],
        ) == original_values
    finally:
        app.dependency_overrides.clear()


def test_identical_label_set_put_is_a_version_and_stale_noop() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)
        evaluation = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )
        for stage in ("separation", "drift", "detection"):
            calculated = client.post(
                f"/api/evaluations/{evaluation['id']}/calculate/{stage}"
            )
            assert calculated.status_code == 200, calculated.text

        identical = client.put(
            f"/api/evaluation-label-sets/{labels['id']}",
            json=_label_update_payload(labels),
        )
        assert identical.status_code == 200, identical.text
        assert identical.json()["version"] == labels["version"]
        persisted = client.get(f"/api/evaluations/{evaluation['id']}").json()
        assert persisted["label_snapshot"]["version"] == labels["version"]
        assert all(
            persisted[f"{stage}_status"] == "current"
            for stage in ("separation", "drift", "detection")
        )
    finally:
        app.dependency_overrides.clear()


def test_drift_accepts_an_exclusion_only_label_set() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = client.post(
            "/api/evaluation-label-sets",
            json={
                "training_dataset_id": dataset_id,
                "name": "Exclusions only",
                "events": [
                    {
                        "event_id": "maintenance",
                        "type": "exclusion",
                        "start_timestamp": (BASE + timedelta(seconds=90)).isoformat(),
                        "end_timestamp": (BASE + timedelta(seconds=95)).isoformat(),
                    }
                ],
            },
        )
        assert labels.status_code == 200, labels.text
        evaluation = _evaluation(
            client,
            profile["id"],
            labels.json()["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )

        calculated = client.post(
            f"/api/evaluations/{evaluation['id']}/calculate/drift"
        )
        assert calculated.status_code == 200, calculated.text
        assert calculated.json()["drift_status"] == "current"
        assert calculated.json()["drift_mean"] is not None
    finally:
        app.dependency_overrides.clear()


def test_target_exclusion_overlap_blocks_only_separation() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = client.post(
            "/api/evaluation-label-sets",
            json={
                "training_dataset_id": dataset_id,
                "name": "Target and invalid overlap",
                "events": [
                    {
                        "event_id": "event-1",
                        "type": "target",
                        "name": "Event",
                        "category": "fault",
                        "start_timestamp": (BASE + timedelta(seconds=70)).isoformat(),
                        "end_timestamp": (BASE + timedelta(seconds=80)).isoformat(),
                    },
                    {
                        "event_id": "invalid-frames",
                        "type": "exclusion",
                        "start_timestamp": (BASE + timedelta(seconds=75)).isoformat(),
                        "end_timestamp": (BASE + timedelta(seconds=76)).isoformat(),
                    },
                ],
            },
        )
        assert labels.status_code == 200, labels.text
        evaluation = _evaluation(
            client,
            profile["id"],
            labels.json()["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )

        drift = client.post(f"/api/evaluations/{evaluation['id']}/calculate/drift")
        detection = client.post(
            f"/api/evaluations/{evaluation['id']}/calculate/detection"
        )
        assert drift.status_code == 200, drift.text
        assert detection.status_code == 200, detection.text
        drift_values = (drift.json()["drift_mean"], drift.json()["drift_max"])
        detection_values = (
            detection.json()["event_recall"],
            detection.json()["median_delay_seconds"],
            detection.json()["frame_fpr"],
            detection.json()["false_alarm_rate_t0"],
        )

        separation = client.post(
            f"/api/evaluations/{evaluation['id']}/calculate/separation"
        )
        assert separation.status_code == 400
        assert "overlaps an excluded" in separation.json()["detail"]
        persisted = client.get(f"/api/evaluations/{evaluation['id']}").json()
        assert persisted["separation_status"] == "error"
        assert persisted["drift_status"] == "current"
        assert persisted["detection_status"] == "current"
        assert (persisted["drift_mean"], persisted["drift_max"]) == drift_values
        assert (
            persisted["event_recall"],
            persisted["median_delay_seconds"],
            persisted["frame_fpr"],
            persisted["false_alarm_rate_t0"],
        ) == detection_values
    finally:
        app.dependency_overrides.clear()


def test_evaluation_patch_rejects_null_for_required_draft_fields() -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)
        evaluation = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )

        for field in ("name", "score_series", "active_quantile"):
            rejected = client.patch(
                f"/api/evaluations/{evaluation['id']}", json={field: None}
            )
            assert rejected.status_code == 422, rejected.text

        unchanged = client.get(f"/api/evaluations/{evaluation['id']}").json()
        assert unchanged["name"] == evaluation["name"]
        assert unchanged["score_series"] == evaluation["score_series"]
        assert unchanged["active_quantile"] == evaluation["active_quantile"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("artifact", "different model artifacts"),
        ("missing_artifact", "missing an artifact signature"),
        ("roi", "different roi geometry"),
        ("config", "different inference/score configuration"),
        ("unfinished", "is not finished"),
        ("missing_score", "is missing values"),
        ("leakage", "overlaps data used to train"),
    ],
)
def test_drift_rejects_incompatible_leaky_or_incomplete_sources(
    failure: str, message: str
) -> None:
    client, session_factory = _client_and_session()
    try:
        dataset_id, evaluation_run, reference_run, calibration_run = _seed_sources(
            session_factory
        )
        profile = _profile(client)
        labels = _labels(client, dataset_id)
        evaluation = _evaluation(
            client,
            profile["id"],
            labels["id"],
            evaluation_run,
            reference_run,
            calibration_run,
        )
        with session_factory() as db:
            reference = db.get(models.TestingRun, reference_run)
            assert reference is not None
            if failure == "artifact":
                reference.artifact_signature = "b" * 64
            elif failure == "missing_artifact":
                reference.artifact_signature = None
            elif failure == "roi":
                reference.roi_geometry = {"x": 1, "y": 2, "width": 4, "height": 4}
            elif failure == "config":
                reference.inference_config = {"score_aggregation": "max"}
            elif failure == "unfinished":
                reference.status = "failed"
            elif failure == "missing_score":
                missing = db.scalar(
                    select(models.TestingRunResult).where(
                        models.TestingRunResult.testing_run_id == reference_run
                    )
                )
                assert missing is not None
                missing.roi_mse = None
            elif failure == "leakage":
                evaluation_dataset = db.get(models.TrainingDataset, dataset_id)
                evaluation_rule = db.scalar(
                    select(models.TrainingDatasetRule).where(
                        models.TrainingDatasetRule.training_dataset_id == dataset_id
                    )
                )
                assert evaluation_dataset is not None
                assert evaluation_rule is not None
                training_dataset = models.TrainingDataset(
                    dataset_id=evaluation_dataset.dataset_id,
                    name="model-training",
                    usage_label="train",
                )
                db.add(training_dataset)
                db.flush()
                db.add(
                    models.TrainingDatasetRule(
                        training_dataset_id=training_dataset.id,
                        folder_id=evaluation_rule.folder_id,
                        start_timestamp=BASE - timedelta(seconds=10),
                        # Outer role and training ranges are closed. Sharing only
                        # this endpoint is therefore still leakage.
                        end_timestamp=BASE,
                        stride=1,
                    )
                )
                db.add(
                    models.TrainingRun(
                        id=999,
                        training_pipeline_id=777,
                        status="finished",
                        training_pipeline_name="training",
                        method_type="mean_image",
                        method_family="baseline",
                        training_mode="fit",
                        builder_kind="form",
                        preprocessing_pipeline_name="preprocessing",
                        dataset_names=["evaluation"],
                        dataset_names_text="evaluation",
                        shuffle=False,
                        training_parameters={},
                    )
                )
                db.add(
                    models.TrainingPipelineDataset(
                        training_pipeline_id=777,
                        training_dataset_id=training_dataset.id,
                        position=0,
                    )
                )
            db.commit()

        if failure == "missing_score":
            changed = client.patch(
                f"/api/evaluations/{evaluation['id']}",
                json={"score_series": "roi_mse"},
            )
            assert changed.status_code == 200, changed.text

        failed = client.post(
            f"/api/evaluations/{evaluation['id']}/calculate/drift"
        )
        assert failed.status_code == 400
        assert message in failed.json()["detail"].lower()
        persisted = client.get(f"/api/evaluations/{evaluation['id']}").json()
        assert persisted["drift_status"] == "error"
        assert persisted["drift_result"]["error"]["message"]
    finally:
        app.dependency_overrides.clear()
