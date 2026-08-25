from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.redundancy.engine import INCOMPLETE_MATRIX_MESSAGE, analyze_sensor_redundancy, cluster_cut
from app.redundancy import service


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_spearman_qc_and_incomplete_matrix_exclusion(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "sensors.csv",
        "time,a,b,c,d,constant\n"
        "2026-01-01 00:00:00,1,10,1,,5\n"
        "2026-01-01 00:01:00,2,8,2,,5\n"
        "2026-01-01 00:02:00,2,8,3,,5\n"
        "2026-01-01 00:03:00,4,4,,4,5\n"
        "2026-01-01 00:04:00,5,2,,5,5\n"
        "2026-01-01 00:05:00,6,0,,6,5\n"
        "2026-01-01 00:06:00,broken,-2,,,5\n",
    )
    result = analyze_sensor_redundancy(
        source,
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, 0, 6),
        "time",
        ["a", "b", "c", "d", "constant"],
        {"min_valid_values": 3, "min_pair_values": 3},
    )

    a_b = next(item for item in result["pairs"] if {item["variable_a"], item["variable_b"]} == {"a", "b"})
    assert a_b["spearman_rho"] == pytest.approx(-1.0)
    assert a_b["common_n"] == 6
    assert result["common_n"][2][3] == 0
    assert set(result["clusterable_variables"]) == {"a", "b"}
    exclusions = {item["variable"]: item for item in result["clustering_exclusions"]}
    assert exclusions["c"]["message"] == INCOMPLETE_MATRIX_MESSAGE
    assert exclusions["c"]["missing_with"] == ["d"]
    quality = {item["variable"]: item for item in result["quality"]}
    assert "Invalid numeric values treated as missing" in quality["a"]["statuses"]
    assert "Constant" in quality["constant"]["statuses"]
    assert INCOMPLETE_MATRIX_MESSAGE in quality["c"]["statuses"]

    cut = cluster_cut(result, 0.8)
    assert {item["variable"] for item in cut["assignments"]} == {"a", "b"}


def test_average_ranks_for_ties_and_inclusive_range(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "ties.csv",
        "time,x,y\n"
        "2026.01.01T00:00,1,1\n"
        "2026.01.01T00:01,1,2\n"
        "2026.01.01T00:02,2,2\n"
        "2026.01.01T00:03,3,4\n",
    )
    result = analyze_sensor_redundancy(
        source,
        datetime(2026, 1, 1, 0, 1),
        datetime(2026, 1, 1, 0, 3),
        "time",
        ["x", "y"],
        {"min_valid_values": 3, "min_pair_values": 3},
    )
    assert result["summary"]["timepoint_count"] == 3
    assert result["pairs"][0]["spearman_rho"] == pytest.approx(0.8660254037844387)


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_db():
        db: Session = factory()
        try:
            yield db
        finally:
            db.close()

    artifact_dir = tmp_path / "redundancy"
    artifact_dir.mkdir()
    monkeypatch.setattr(service, "_source_dir", lambda: artifact_dir)
    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def test_api_upload_calculate_export_finalize_and_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    try:
        content = (
            "time,a,b,c\n"
            "2026-02-01 00:00:00,1,2,3\n"
            "2026-02-01 00:01:00,2,4,2\n"
            "2026-02-01 00:02:00,3,6,1\n"
        ).encode()
        uploaded = client.post(
            "/api/redundancy/sources?name=Sensors",
            files={"file": ("sensors.csv", io.BytesIO(content), "text/csv")},
        )
        assert uploaded.status_code == 200, uploaded.text
        source = uploaded.json()
        assert source["row_count"] == 3
        assert next(item for item in source["column_profiles"] if item["name"] == "time")["timestamp_start"] == "2026-02-01T00:00:00"
        reused = client.post(
            "/api/redundancy/sources?name=Ignored duplicate name",
            files={"file": ("copy.csv", io.BytesIO(content), "text/csv")},
        )
        assert reused.status_code == 200
        assert reused.json()["id"] == source["id"]

        created = client.post("/api/redundancy/analyses", json={
            "source_id": source["id"],
            "name": "February redundancy",
            "time_column": "time",
            "start_timestamp": "2026-02-01T00:00:00",
            "end_timestamp": "2026-02-01T00:02:00",
            "selected_columns": ["a", "b", "c"],
            "config": {"min_valid_values": 3, "min_pair_values": 3},
        })
        assert created.status_code == 200, created.text
        analysis_id = created.json()["id"]
        assert client.patch(f"/api/redundancy/analyses/{analysis_id}", json={"name": None}).status_code == 422

        calculated = client.post(f"/api/redundancy/analyses/{analysis_id}/calculate")
        assert calculated.status_code == 200, calculated.text
        assert calculated.json()["job_status"] == "ready"
        assert calculated.json()["result"]["pairs"][0]["common_n"] == 3

        series = client.get(f"/api/redundancy/analyses/{analysis_id}/series", params=[("columns", "a"), ("columns", "b")])
        assert series.status_code == 200
        assert len(series.json()["points"]) == 3
        assert "continuity_segment" in series.json()["points"][0]
        first_page = client.get(
            f"/api/redundancy/analyses/{analysis_id}/series",
            params=[("columns", "a"), ("page_size", "2"), ("offset", "0")],
        ).json()
        assert len(first_page["points"]) == 2
        assert first_page["next_offset"] == 2
        assert client.get(f"/api/redundancy/analyses/{analysis_id}/exports/quality").text.startswith("\ufeffVariable")

        finalized = client.post(f"/api/redundancy/analyses/{analysis_id}/finalize", json={"cutoff": 0.9})
        assert finalized.status_code == 200
        assert finalized.json()["status"] == "finalized"
        assert client.patch(f"/api/redundancy/analyses/{analysis_id}", json={"name": "changed"}).status_code == 409

        duplicated = client.post(f"/api/redundancy/analyses/{analysis_id}/duplicate")
        assert duplicated.status_code == 200
        assert duplicated.json()["status"] == "draft"
        assert duplicated.json()["result"] is None
        assert client.delete(f"/api/redundancy/sources/{source['id']}").status_code == 409
    finally:
        app.dependency_overrides.clear()
        client.close()
