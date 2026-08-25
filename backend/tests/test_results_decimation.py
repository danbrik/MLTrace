from datetime import datetime, timedelta
from pathlib import Path

from app import models
from app.testing.service import get_testing_run_plot_series, get_testing_run_results

from tests.test_testing_service import make_db


def _seed_run_with_results(db, count: int) -> int:
    run = models.TestingRun(
        name="R",
        training_run_id=1,
        training_dataset_id=1,
        status="finished",
        training_run_name="x",
        training_pipeline_name="x",
        training_dataset_name="x",
        preprocessing_pipeline_name="x",
        method_type="m",
        method_family="f",
        training_mode="gradient",
        artifact_kind="weights",
        artifact_path="/tmp/a.pt",
    )
    db.add(run)
    db.flush()
    base = datetime(2026, 1, 1, 0, 0, 0)
    db.bulk_insert_mappings(
        models.TestingRunResult,
        [
            {
                "testing_run_id": run.id,
                "position": i,
                "image_path": f"/img/{i}.tiff",
                "timestamp": base + timedelta(seconds=i),
                "score": float(i),
                "full_mse": float(i),
                "roi_mse": None,
                "tile_scores": None,
                "width": 8,
                "height": 6,
            }
            for i in range(count)
        ],
    )
    db.commit()
    return run.id


def test_results_decimated_to_max_points_with_bounds(tmp_path: Path) -> None:
    db = make_db()
    try:
        run_id = _seed_run_with_results(db, count=1000)

        full = get_testing_run_results(db, run_id)
        assert full is not None
        assert full.total == 1000
        assert full.decimated is False
        assert len(full.results) == 1000

        capped = get_testing_run_results(db, run_id, max_points=100)
        assert capped is not None
        assert capped.total == 1000  # true count preserved
        assert capped.decimated is True
        assert len(capped.results) <= 110  # ~100 + last
        # First and last rows are always present for accurate bounds.
        assert capped.results[0].position == 0
        assert capped.results[-1].position == 999
        # Positions are monotonically increasing (ordered).
        positions = [r.position for r in capped.results]
        assert positions == sorted(positions)
    finally:
        db.close()


def test_decimation_preserves_full_resolution_continuity_segments() -> None:
    db = make_db()
    try:
        run_id = _seed_run_with_results(db, count=1000)
        rows = db.query(models.TestingRunResult).filter(
            models.TestingRunResult.testing_run_id == run_id,
            models.TestingRunResult.position >= 500,
        ).all()
        for row in rows:
            row.timestamp += timedelta(seconds=16)
        db.commit()

        capped = get_testing_run_results(db, run_id, max_points=100)
        assert capped is not None
        assert capped.decimated is True
        before = next(row for row in reversed(capped.results) if row.position < 500)
        after = next(row for row in capped.results if row.position >= 500)
        assert before.continuity_segment == 0
        assert after.continuity_segment == 1
        assert capped.results[-1].continuity_segment == 1
    finally:
        db.close()


def test_plot_series_is_full_resolution_range_filtered_and_keyset_paginated() -> None:
    db = make_db()
    try:
        run_id = _seed_run_with_results(db, count=7)
        run = db.get(models.TestingRun, run_id)
        run.result_revision = 4
        duplicate = db.query(models.TestingRunResult).filter_by(testing_run_id=run_id, position=2).one()
        duplicate.timestamp = db.query(models.TestingRunResult).filter_by(testing_run_id=run_id, position=1).one().timestamp
        db.commit()
        start = datetime(2026, 1, 1, 0, 0, 1)
        end = datetime(2026, 1, 1, 0, 0, 5)

        first = get_testing_run_plot_series(
            db, run_id, score_series="score", start_timestamp=start, end_timestamp=end,
            after_timestamp=None, after_position=None, expected_result_revision=None, limit=2,
        )
        assert first is not None
        assert first.total == 5
        assert [(point.position, point.value) for point in first.points] == [(1, 1.0), (2, 2.0)]
        assert first.next_timestamp == first.points[-1].timestamp
        second = get_testing_run_plot_series(
            db, run_id, score_series="score", start_timestamp=start, end_timestamp=end,
            after_timestamp=first.next_timestamp, after_position=first.next_position,
            expected_result_revision=first.result_revision, limit=2,
        )
        assert second is not None
        assert [point.position for point in second.points] == [3, 4]

        run.result_revision += 1
        db.commit()
        try:
            get_testing_run_plot_series(
                db, run_id, score_series="score", start_timestamp=start, end_timestamp=end,
                after_timestamp=second.next_timestamp, after_position=second.next_position,
                expected_result_revision=4, limit=2,
            )
            raise AssertionError("Expected revision conflict")
        except RuntimeError as exc:
            assert "changed" in str(exc)
    finally:
        db.close()


def test_plot_series_resolves_metadata_scores_without_silent_fast_or_future_fallback() -> None:
    db = make_db()
    try:
        run_id = _seed_run_with_results(db, count=2)
        rows = db.query(models.TestingRunResult).filter_by(testing_run_id=run_id).order_by(models.TestingRunResult.position).all()
        rows[0].result_metadata = {
            "reconstruction_score": 10.0,
            "prediction_score": 20.0,
            "fast_anogan": {"combined_score": 30.0},
            "future_scores": [{"horizon": 2, "score": 40.0}],
        }
        rows[1].result_metadata = dict(rows[0].result_metadata)
        db.commit()
        page = get_testing_run_plot_series(
            db, run_id, score_series="future+2", start_timestamp=None, end_timestamp=None,
            after_timestamp=None, after_position=None, expected_result_revision=None, limit=100,
        )
        assert page is not None
        assert [point.value for point in page.points] == [40.0, 40.0]
        rows[1].result_metadata = {}
        db.commit()
        try:
            get_testing_run_plot_series(
                db, run_id, score_series="fast_combined", start_timestamp=None, end_timestamp=None,
                after_timestamp=None, after_position=None, expected_result_revision=None, limit=100,
            )
            raise AssertionError("Expected missing-series error")
        except ValueError as exc:
            assert "position 1" in str(exc)
    finally:
        db.close()


def test_plot_series_export_is_not_limited_to_the_8000_point_chart_preview() -> None:
    db = make_db()
    try:
        run_id = _seed_run_with_results(db, count=8_005)
        preview = get_testing_run_results(db, run_id, max_points=8_000)
        export = get_testing_run_plot_series(
            db, run_id, score_series="score", start_timestamp=None, end_timestamp=None,
            after_timestamp=None, after_position=None, expected_result_revision=None, limit=50_000,
        )
        assert preview is not None and preview.decimated is True
        assert export is not None
        assert export.total == 8_005
        assert len(export.points) == 8_005
        assert export.next_timestamp is None
    finally:
        db.close()
