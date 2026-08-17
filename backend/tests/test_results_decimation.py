from datetime import datetime, timedelta
from pathlib import Path

from app import models
from app.testing.service import get_testing_run_results

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
            row.timestamp += timedelta(seconds=1)
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
