from datetime import datetime
from pathlib import Path

from app import models
from app.schemas import HeatmapRunCreate, TestingRunCreate
from app.testing import service as testing_service
from app.testing.service import compute_heatmap_run, create_testing_run, list_heatmap_runs

from tests.test_testing_service import make_db, seed_finished_mean_image_run


def test_heatmap_artifacts_on_disk_and_light_list(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        # Route all run artifacts into a temp data dir.
        monkeypatch.setattr(testing_service, "data_dir", lambda: tmp_path / "data")

        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        testing_run = create_testing_run(
            db,
            TestingRunCreate(training_run_id=training_run_id, training_dataset_id=test_set_id),
        )

        heatmap = compute_heatmap_run(
            db,
            HeatmapRunCreate(testing_run_id=testing_run.id, timestamp=datetime(2026, 4, 1, 12, 0, 0)),
        )

        # The returned read still carries the full payload (reconstructed from disk).
        assert heatmap.status == "finished"
        assert heatmap.source_image_data_url.startswith("data:image/png;base64,")
        assert heatmap.heatmap_image_data_url.startswith("data:image/png;base64,")
        assert heatmap.error_matrix is not None and len(heatmap.error_matrix) > 0

        # The DB row stores only metadata + a pointer; heavy columns stay empty.
        row = db.get(models.HeatmapRun, heatmap.id)
        assert row.artifacts_dir
        assert row.source_image_data_url == ""
        assert row.reconstruction_image_data_url == ""
        assert row.heatmap_image_data_url == ""
        assert row.error_matrix is None

        # Files exist on disk.
        artifacts = Path(row.artifacts_dir)
        for name in ("source.png", "reconstruction.png", "overlay.png", "error_matrix.npy"):
            assert (artifacts / name).exists() and (artifacts / name).stat().st_size > 0

        # The list endpoint returns lightweight summaries (no image data URLs / matrix).
        summaries = list_heatmap_runs(db)
        assert len(summaries) == 1
        assert summaries[0].id == heatmap.id
        assert summaries[0].max_error == row.max_error
        assert not hasattr(summaries[0], "source_image_data_url")
        assert not hasattr(summaries[0], "error_matrix")

        # Dedup returns the same heatmap, still fully serialized from disk.
        again = compute_heatmap_run(
            db,
            HeatmapRunCreate(testing_run_id=testing_run.id, timestamp=datetime(2026, 4, 1, 12, 0, 0)),
        )
        assert again.id == heatmap.id
        assert again.error_matrix is not None
    finally:
        db.close()
