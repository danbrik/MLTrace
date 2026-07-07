from pathlib import Path

from sqlalchemy import select

from app import models
from app.registry import service as registry_service
from app.registry.service import RegistryConflict, delete_entities, delete_preview
from app.schemas import TestingRunCreate as TestingRunCreatePayload
from app.testing import service as testing_service
from app.testing.service import create_testing_run

from tests.test_testing_service import make_db, seed_finished_mean_image_run


def _patch_data_dir(monkeypatch, root: Path) -> None:
    """Route every disk artifact location used by registry deletes to tmp."""
    import app.heatmap.service as heatmap_service_module
    import app.inspect.service as inspect_service_module
    import app.registry.specs as registry_specs_module
    import app.training.service as training_service_module

    for module in (
        registry_specs_module,
        testing_service,
        training_service_module,
        heatmap_service_module,
        inspect_service_module,
    ):
        monkeypatch.setattr(module, "data_dir", lambda: root)


def test_registry_summary_lists_all_types(tmp_path: Path) -> None:
    db = make_db()
    try:
        seed_finished_mean_image_run(db, tmp_path)
        summary = registry_service.registry_summary(db)
        keys = {t["key"] for t in summary["types"]}
        assert len(keys) == 13
        counts = {t["key"]: t["count"] for t in summary["types"]}
        assert counts["preprocessing_pipeline"] == 1
        assert counts["training_pipeline"] == 1
        assert counts["training_run"] == 1
        method_filters = next(t for t in summary["types"] if t["key"] == "method_configuration")["filters"]
        method_type = next(f for f in method_filters if f["key"] == "method_type")
        assert method_type["options"] == ["mean_image"]
    finally:
        db.close()


def test_registry_list_search_filter_and_usage(tmp_path: Path) -> None:
    db = make_db()
    try:
        seed_finished_mean_image_run(db, tmp_path)
        # Second, unused method configuration.
        unused = models.MethodConfiguration(
            name="Unused method",
            method_type="mean_image",
            method_family="statistical_baseline",
            method_version="1",
            training_mode="fit",
            requires_training=True,
            supports_training_pipeline=False,
            artifact_kind="mean_image",
            builder_kind="form",
            method_graph={},
            method_config={},
            training_config={},
            inference_config={},
            diagram={},
        )
        db.add(unused)
        db.commit()

        result = registry_service.list_registry_rows(db, "method_configuration", search="Unused")
        assert result["total"] == 1
        assert result["rows"][0]["name"] == "Unused method"
        assert result["rows"][0]["usage_count"] == 0

        unused_only = registry_service.list_registry_rows(
            db, "method_configuration", filters={"usage": "unused"}
        )
        assert [r["name"] for r in unused_only["rows"]] == ["Unused method"]

        used_only = registry_service.list_registry_rows(db, "method_configuration", filters={"usage": "used"})
        assert used_only["total"] == 1
        assert used_only["rows"][0]["usage_count"] == 1

        finished = registry_service.list_registry_rows(db, "training_run", filters={"status": "finished"})
        assert finished["total"] == 1
    finally:
        db.close()


def test_registry_detail_contains_full_fields_and_dependents(tmp_path: Path) -> None:
    db = make_db()
    try:
        seed_finished_mean_image_run(db, tmp_path)
        pipeline_id = db.scalar(select(models.PreprocessingPipeline.id))
        detail = registry_service.get_registry_detail(db, "preprocessing_pipeline", pipeline_id)
        assert detail is not None
        assert detail["fields"]["graph"]["nodes"][0]["type"] == "load_image"  # full JSON exposed
        dependent_types = {d["entity_type"] for d in detail["dependents"]}
        assert "training_pipeline" in dependent_types
    finally:
        db.close()


def test_delete_preview_shows_full_chain_and_files(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        _patch_data_dir(monkeypatch, tmp_path / "data")
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        testing_run = create_testing_run(
            db, TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id)
        )
        assert testing_run.status == "finished"

        pipeline_id = db.scalar(select(models.PreprocessingPipeline.id))
        preview = delete_preview(db, [("preprocessing_pipeline", pipeline_id)])

        group_types = [g["entity_type"] for g in preview["groups"]]
        assert group_types.index("testing_run") < group_types.index("training_run")
        assert group_types.index("training_run") < group_types.index("training_pipeline")
        assert group_types.index("training_pipeline") < group_types.index("preprocessing_pipeline")
        assert preview["dependent_objects"] == 3  # pipeline, run, testing run
        assert preview["blockers"] == []
        # Inference CSV exists on disk and is included with its size.
        csv_files = [f for f in preview["files"] if "testing_runs" in f["path"]]
        assert csv_files and csv_files[0]["exists"]
        assert preview["total_bytes"] > 0
    finally:
        db.close()


def test_cascade_delete_removes_chain_and_disk(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        data_root = tmp_path / "data"
        _patch_data_dir(monkeypatch, data_root)
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        testing_run = create_testing_run(
            db, TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id)
        )
        testing_dir = data_root / "testing_runs" / str(testing_run.id)
        assert testing_dir.exists()

        pipeline_id = db.scalar(select(models.PreprocessingPipeline.id))
        result = delete_entities(db, [("preprocessing_pipeline", pipeline_id)], cascade=True)

        assert result["deleted"] == {
            "testing_run": 1,
            "training_run": 1,
            "training_pipeline": 1,
            "preprocessing_pipeline": 1,
        }
        assert db.scalar(select(models.PreprocessingPipeline.id)) is None
        assert db.scalar(select(models.TrainingPipeline.id)) is None
        assert db.scalar(select(models.TrainingRun.id)) is None
        assert db.scalar(select(models.TestingRun.id)) is None
        assert not testing_dir.exists()
    finally:
        db.close()


def test_delete_without_cascade_conflicts_when_dependents_exist(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        _patch_data_dir(monkeypatch, tmp_path / "data")
        seed_finished_mean_image_run(db, tmp_path)
        pipeline_id = db.scalar(select(models.PreprocessingPipeline.id))
        try:
            delete_entities(db, [("preprocessing_pipeline", pipeline_id)], cascade=False)
            raise AssertionError("Expected RegistryConflict without cascade")
        except RegistryConflict as exc:
            assert "cascade" in str(exc)
        assert db.scalar(select(models.PreprocessingPipeline.id)) is not None
    finally:
        db.close()


def test_delete_blocked_by_running_job(tmp_path: Path, monkeypatch) -> None:
    db = make_db()
    try:
        _patch_data_dir(monkeypatch, tmp_path / "data")
        training_run_id, test_set_id = seed_finished_mean_image_run(db, tmp_path)
        testing_run = create_testing_run(
            db, TestingRunCreatePayload(training_run_id=training_run_id, training_dataset_id=test_set_id)
        )
        row = db.get(models.TestingRun, testing_run.id)
        row.status = "running"
        db.commit()

        preview = delete_preview(db, [("testing_run", testing_run.id)])
        assert preview["blockers"]

        try:
            delete_entities(db, [("testing_run", testing_run.id)], cascade=True)
            raise AssertionError("Expected RegistryConflict for running job")
        except RegistryConflict as exc:
            assert "running" in str(exc)
    finally:
        db.close()
