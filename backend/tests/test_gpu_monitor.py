from app import gpu_monitor


def test_gpu_snapshot_attributes_mltrace_memory_by_pid(monkeypatch) -> None:
    project_usage = {
        "project_id": "project-a",
        "project_name": "Project A",
        "gpu_memory_mb": 0,
        "running_jobs": 1,
        "queued_jobs": 2,
        "gpu_slots": 1,
    }
    monkeypatch.setattr(gpu_monitor, "_job_usage", lambda: ({123: {"project": project_usage, "gpu_index": 0}}, [project_usage]))

    def fake_smi(query: str):
        if query.startswith("--query-gpu"):
            return [["0", "GPU-1", "Test GPU", "75", "4000", "24000", "54"]]
        return [["123", "GPU-1", "1024"], ["999", "GPU-1", "512"]]

    monkeypatch.setattr(gpu_monitor, "_run_nvidia_smi", fake_smi)
    gpu_monitor.invalidate_gpu_snapshot()
    snapshot = gpu_monitor.gpu_snapshot(force=True)

    assert snapshot["available"] is True
    assert snapshot["mltrace_memory_mb"] == 1024
    assert snapshot["devices"][0]["mltrace_memory_mb"] == 1024
    assert snapshot["projects"][0]["queued_jobs"] == 2


def test_gpu_snapshot_survives_missing_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(gpu_monitor, "_job_usage", lambda: ({}, []))
    monkeypatch.setattr(gpu_monitor, "_run_nvidia_smi", lambda _query: (_ for _ in ()).throw(FileNotFoundError()))
    snapshot = gpu_monitor.gpu_snapshot(force=True)
    assert snapshot["available"] is False
    assert "not installed" in snapshot["error"]
