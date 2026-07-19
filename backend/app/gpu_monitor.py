from __future__ import annotations

import subprocess
import threading
import time
from datetime import UTC, datetime

from sqlalchemy import select

from app import models
from app.database import SessionLocal, project_context
from app.projects import list_projects

_CACHE_SECONDS = 60
_cache: dict | None = None
_cache_time = 0.0
_lock = threading.Lock()
_JOB_MODELS = (models.TrainingRun, models.TestingRun, models.HeatmapRangeRun)


def invalidate_gpu_snapshot() -> None:
    global _cache_time
    with _lock:
        _cache_time = 0.0


def _number(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default


def _run_nvidia_smi(query: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def _job_usage() -> tuple[dict[int, dict], list[dict]]:
    pid_map: dict[int, dict] = {}
    projects_usage: list[dict] = []
    for project in list_projects():
        usage = {
            "project_id": project.id,
            "project_name": project.name,
            "gpu_memory_mb": 0,
            "running_jobs": 0,
            "queued_jobs": 0,
            "gpu_slots": 0,
        }
        with project_context(project.database_url, project.artifact_dir):
            db = SessionLocal()
            try:
                for model in _JOB_MODELS:
                    for run in db.scalars(select(model).where(model.status.in_(("queued", "running")))):
                        if run.status == "queued":
                            usage["queued_jobs"] += 1
                        else:
                            usage["running_jobs"] += 1
                            if getattr(run, "gpu_index", None) is not None:
                                usage["gpu_slots"] += 1
                            if getattr(run, "pid", None):
                                pid_map[int(run.pid)] = {
                                    "project": usage,
                                    "gpu_index": getattr(run, "gpu_index", None),
                                }
            finally:
                db.close()
        projects_usage.append(usage)
    return pid_map, projects_usage


def gpu_snapshot(force: bool = False) -> dict:
    global _cache, _cache_time
    now = time.monotonic()
    with _lock:
        if not force and _cache is not None and now - _cache_time < _CACHE_SECONDS:
            return _cache

    pid_map, projects_usage = _job_usage()
    captured_at = datetime.now(UTC).replace(tzinfo=None)
    devices: list[dict] = []
    error: str | None = None
    try:
        rows = _run_nvidia_smi(
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
        )
        for row in rows:
            if len(row) < 7:
                continue
            devices.append({
                "index": int(_number(row[0])),
                "uuid": row[1],
                "name": row[2],
                "utilization_percent": _number(row[3]),
                "memory_used_mb": int(_number(row[4])),
                "memory_total_mb": int(_number(row[5])),
                "temperature_c": _number(row[6]) if row[6] not in {"N/A", "[N/A]"} else None,
                "mltrace_memory_mb": 0,
                "projects": [],
            })
        by_uuid = {device["uuid"]: device for device in devices}
        process_rows = _run_nvidia_smi("--query-compute-apps=pid,gpu_uuid,used_gpu_memory")
        device_project_usage: dict[tuple[int, str], dict] = {}
        for row in process_rows:
            if len(row) < 3:
                continue
            pid = int(_number(row[0], -1))
            owner = pid_map.get(pid)
            device = by_uuid.get(row[1])
            if owner is None or device is None:
                continue
            memory = int(_number(row[2]))
            owner["project"]["gpu_memory_mb"] += memory
            device["mltrace_memory_mb"] += memory
            key = (device["index"], owner["project"]["project_id"])
            aggregate = device_project_usage.setdefault(key, {**owner["project"], "gpu_memory_mb": 0})
            aggregate["gpu_memory_mb"] += memory
        for device in devices:
            device["projects"] = [
                value for (index, _), value in device_project_usage.items() if index == device["index"]
            ]
    except FileNotFoundError:
        error = "nvidia-smi is not installed or not available on PATH."
    except subprocess.TimeoutExpired:
        error = "nvidia-smi timed out."
    except (subprocess.CalledProcessError, ValueError) as exc:
        error = f"Could not read NVIDIA status: {exc}"

    snapshot = {
        "captured_at": captured_at,
        "available": error is None,
        "error": error,
        "devices": devices,
        "mltrace_memory_mb": sum(item["gpu_memory_mb"] for item in projects_usage),
        "running_jobs": sum(item["running_jobs"] for item in projects_usage),
        "queued_jobs": sum(item["queued_jobs"] for item in projects_usage),
        "gpu_slots": sum(item["gpu_slots"] for item in projects_usage),
        "projects": projects_usage,
    }
    with _lock:
        _cache = snapshot
        _cache_time = time.monotonic()
    return snapshot
