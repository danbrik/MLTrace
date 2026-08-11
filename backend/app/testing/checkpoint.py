"""Checkpoint primitives shared by the testing service and worker engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


CHECKPOINT_INTERVAL = 20_000
CHECKPOINT_VERSION = 1
MAX_SKIPPED_PATHS = 200


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _artifact_snapshot(path_value: str | None) -> dict:
    path = Path(path_value or "")
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def source_signature(
    *,
    kind: str,
    inputs: Iterable,
    training_run,
    graph,
    roi_geometry: dict | None,
    inference_config: dict | None,
) -> str:
    """Hash the ordered inference inputs and every effective scoring input.

    Source image bytes are intentionally not read. The ordered path/timestamp
    sequence plus the cached dataset selection and artifact stat make resume
    validation cheap even for hundreds of thousands of frames.
    """

    configuration = training_run.training_pipeline.method_configuration
    effective_inference_config = {
        **(configuration.inference_config or {}),
        **(inference_config or {}),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json({
        "version": CHECKPOINT_VERSION,
        "kind": kind,
        "training_run_id": training_run.id,
        "artifact_kind": training_run.artifact_kind,
        "artifact": _artifact_snapshot(training_run.artifact_path),
        "preprocessing_graph": graph.model_dump(mode="json"),
        "method_config": configuration.method_config or {},
        "effective_inference_config": effective_inference_config,
        "roi_geometry": roi_geometry,
    }))
    digest.update(b"\n")
    if kind == "clip":
        for clip in inputs:
            digest.update(_canonical_json({
                "score_timestamp": clip.score_timestamp.isoformat(),
                "input_frames": [
                    [frame.timestamp_parsed.isoformat(), frame.file_path]
                    for frame in clip.input_frames
                ],
                "future_frames": [
                    [frame.timestamp_parsed.isoformat(), frame.file_path]
                    for frame in clip.future_frames
                ],
            }))
            digest.update(b"\n")
    else:
        for record in inputs:
            digest.update(
                f"{record.timestamp_parsed.isoformat()}\0{record.file_path}\n".encode("utf-8")
            )
    return digest.hexdigest()


def make_checkpoint_state(
    *,
    kind: str,
    signature: str,
    input_count: int,
    result_count: int,
    total_inputs: int,
    score_sum: float,
    full_sum: float,
    roi_sum: float,
    roi_count: int,
    score_min: float | None,
    score_max: float | None,
    skipped_count: int,
    skipped_paths: list[str],
) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "kind": kind,
        "signature": signature,
        "input_count": int(input_count),
        "result_count": int(result_count),
        "total_inputs": int(total_inputs),
        "score_sum": float(score_sum),
        "full_sum": float(full_sum),
        "roi_sum": float(roi_sum),
        "roi_count": int(roi_count),
        "score_min": None if score_min is None else float(score_min),
        "score_max": None if score_max is None else float(score_max),
        "skipped_count": int(skipped_count),
        "skipped_paths": list(skipped_paths),
    }


def validated_checkpoint_state(run, *, kind: str, signature: str, total_inputs: int) -> dict:
    state = run.checkpoint_state
    if not isinstance(state, dict) or state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("No compatible inference checkpoint is available.")
    if state.get("kind") != kind:
        raise ValueError("The checkpoint belongs to a different inference sample type.")
    if state.get("signature") != signature or int(state.get("total_inputs", -1)) != total_inputs:
        raise ValueError(
            "The inference source, model, preprocessing, ROI, or scoring configuration changed since the checkpoint. "
            "Restart the inference completely."
        )
    try:
        input_count = int(state["input_count"])
        result_count = int(state["result_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The inference checkpoint is incomplete.") from exc
    if input_count <= 0 or input_count > total_inputs or result_count < 0 or result_count > input_count:
        raise ValueError("The inference checkpoint contains invalid progress counters.")
    try:
        float(state["score_sum"])
        float(state["full_sum"])
        float(state["roi_sum"])
        int(state["roi_count"])
        int(state["skipped_count"])
        if state.get("score_min") is not None:
            float(state["score_min"])
        if state.get("score_max") is not None:
            float(state["score_max"])
        if not isinstance(state.get("skipped_paths", []), list):
            raise TypeError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The inference checkpoint aggregate state is incomplete.") from exc
    if run.checkpoint_input_count != input_count or run.checkpoint_result_count != result_count:
        raise ValueError("The inference checkpoint summary does not match its stored state.")
    return state
