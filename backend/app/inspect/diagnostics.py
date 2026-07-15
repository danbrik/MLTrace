"""CPU-only Inspect diagnostics for temporal Energy and Optical Flow.

The functions in this module operate on already preprocessed frames. They keep
only the previous/current frame and small rolling windows in memory, so large
video-like datasets remain practical.
"""

from __future__ import annotations

import csv
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app import models
from app.inspect.contrast import to_intensity_16scale
from app.preprocessing.pipeline import encode_absolute_image_data_url
from app.video import add_timestamp_watermark

TILE_COLORS = [
    (28, 126, 214),
    (232, 89, 12),
    (47, 158, 68),
    (156, 54, 181),
    (12, 133, 153),
    (224, 49, 49),
    (95, 61, 196),
    (102, 168, 15),
    (245, 159, 0),
    (25, 113, 194),
]


def _float01(image: np.ndarray) -> np.ndarray:
    return to_intensity_16scale(image).astype(np.float32) / 65535.0


def _roi_points(roi: models.RoiDefinition) -> list[dict[str, float]]:
    if roi.points and len(roi.points) == 4:
        return [{"x": float(point["x"]), "y": float(point["y"])} for point in roi.points]
    return [
        {"x": float(roi.x), "y": float(roi.y)},
        {"x": float(roi.x + roi.width), "y": float(roi.y)},
        {"x": float(roi.x + roi.width), "y": float(roi.y + roi.height)},
        {"x": float(roi.x), "y": float(roi.y + roi.height)},
    ]


def _interpolate_quad(points: list[dict[str, float]], u: float, v: float) -> dict[str, float]:
    tl, tr, br, bl = points
    top_x = tl["x"] + (tr["x"] - tl["x"]) * u
    top_y = tl["y"] + (tr["y"] - tl["y"]) * u
    bottom_x = bl["x"] + (br["x"] - bl["x"]) * u
    bottom_y = bl["y"] + (br["y"] - bl["y"]) * u
    return {"x": top_x + (bottom_x - top_x) * v, "y": top_y + (bottom_y - top_y) * v}


def _tile_polygons(points: list[dict[str, float]], rows: int, cols: int) -> list[dict[str, Any]]:
    tiles: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            u0 = col / cols
            u1 = (col + 1) / cols
            v0 = row / rows
            v1 = (row + 1) / rows
            tiles.append(
                {
                    "key": f"tile_{row + 1}_{col + 1}",
                    "label": f"Tile {row + 1},{col + 1}",
                    "row": row + 1,
                    "col": col + 1,
                    "points": [
                        _interpolate_quad(points, u0, v0),
                        _interpolate_quad(points, u1, v0),
                        _interpolate_quad(points, u1, v1),
                        _interpolate_quad(points, u0, v1),
                    ],
                }
            )
    return tiles


def _polygon_mask(width: int, height: int, points: list[dict[str, float]]) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon([(point["x"], point["y"]) for point in points], fill=1)
    return np.asarray(mask, dtype=bool)


def _mask_specs(width: int, height: int, roi: models.RoiDefinition | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if roi is None:
        return [
            {
                "key": "total",
                "label": "Full image",
                "mask": np.ones((height, width), dtype=bool),
                "points": [
                    {"x": 0, "y": 0},
                    {"x": width, "y": 0},
                    {"x": width, "y": height},
                    {"x": 0, "y": height},
                ],
                "color": TILE_COLORS[0],
            }
        ], None
    if roi.image_width != width or roi.image_height != height:
        raise ValueError(f"ROI '{roi.name}' is tuned for {roi.image_width}x{roi.image_height}, but Inspect output is {width}x{height}.")
    points = _roi_points(roi)
    rows = max(1, int(roi.tile_rows or 1))
    cols = max(1, int(roi.tile_cols or 1))
    specs = []
    for index, tile in enumerate(_tile_polygons(points, rows, cols)):
        specs.append(
            {
                **tile,
                "mask": _polygon_mask(width, height, tile["points"]),
                "color": TILE_COLORS[index % len(TILE_COLORS)],
            }
        )
    return specs, {
        "id": roi.id,
        "name": roi.name,
        "image_width": roi.image_width,
        "image_height": roi.image_height,
        "tile_rows": rows,
        "tile_cols": cols,
        "points": points,
    }


def _aggregate(values: np.ndarray, aggregation: str, normalize_by_pixels: bool) -> float:
    if values.size == 0:
        return float("nan")
    if aggregation in {"mean", "mean_magnitude"}:
        return float(np.mean(values))
    if aggregation in {"p95", "p95_magnitude"}:
        return float(np.percentile(values, 95))
    if aggregation in {"max", "max_magnitude"}:
        return float(np.max(values))
    total = float(np.sum(values))
    return total / float(values.size) if normalize_by_pixels else total


def _overlay_image(base01: np.ndarray, specs: list[dict[str, Any]], values: dict[str, float] | None = None) -> np.ndarray:
    gray = np.clip(base01 * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)
    image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    max_value = max([abs(value) for value in (values or {}).values() if np.isfinite(value)] or [1.0])
    for index, spec in enumerate(specs):
        color = spec["color"]
        alpha = 42
        if values:
            alpha = int(35 + 105 * min(1.0, abs(values.get(spec["key"], 0.0)) / max_value))
        polygon = [(point["x"], point["y"]) for point in spec["points"]]
        draw.polygon(polygon, fill=(*color, alpha), outline=(*color, 230))
        label_pos = polygon[0]
        draw.text((label_pos[0] + 4, label_pos[1] + 4), str(index + 1), fill=(*color, 255))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _plot_preview(example_rgb: np.ndarray, rows: list[dict[str, Any]], specs: list[dict[str, Any]], value_prefix: str) -> np.ndarray:
    width = max(760, example_rgb.shape[1])
    top_height = min(360, max(220, int(example_rgb.shape[0] * min(1.0, width / max(1, example_rgb.shape[1])))))
    plot_height = 300
    canvas = Image.new("RGB", (width, top_height + plot_height + 70), "white")
    example = Image.fromarray(example_rgb).resize((int(example_rgb.shape[1] * top_height / example_rgb.shape[0]), top_height))
    canvas.paste(example, ((width - example.width) // 2, 10))
    draw = ImageDraw.Draw(canvas)
    chart = (60, top_height + 35, width - 30, top_height + plot_height + 15)
    draw.rectangle(chart, outline=(210, 210, 210))
    if rows:
        x_count = max(1, len(rows) - 1)
        all_values = []
        for spec in specs:
            key = value_prefix if spec["key"] == "total" else spec["key"]
            all_values.extend(float(row.get(key, 0.0)) for row in rows if row.get(key) is not None)
        vmax = max(all_values) if all_values else 1.0
        vmin = min(all_values) if all_values else 0.0
        if vmax == vmin:
            vmax = vmin + 1.0
        for spec in specs:
            key = value_prefix if spec["key"] == "total" else spec["key"]
            points = []
            for index, row in enumerate(rows):
                value = float(row.get(key, 0.0) or 0.0)
                x = chart[0] + (chart[2] - chart[0]) * index / x_count
                y = chart[3] - (chart[3] - chart[1]) * (value - vmin) / (vmax - vmin)
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=spec["color"], width=2)
            label = "Total" if spec["key"] == "total" else spec["label"]
            draw.text((chart[0] + 8, chart[1] + 16 * specs.index(spec)), label, fill=spec["color"])
    draw.text((60, top_height + plot_height + 30), f"{len(rows)} temporal samples · {value_prefix}", fill=(80, 80, 80))
    return np.asarray(canvas)


def _summarize(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, dict[str, float | None]]:
    summary = {}
    for column in columns:
        values = np.asarray([row[column] for row in rows if isinstance(row.get(column), (int, float)) and np.isfinite(row[column])], dtype=np.float64)
        summary[column] = {
            "min": None if values.size == 0 else float(values.min()),
            "max": None if values.size == 0 else float(values.max()),
            "mean": None if values.size == 0 else float(values.mean()),
        }
    return summary


def _energy_values(prev: np.ndarray, current: np.ndarray, specs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    diff = np.square(current - prev)
    aggregation = str(config.get("aggregation", "sum"))
    normalize = bool(config.get("normalize_by_pixels", False))
    output = {"energy_total": _aggregate(diff.reshape(-1), aggregation, normalize)}
    for spec in specs:
        if spec["key"] == "total":
            continue
        output[spec["key"]] = _aggregate(diff[spec["mask"]], aggregation, normalize)
    return output


def _flow_values(prev: np.ndarray, current: np.ndarray, specs: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
    flow = cv2.calcOpticalFlowFarneback(
        prev.astype(np.float32),
        current.astype(np.float32),
        None,
        float(config.get("pyr_scale", 0.5)),
        int(config.get("levels", 3)),
        int(config.get("winsize", 15)),
        int(config.get("iterations", 3)),
        int(config.get("poly_n", 5)),
        float(config.get("poly_sigma", 1.2)),
        0,
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    aggregation = str(config.get("aggregation", "mean_magnitude"))
    normalize = bool(config.get("normalize_by_pixels", True))
    output = {
        "flow_mean": float(np.mean(magnitude)),
        "flow_p95": float(np.percentile(magnitude, 95)),
        "flow_selected": _aggregate(magnitude.reshape(-1), aggregation, normalize),
    }
    for spec in specs:
        if spec["key"] == "total":
            continue
        output[spec["key"]] = _aggregate(magnitude[spec["mask"]], aggregation, normalize)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8((angle * 90 / np.pi) % 180)
    hsv[..., 1] = 255
    max_mag = float(magnitude.max()) or 1.0
    hsv[..., 2] = np.uint8(np.clip(magnitude / max_mag, 0, 1) * 255)
    return output, cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _apply_window(rows: list[dict[str, Any]], value_columns: list[str], window_size: int, mode: str) -> list[dict[str, Any]]:
    if window_size <= 1:
        return rows
    buffers = {column: deque(maxlen=window_size) for column in value_columns}
    next_rows = []
    for row in rows:
        next_row = dict(row)
        for column in value_columns:
            buffers[column].append(float(row[column]))
            values = list(buffers[column])
            next_row[column] = float(sum(values) if mode == "sum" else sum(values) / len(values))
        next_rows.append(next_row)
    return next_rows


def compute_diagnostic(
    mode: str,
    compiled,
    records,
    config: dict[str, Any] | None,
    roi: models.RoiDefinition | None,
    abort_event: threading.Event | None = None,
    artifact_dir: Path | None = None,
    fps: int = 12,
    generate_video: bool = False,
    preview_limit: int | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    config = dict(config or {})
    if mode not in {"energy", "optical_flow"}:
        raise ValueError(f"Unsupported Inspect diagnostic mode: {mode}.")
    selected_records = records[:preview_limit] if preview_limit else records
    if len(selected_records) < 2:
        raise ValueError(f"{mode} requires at least two selected frames.")

    first = _float01(compiled.run(selected_records[0].file_path))
    height, width = first.shape[:2]
    specs, roi_meta = _mask_specs(width, height, roi)
    rows: list[dict[str, Any]] = []
    prev = first
    writer = None
    frames_dir = artifact_dir / "frames" if artifact_dir and generate_video else None
    video_path = artifact_dir / "inspect.mp4" if artifact_dir and generate_video else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if video_path:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise ValueError("Could not open MP4 video writer.")

    for pair_index, record in enumerate(selected_records[1:]):
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("aborted")
        current = _float01(compiled.run(record.file_path))
        if current.shape != prev.shape:
            raise ValueError(f"Preprocessing output size changed: expected {width}x{height}, got {current.shape[1]}x{current.shape[0]} for {record.file_name}.")
        if mode == "energy":
            values = _energy_values(prev, current, specs, config)
            overlay = _overlay_image(current, specs, {key: value for key, value in values.items() if key.startswith("tile_")})
        else:
            values, flow_rgb = _flow_values(prev, current, specs, config)
            overlay = flow_rgb if generate_video else _overlay_image(current, specs, {key: value for key, value in values.items() if key.startswith("tile_")})
        row = {
            "timestamp": record.timestamp_parsed.isoformat(),
            "frame_a": selected_records[pair_index].file_path,
            "frame_b": record.file_path,
            **values,
        }
        rows.append(row)
        if writer is not None and frames_dir is not None:
            stamped = add_timestamp_watermark(overlay, record.timestamp_parsed)
            Image.fromarray(stamped).save(frames_dir / f"frame_{pair_index:05d}.png", format="PNG")
            writer.write(cv2.cvtColor(stamped, cv2.COLOR_RGB2BGR))
        if progress_callback is not None:
            progress_callback(pair_index + 1)
        prev = current

    if writer is not None:
        writer.release()

    value_prefix = "energy_total" if mode == "energy" else "flow_selected"
    value_columns = [value_prefix] + [spec["key"] for spec in specs if spec["key"] != "total"]
    if mode == "energy" and str(config.get("energy_variant", "pairwise")) == "window":
        rows = _apply_window(rows, value_columns, max(1, int(config.get("window_size", 5))), str(config.get("window_aggregation", "sum")))
    example_values = {column: float(rows[0].get(column, 0.0) or 0.0) for column in value_columns} if rows else {}
    example_rgb = _overlay_image(first, specs, example_values)
    plot_rgb = _plot_preview(example_rgb, rows, specs, value_prefix)
    output = {
        "rows": rows,
        "columns": ["timestamp", "frame_a", "frame_b", *value_columns],
        "summary": {
            "mode": mode,
            "config": config,
            "roi": roi_meta,
            "series": _summarize(rows, value_columns),
            "samples": len(rows),
        },
        "example_rgb": example_rgb,
        "plot_rgb": plot_rgb,
        "video_path": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir) if frames_dir else None,
    }
    return output


def write_diagnostic_artifacts(result: dict[str, Any], artifact_dir: Path) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifact_dir / "results.csv"
    summary_path = artifact_dir / "summary.json"
    plot_path = artifact_dir / "plot-preview.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["columns"])
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({column: row.get(column) for column in result["columns"]})
    summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    Image.fromarray(result["plot_rgb"]).save(plot_path, format="PNG")
    return {
        "csv_path": str(csv_path),
        "summary_json_path": str(summary_path),
        "plot_preview_path": str(plot_path),
    }


def result_to_preview_response(result: dict[str, Any]) -> dict[str, Any]:
    first_row = result["rows"][0] if result["rows"] else {}
    return {
        "diagnostic_columns": result["columns"],
        "diagnostic_series": result["rows"],
        "image_data_url": encode_absolute_image_data_url(result["example_rgb"]),
        "plot_image_data_url": encode_absolute_image_data_url(result["plot_rgb"]),
        "first_timestamp": first_row.get("timestamp"),
    }
