"""Shared ROI presentation and positive-increase metrics on the original pixel grid."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from app.mean_variance.engine import comparison_style
from app.mean_variance.roi_geometry import RotatedRectangle, oriented_roi, selection_mask, aligned_crop


class Rectangle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0, strict=True)
    y: int = Field(ge=0, strict=True)
    width: int = Field(ge=1, strict=True)
    height: int = Field(ge=1, strict=True)

    def validate_bounds(self, width, height):
        if self.x + self.width > width or self.y + self.height > height:
            raise ValueError("Die ROI muss vollständig innerhalb des Bildes liegen.")

    @property
    def slices(self):
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))


class HeatmapDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opacity: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False)
    # Missing mode denotes an existing job and retains its original rendering.
    heatmap_mode: Literal["global", "local"] = "global"
    sensitivity: float = Field(default=1, ge=.25, le=4, allow_inf_nan=False)


class RoiConfig(HeatmapDisplay):
    roi: Rectangle | RotatedRectangle


def load_basis(directory):
    return json.loads((directory / "basis.json").read_text(encoding="utf-8"))


def finish_basis(directory, config, result, dataset_name, pipeline_name, check_abort=lambda: None):
    """Finalize immutable numerical data and display layers before publishing them."""
    if result.get("version") == 2:
        pairs = [{key: pair[key] for key in ("label", "periods", "counts")} for pair in result["pairs"]]
        limit = result["difference_scale_limit"]
    else:
        pairs = [{"label": "u1", "periods": {"normal": config["reference"], "anomaly": config["anomaly"]},
                  "counts": {"normal": result["reference_count"], "anomaly": result["anomaly_count"]}}]
        limit = result["maps"]["variance"]["scale_limit"]
    low, high = float("inf"), float("-inf")
    for index in range(len(pairs)):
        check_abort()
        mean = np.load(directory / f"{index}_mean.npy", mmap_mode="r")
        difference = np.load(directory / f"{index}_difference.npy", mmap_mode="r")
        if mean.shape != (result["height"], result["width"]) or difference.shape != mean.shape:
            raise ValueError("Die Bildgrößen der ROI-Daten stimmen nicht überein.")
        if not np.isfinite(mean).all() or not np.isfinite(difference).all():
            raise ValueError("ROI-Daten enthalten nicht endliche Werte.")
        low, high = min(low, float(mean.min())), max(high, float(mean.max()))
    basis = {"version": 1, "width": result["width"], "height": result["height"], "pairs": pairs,
             "difference_scale_limit": limit, "background_min": low, "background_max": high,
             "dataset_name": dataset_name, "pipeline_name": pipeline_name, "config": config,
             "unit": "gray value²"}
    cmap, norm = comparison_style(limit, True)
    for index in range(len(pairs)):
        check_abort()
        mean = np.load(directory / f"{index}_mean.npy", mmap_mode="r")
        gray = np.full(mean.shape, 128, dtype=np.uint8) if high == low else np.rint((mean / 2 - low / 2) / (high / 2 - low / 2) * 255).astype(np.uint8)
        Image.fromarray(gray).convert("RGB").save(directory / f"{index}_background.png")
        difference = np.load(directory / f"{index}_difference.npy", mmap_mode="r")
        color = np.rint(cmap(norm(difference))[..., :3] * 255).astype(np.uint8)
        Image.fromarray(color).save(directory / f"{index}_heatmap.png")
    check_abort()
    (directory / "basis.json").write_text(json.dumps(basis, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return basis


def positive_metrics(difference, roi, mask=None):
    if mask is None:
        mask = selection_mask(roi, difference.shape[1], difference.shape[0])
    positive = np.maximum(difference, 0)
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(positive, dtype=np.float64))
        inside = float(np.sum(positive[mask], dtype=np.float64))
    if not np.isfinite(total) or not np.isfinite(inside):
        raise ValueError("Die Summe der Varianzzunahme ist nicht endlich.")
    return {"positive_total": total, "positive_roi": inside,
            "selected_pixels": int(mask.sum()),
            "area_percent": int(mask.sum()) / difference.size * 100,
            "increase_percent": inside / total * 100 if total > 0 else None}


def composite(directory, index, opacity, heatmap_mode="global", sensitivity=1):
    """The editor preview and export share exactly the same presentation pixels."""
    display = HeatmapDisplay(opacity=opacity, heatmap_mode=heatmap_mode, sensitivity=sensitivity)
    alpha = display.opacity
    if display.heatmap_mode == "local":
        limit = load_basis(directory)["difference_scale_limit"]
        difference = np.load(directory / f"{index}_difference.npy", mmap_mode="r")
        if not np.isfinite(difference).all():
            raise ValueError("ROI-Daten enthalten nicht endliche Werte.")
        if limit > 0:
            # Clip before division to avoid overflow for very small manual limits.
            magnitude = np.minimum(np.abs(difference), limit) / limit
            alpha = (display.opacity * np.power(magnitude, 1 / display.sensitivity))[..., None]
        else:
            alpha = 0
    with Image.open(directory / f"{index}_background.png") as background, Image.open(directory / f"{index}_heatmap.png") as heatmap:
        return np.rint(np.asarray(background, dtype=np.float64) * (1 - alpha)
                       + np.asarray(heatmap, dtype=np.float64) * alpha).astype(np.uint8)


def export_roi(directory: Path, output: Path, config: RoiConfig, check_abort=lambda: None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Polygon

    basis = load_basis(directory)
    roi = config.roi
    mask = selection_mask(roi, basis["width"], basis["height"])
    rows = []
    for index, pair in enumerate(basis["pairs"]):
        check_abort()
        values = np.load(directory / f"{index}_difference.npy", mmap_mode="r")
        rows.append({"label": pair["label"], **positive_metrics(values, roi, mask)})
    valid = [row["increase_percent"] for row in rows if row["increase_percent"] is not None]
    result = {**basis, **config.model_dump(), "roi": oriented_roi(roi).model_dump(), "rows": rows, "valid_pairs": len(valid),
              "area_percent": rows[0]["area_percent"], "mean_increase_percent": float(np.mean(valid)) if valid else None,
              "plot": "roi_comparison.png", "table": "roi_table.png",
              "roi_corners": oriented_roi(roi).corners().tolist(), "selected_pixels": int(mask.sum()),
              "output_width": roi.width, "output_height": roi.height, "resampling": "nearest",
              "warnings": [f"{row['label']}: Keine positive Varianzzunahme; nicht im Mittelwert enthalten." for row in rows if row["increase_percent"] is None]}
    metadata = {"Title": "ROI-Auswertung", "Description": json.dumps(result, ensure_ascii=False, allow_nan=False)}
    count = len(rows)
    row_height = min(6, max(1.5, 4.6 * max(basis["height"] / basis["width"], roi.height / roi.width)))
    height = 1.9 + count * row_height + (count - 1) * .3
    fig = Figure(figsize=(12, height), dpi=180)
    FigureCanvasAgg(fig)
    try:
        for index, pair in enumerate(basis["pairs"]):
            check_abort()
            pixels = composite(directory, index, config.opacity, config.heatmap_mode, config.sensitivity)
            y = height - .5 - row_height - index * (row_height + .3)
            fig.text(.025, (y + row_height / 2) / height, pair["label"], va="center", fontsize=12)
            for column, data in enumerate((pixels, aligned_crop(pixels, roi))):
                ax = fig.add_axes(((1.2 + column * 5.25) / 12, y / height, 4.9 / 12, row_height / height))
                ax.imshow(data, origin="upper", interpolation="nearest", aspect="equal")
                if column == 0:
                    ax.add_patch(Polygon(oriented_roi(roi).corners() - .5, closed=True,
                                       fill=False, edgecolor="#ff9800", linewidth=1.5, clip_on=False))
                ax.tick_params(labelsize=9, labelleft=column == 0, left=column == 0,
                               labelbottom=index == count - 1, bottom=index == count - 1)
                if index == 0:
                    fig.text((1.2 + column * 5.25 + 4.9 / 2) / 12, 1 - .22 / height,
                             ("Gesamtbild mit ROI", "ROI-Ausschnitt")[column], ha="center", fontsize=14)
        fig.text(.062, .55, "y (Pixel)", rotation=90, va="center", ha="center", fontsize=11)
        fig.text(.55, .98 / height, "x (Pixel)", ha="center", fontsize=11)
        cmap, norm = comparison_style(basis["difference_scale_limit"], True)
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                           cax=fig.add_axes((.23, .58 / height, .6, .14 / height)), orientation="horizontal")
        bar.set_label("Variance difference (gray value²)", fontsize=11)
        if basis["difference_scale_limit"] == 0:
            bar.set_ticks([0])
        else:
            bar.formatter.set_powerlimits((-3, 4)); bar.formatter.set_useMathText(True); bar.update_ticks()
        check_abort()
        fig.savefig(output / result["plot"], metadata=metadata)
    finally:
        fig.clear()
    check_abort()
    table_figure = Figure(figsize=(10, 1.8 + (count + 1) * .45), dpi=180)
    FigureCanvasAgg(table_figure)
    try:
        ax = table_figure.add_axes((.025, .23, .95, .65)); ax.axis("off")
        fmt = lambda value: "—" if value is None else f"{value:.2f}"
        cells = [[row["label"], fmt(row["area_percent"]), fmt(row["increase_percent"])] for row in rows]
        cells.append(["Mean", fmt(result["area_percent"]), fmt(result["mean_increase_percent"])])
        table = ax.table(cellText=cells, colLabels=["Unruhe", "ROI-Flächenanteil (%)", "Anteil der Varianzzunahme (%)"],
                         colWidths=[.15, .32, .53], cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if row in (0, count + 1):
                cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold")
        table_figure.text(.5, .95, "ROI-Auswertung · positive Varianzzunahme", ha="center", va="top", fontsize=14)
        note = f"Mean: {len(valid)} von {count} Paaren"
        if len(valid) < count:
            note += " · „—“: keine positive Varianzzunahme"
        table_figure.text(.5, .09, note, ha="center", fontsize=10)
        table_figure.savefig(output / result["table"], metadata=metadata)
    finally:
        table_figure.clear()
    check_abort()
    return result
