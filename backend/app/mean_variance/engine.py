"""Streaming population moments and static heatmaps, without intensity transforms."""
import json
import textwrap

import numpy as np

from app.reference_image.engine import grayscale


class OnlineMoments:
    """Welford's algorithm: O(image size) memory, population variance (ddof=0)."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, image):
        array = grayscale(image, self.mean.shape if self.mean is not None else None)
        if self.mean is None:
            self.mean = array.copy()
            self.m2 = np.zeros_like(array)
            self.count = 1
            return
        self.count += 1
        with np.errstate(over="ignore", invalid="ignore"):
            delta = array - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (array - self.mean)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.m2).all():
            raise ValueError("Mittelwert oder Varianz enthält nicht endliche Werte.")

    def finish(self):
        if self.count == 0:
            raise ValueError("Der Zeitraum enthält keine ausgewählten Bilder.")
        return self.mean, np.maximum(self.m2 / self.count, 0)


def difference_maps(normal, anomaly):
    normal_mean, normal_var = normal
    anomaly_mean, anomaly_var = anomaly
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.abs(normal_mean - anomaly_mean)
        variance = anomaly_var - normal_var
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError("Die Differenz enthält nicht endliche Pixelwerte.")
    return mean, variance


def heatmap_style(values, scale, signed):
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    maximum = float(np.abs(values).max())
    limit = maximum if scale.mode == "auto" else scale.limit
    # A zero-only map has a degenerate data range; use a display-only denominator.
    display_limit = limit if limit > 0 else 1.0
    cmap = LinearSegmentedColormap.from_list("variance_change", ["#2166ac", "#ffffff", "#b2182b"], N=257) if signed else colormaps["viridis"]
    norm = Normalize(vmin=-display_limit if signed else 0, vmax=display_limit, clip=True)
    return cmap, norm, {"scale_limit": limit, "maximum_absolute": maximum,
                        "minimum": float(values.min()), "maximum": float(values.max()), "all_zero": maximum == 0}


def render_heatmap(values, scale, signed, path, config, counts, dataset_name):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    cmap, norm, result = heatmap_style(values, scale, signed)
    title = "Varianzdifferenz · Anomalie − Normalphase" if signed else "Absoluter Mittelwertunterschied"
    unit = "Pipeline-Einheiten²" if signed else "Pipeline-Einheiten"
    normal, anomaly = config.reference, config.anomaly
    describe = lambda interval: f"{interval.start.isoformat(sep=' ')} – {interval.end.isoformat(sep=' ')}"
    caption = (f"Normalphase: {describe(normal)} · {counts['reference']} Bilder\n"
               f"Anomaliephase: {describe(anomaly)} · {counts['anomaly']} Bilder")
    notes = []
    if result["all_zero"]:
        notes.append("Keine Unterschiede: alle Pixelwerte sind 0.")
    if 1 in counts.values():
        notes.append("Zeitraum mit nur einem Bild: Varianz 0, keine zeitliche Vergleichsbasis.")
    fig = Figure(figsize=(10, 8), dpi=160)
    FigureCanvasAgg(fig)
    try:
        ax = fig.add_axes((.08, .22, .73, .57))
        shown = ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest", origin="upper", aspect="equal")
        ax.set_xlabel("x (Pixel)"); ax.set_ylabel("y (Pixel)")
        fig.suptitle(title, y=.97, fontsize=14)
        fig.text(.5, .925, textwrap.fill(dataset_name, 90), ha="center", va="top", fontsize=10)
        colorbar = fig.colorbar(shown, cax=fig.add_axes((.85, .25, .025, .5)))
        colorbar.set_label(unit)
        if result["all_zero"] and scale.mode == "auto":
            colorbar.set_ticks([0])
        fig.text(.08, .135, caption, fontsize=9, va="top")
        if notes:
            fig.text(.08, .06, "\n".join(notes), fontsize=9, va="top")
        description = {"title": title, "unit": unit, "periods": caption, "counts": counts,
                       "ddof": 0, "origin": "upper", "interpolation": "nearest", **result}
        fig.savefig(path, format="png", metadata={"Title": title, "Description": json.dumps(description, ensure_ascii=False)})
    finally:
        fig.clear()
    return {**result, "filename": path.name, "title": title, "unit": unit}
