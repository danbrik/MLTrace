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


def map_statistics(values):
    maximum_absolute = float(np.abs(values).max())
    return {"minimum": float(values.min()), "maximum": float(values.max()),
            "maximum_absolute": maximum_absolute, "all_zero": maximum_absolute == 0}


def comparison_style(limit, signed):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    colors = ["#2166ac", "#ffffff", "#b2182b"] if signed else ["#ffffff", "#b2182b"]
    cmap = LinearSegmentedColormap.from_list("variance_signed" if signed else "variance", colors, N=257)
    return cmap, Normalize(vmin=-(limit or 1) if signed else 0, vmax=limit or 1, clip=True)


def render_comparison(pairs, config, path, dataset_name, pipeline_name, shape, check_abort=lambda: None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    maximum = max(pair["maps"][role]["maximum"] for pair in pairs for role in ("normal", "anomaly"))
    difference_maximum = max(pair["maps"]["difference"]["maximum_absolute"] for pair in pairs)
    variance_limit = maximum if config.variance_scale.mode == "auto" else config.variance_scale.limit
    difference_limit = difference_maximum if config.difference_scale.mode == "auto" else config.difference_scale.limit
    styles = [comparison_style(variance_limit, False), comparison_style(difference_limit, True)]
    width, left, right, gap = 12.0, 1.3, .25, .22
    cell_width = (width - left - right - 2 * gap) / 3
    cell_height = cell_width * shape[0] / shape[1]
    bottom, top = 1.5, .48
    height = bottom + top + len(pairs) * cell_height + (len(pairs) - 1) * .22
    result = {"version": 2, "filename": path.name, "unit": "gray value²", "ddof": 0,
              "width": shape[1], "height": shape[0], "dataset_name": dataset_name, "pipeline_name": pipeline_name,
              "origin": "upper", "interpolation": "nearest", "sampling_rate": config.sampling_rate,
              "variance_scale_limit": variance_limit, "difference_scale_limit": difference_limit,
              "pairs": [{**pair, "maps": {role: {k: v for k, v in stats.items() if k != "path"}
                                         for role, stats in pair["maps"].items()}} for pair in pairs]}
    fig = Figure(figsize=(width, height), dpi=180)
    FigureCanvasAgg(fig)
    try:
        for row, pair in enumerate(pairs):
            y = height - top - cell_height - row * (cell_height + .22)
            fig.text(.025, (y + cell_height / 2) / height, pair["label"], va="center", fontsize=12)
            for column, role in enumerate(("normal", "anomaly", "difference")):
                check_abort()
                ax = fig.add_axes(((left + column * (cell_width + gap)) / width, y / height,
                                   cell_width / width, cell_height / height))
                values = np.load(pair["maps"][role]["path"], mmap_mode="r")
                cmap, norm = styles[1 if column == 2 else 0]
                ax.imshow(values, cmap=cmap, norm=norm, origin="upper", interpolation="nearest", aspect="equal")
                ax.tick_params(labelsize=9, labelleft=column == 0, left=column == 0,
                               labelbottom=row == len(pairs) - 1, bottom=row == len(pairs) - 1)
                if row == 0:
                    ax.set_title(("Normalzustand", "Unruhe", "Differenz")[column], fontsize=14, pad=10)
        fig.text(.065, (bottom + (height - bottom - top) / 2) / height, "y (Pixel)", rotation=90, va="center", ha="center", fontsize=11)
        fig.text((left + (width - left - right) / 2) / width, 1.02 / height, "x (Pixel)", ha="center", fontsize=11)
        for index, (x, bar_width, label, limit) in enumerate((
            (left, cell_width * 2 + gap, "Variance (gray value²)", variance_limit),
            (left + 2 * (cell_width + gap), cell_width, "Variance difference (gray value²)", difference_limit),
        )):
            cmap, norm = styles[index]
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                               cax=fig.add_axes((x / width, .6 / height, bar_width / width, .14 / height)), orientation="horizontal")
            bar.set_label(label, fontsize=11)
            bar.ax.tick_params(labelsize=9)
            if limit == 0:
                bar.set_ticks([0])
            else:
                bar.formatter.set_powerlimits((-3, 4)); bar.formatter.set_useMathText(True); bar.update_ticks()
        check_abort()
        fig.savefig(path, format="png", metadata={"Title": "Varianzvergleich", "Description": json.dumps(result, ensure_ascii=False)})
        check_abort()
    finally:
        fig.clear()
    return result
