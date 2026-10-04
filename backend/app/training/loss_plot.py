"""Read-only, in-memory exports of persisted epoch metrics."""
import base64
import math
from io import BytesIO
from threading import Lock

from sqlalchemy import select
from app import models
from app.schemas import TrainingLossPlotRead, TrainingRunMetricRead

SUPPORTED_BUILDERS = frozenset({
    'sequential_autoencoder', 'sequential_spatial_autoencoder',
    'sequential_variational_autoencoder', 'spatiotemporal_autoencoder',
})
_RENDER_LOCK = Lock()


def finite_loss(value):
    return float(value) if value is not None and math.isfinite(value) else None


def render_loss_plot(metrics, has_validation):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import MaxNLocator

    # Explicit Figure/Agg avoids pyplot's global figure registry. Serialize font rendering.
    with _RENDER_LOCK:
        fig = Figure(figsize=(9, 5.5), dpi=200, facecolor='white')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        epochs = [m.epoch for m in metrics]
        # Include missing epochs as gaps rather than drawing an interpolated segment.
        x, training, validation = [], [], []
        previous = None
        for metric in metrics:
            if previous is not None and metric.epoch > previous + 1:
                x.append(previous + 1); training.append(math.nan); validation.append(math.nan)
            x.append(metric.epoch)
            training.append(metric.train_loss if metric.train_loss is not None else math.nan)
            validation.append(metric.val_loss if metric.val_loss is not None else math.nan)
            previous = metric.epoch
        ax.plot(x, training, color='blue', marker='o', markersize=4, linewidth=1.5, label='Train Loss')
        if has_validation:
            ax.plot(x, validation, color='orange', marker='o', markersize=4, linewidth=1.5, label='Validation Loss')
        ax.set_xlabel('Epochs'); ax.set_ylabel('Loss')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(epochs) == 1:
            ax.set_xlim(epochs[0] - .5, epochs[0] + .5)
            ax.set_xticks([epochs[0]])
        ax.set_axisbelow(True)
        ax.grid(True, color='#b0b0b0', linewidth=.7)
        ax.legend(loc='upper right')
        fig.tight_layout()
        output = BytesIO()
        fig.savefig(output, format='png', dpi=200)
        return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')


def get_loss_plot(db, run_id):
    run = db.get(models.TrainingRun, run_id)
    if run is None:
        return None
    if run.training_mode != 'gradient' or run.builder_kind not in SUPPORTED_BUILDERS:
        raise ValueError('Dieser Lauf unterstützt keinen Train-/Validation-Loss-Plot.')
    rows = db.execute(select(models.TrainingRunMetric.epoch, models.TrainingRunMetric.train_loss,
        models.TrainingRunMetric.val_loss).where(models.TrainingRunMetric.training_run_id == run_id)
        .order_by(models.TrainingRunMetric.epoch)).all()
    metrics = [TrainingRunMetricRead(epoch=row.epoch, train_loss=finite_loss(row.train_loss),
        val_loss=finite_loss(row.val_loss)) for row in rows]
    has_validation = any(m.val_loss is not None for m in metrics) or run.validation_mode == 'external' or (
        run.validation_mode == 'legacy_fraction' and float((run.training_parameters or {}).get('validation_fraction', 0) or 0) > 0)
    has_values = any(m.train_loss is not None or m.val_loss is not None for m in metrics)
    return TrainingLossPlotRead(run_id=run.id, metrics=metrics, has_validation=has_validation,
        image_data_url=render_loss_plot(metrics, has_validation) if has_values else None)
