import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app import models
from app.database import get_db
from app.main import app
from app.training.loss_plot import get_loss_plot, render_loss_plot
from app.schemas import TrainingRunMetricRead
from tests.test_testing_service import make_db
from tests.test_training_gradient import _seed_ae


@pytest.fixture
def source(tmp_path):
    db = make_db()
    run, _, _ = _seed_ae(db, tmp_path, 0)
    run.validation_mode = 'external'
    db.commit()
    yield db, run
    db.close()


def add_metrics(db, run, rows):
    for epoch, train, val in rows:
        db.add(models.TrainingRunMetric(training_run_id=run.id, epoch=epoch, train_loss=train, val_loss=val))
    db.commit()


def test_snapshot_sorted_finite_and_png_size(source):
    db, run = source
    add_metrics(db, run, [(3, .123456789012345, .4), (1, 1, 1), (2, float('inf'), None)])
    result = get_loss_plot(db, run.id)
    assert [m.epoch for m in result.metrics] == [1, 2, 3]
    assert result.metrics[1].train_loss is None
    assert result.metrics[-1].train_loss == .123456789012345
    image = Image.open(BytesIO(base64.b64decode(result.image_data_url.split(',')[1])))
    assert image.size == (1800, 1100)
    pixels = np.asarray(image)
    assert ((pixels[..., 0] == 0) & (pixels[..., 1] == 0) & (pixels[..., 2] == 255)).any()
    assert ((pixels[..., 0] == 255) & (pixels[..., 1] == 165) & (pixels[..., 2] == 0)).any()
    assert result.has_validation


def test_empty_single_epoch_and_training_only(source):
    db, run = source
    run.validation_mode = 'none'; db.commit()
    assert get_loss_plot(db, run.id).image_data_url is None
    add_metrics(db, run, [(1, -2.0, None)])
    result = get_loss_plot(db, run.id)
    assert result.image_data_url and not result.has_validation


def test_renderer_uses_gaps_labels_and_integer_ticks(monkeypatch):
    from matplotlib.figure import Figure
    original = Figure.savefig
    captured = []
    def inspect(fig, *args, **kwargs):
        ax = fig.axes[0]
        captured.append(ax)
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', inspect)
    rows = [TrainingRunMetricRead(epoch=1, train_loss=1, val_loss=2), TrainingRunMetricRead(epoch=3, train_loss=.4, val_loss=None)]
    render_loss_plot(rows, True)
    ax = captured[0]
    assert ax.get_xlabel() == 'Epochs' and ax.get_ylabel() == 'Loss'
    assert ax.get_xscale() == ax.get_yscale() == 'linear'
    assert [line.get_label() for line in ax.lines] == ['Train Loss', 'Validation Loss']
    assert np.isnan(ax.lines[0].get_ydata()[1])
    assert all(t == int(t) for t in ax.get_xticks())


def test_endpoint_updates_restarts_and_isolation(source):
    db, run = source
    app.dependency_overrides[get_db] = lambda: db
    try:
        client = TestClient(app)
        assert client.get(f'/api/training-runs/{run.id}/loss-plot').json()['metrics'] == []
        add_metrics(db, run, [(1, .5, .6)])
        for status in ['running', 'finished', 'failed', 'aborted']:
            run.status = status; db.commit()
            response = client.get(f'/api/training-runs/{run.id}/loss-plot')
            assert response.status_code == 200
            assert response.json()['metrics'][0]['val_loss'] == .6
        db.query(models.TrainingRunMetric).delete(); db.commit()
        assert client.get(f'/api/training-runs/{run.id}/loss-plot').json()['image_data_url'] is None
        assert client.get('/api/training-runs/99999/loss-plot').status_code == 404
        run.builder_kind = 'fast_anogan'; db.commit()
        assert client.get(f'/api/training-runs/{run.id}/loss-plot').status_code == 422
        other = make_db()
        try:
            app.dependency_overrides[get_db] = lambda: other
            assert client.get(f'/api/training-runs/{run.id}/loss-plot').status_code == 404
        finally: other.close()
    finally:
        app.dependency_overrides.pop(get_db, None)
