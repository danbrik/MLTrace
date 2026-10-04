import json
from io import BytesIO

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app import models
from app.database import get_db
from app.mean_variance import roi_engine as engine, roi_service
from app.mean_variance.roi_api import router
from app.mean_variance.roi_geometry import aligned_crop
from tests.test_reference_image import source
from tests.test_mean_variance import comparison
from tests.test_variance_roi import setup, completed, config


def basis(directory, count=1, limit=8):
    directory.mkdir()
    difference = np.tile(np.array([0., .5, 2, 8, -.5, -2, -8, 16]), (8, 1))
    for i in range(count):
        np.save(directory / f'{i}_mean.npy', np.full((8, 8), 42.))
        np.save(directory / f'{i}_difference.npy', difference)
    engine.finish_basis(directory, {}, {'version': 2, 'width': 8, 'height': 8,
        'difference_scale_limit': limit, 'pairs': [
            {'label': f'u{i+1}', 'periods': {}, 'counts': {}} for i in range(count)]}, 'Data', 'Pipeline')
    return difference


def test_local_heatmap_preserves_zero_background_and_signs(tmp_path):
    difference = basis(tmp_path / 'basis')
    directory = tmp_path / 'basis'
    background = np.asarray(Image.open(directory / '0_background.png'))
    color = np.asarray(Image.open(directory / '0_heatmap.png'))
    image = engine.composite(directory, 0, .8, 'local', 1)
    np.testing.assert_array_equal(image[:, 0], background[:, 0])
    np.testing.assert_array_equal(engine.composite(directory, 0, 0, 'local', 4), background)
    alpha = .8 * np.minimum(np.abs(difference) / 8, 1)[..., None]
    np.testing.assert_array_equal(image, np.rint(background * (1-alpha) + color * alpha))
    assert image[0, 3, 0] > image[0, 3, 2] and image[0, 6, 2] > image[0, 6, 0]
    np.testing.assert_array_equal(image[:, 3], image[:, 7])  # manual display saturation
    distances = [np.linalg.norm(engine.composite(directory, 0, .8, 'local', s)[0, 1].astype(float) - background[0, 1]) for s in [.25, 1, 4]]
    assert distances[0] < distances[1] < distances[2]
    np.testing.assert_array_equal(np.load(directory / '0_difference.npy'), difference)


def test_zero_scale_is_background_and_legacy_keeps_white_overlay(tmp_path):
    directory = tmp_path / 'basis'; basis(directory, limit=0)
    background = np.asarray(Image.open(directory / '0_background.png'))
    np.testing.assert_array_equal(engine.composite(directory, 0, 1, 'local', 4), background)
    old = engine.composite(directory, 0, .5)
    color = np.asarray(Image.open(directory / '0_heatmap.png'))
    np.testing.assert_array_equal(old, np.rint(background * .5 + color * .5))
    assert engine.RoiConfig(roi={'x': 0, 'y': 0, 'width': 1, 'height': 1}).heatmap_mode == 'global'


@pytest.mark.parametrize('change', [{'opacity': -1}, {'opacity': 1.01}, {'sensitivity': .24},
    {'sensitivity': 4.01}, {'sensitivity': float('nan')}, {'opacity': float('inf')}, {'heatmap_mode': 'unknown'}])
def test_display_validation(change):
    with pytest.raises(ValidationError):
        engine.HeatmapDisplay(**change)


@pytest.mark.parametrize('count', [1, 6])
def test_export_matches_preview_pixels_and_metrics_ignore_display(tmp_path, monkeypatch, count):
    from matplotlib.figure import Figure
    directory = tmp_path / 'basis'; basis(directory, count)
    rectangle = {'version': 2, 'center_x': 4, 'center_y': 4, 'width': 4, 'height': 4, 'angle_degrees': 30}
    cfg = engine.RoiConfig(roi=rectangle, heatmap_mode='local', opacity=.7, sensitivity=3)
    original = Figure.savefig
    def inspect(fig, path, *args, **kwargs):
        if path.name == 'roi_comparison.png':
            for index in range(count):
                pixels = engine.composite(directory, index, .7, 'local', 3)
                np.testing.assert_array_equal(fig.axes[2*index].images[0].get_array(), pixels)
                np.testing.assert_array_equal(fig.axes[2*index+1].images[0].get_array(), aligned_crop(pixels, cfg.roi))
        return original(fig, path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Figure, 'savefig', inspect)
        result = engine.export_roi(directory, tmp_path, cfg)
    for name in ['roi_comparison.png', 'roi_table.png']:
        with Image.open(tmp_path / name) as image:
            metadata = json.loads(image.info['Description'])
            assert metadata['sensitivity'] == 3 and metadata['heatmap_mode'] == 'local' and metadata['opacity'] == .7
    other = engine.export_roi(directory, tmp_path, engine.RoiConfig(roi=rectangle, heatmap_mode='local', opacity=0, sensitivity=.25))
    assert result['rows'] == other['rows'] and result['mean_increase_percent'] == other['mean_increase_percent']


def test_preview_api_validation_export_roundtrip_and_queued_legacy(setup, monkeypatch):
    db, _, _ = setup
    run = completed(setup)
    app = FastAPI(); app.include_router(router); app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app)
    root = f'/api/mean-variance-analysis/runs/{run.id}/roi'
    url = root + '/images/0/composite'
    response = client.get(url, params={'opacity': .75, 'sensitivity': 2, 'heatmap_mode': 'local'})
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    np.testing.assert_array_equal(np.array(Image.open(BytesIO(response.content))),
                                 engine.composite(roi_service.basis_dir(run.id), 0, .75, 'local', 2))
    for params in [{'opacity': 'nan'}, {'sensitivity': 'inf'}, {'sensitivity': 0}, {'opacity': 2}, {'heatmap_mode': 'bad'}]:
        assert client.get(url, params=params).status_code == 422
    assert client.get(root + '/images/99/composite').status_code == 404
    assert client.get(root + '/images/0/background').status_code == 200
    assert client.get(root + '/images/0/heatmap').status_code == 200
    cfg = config(opacity=.75, heatmap_mode='local', sensitivity=2)
    job = client.post(root + '/evaluate', json=cfg.model_dump()).json()
    roi_service.run_scheduled(job['id']); db.expire_all()
    saved = client.get(root).json()['saved']
    assert saved['status'] == 'finished' and saved['result']['sensitivity'] == 2
    assert saved['config']['heatmap_mode'] == 'local'
    artifact = roi_service.artifact_dir(job['id']) / 'roi_comparison.png'
    previous = artifact.read_bytes()
    legacy = roi_service.enqueue(db, run.id, 'evaluate', config(), wake_scheduler=False)
    record = db.get(models.VarianceRoiJob, legacy['id'])
    record.config = {k: v for k, v in record.config.items() if k not in ('heatmap_mode', 'sensitivity')}
    db.commit()
    seen = []
    original = engine.composite
    def capture(directory, index, opacity, mode='global', sensitivity=1):
        seen.append(mode)
        return original(directory, index, opacity, mode, sensitivity)
    monkeypatch.setattr(engine, 'composite', capture)
    roi_service.run_scheduled(legacy['id']); db.expire_all()
    assert roi_service.state(db, run.id)['saved']['result']['heatmap_mode'] == 'global'
    assert seen == ['global'] and artifact.read_bytes() == previous
