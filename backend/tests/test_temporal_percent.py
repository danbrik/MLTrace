import csv
import json
import numpy as np
import pytest
from app.temporal_difference import engine, service, matrix
from app.temporal_difference.units import to_percent
from tests.test_temporal_difference import temporal
from tests.test_reference_image import source


def test_fixed_scale_unsigned_and_unclipped():
    np.testing.assert_allclose(to_percent(np.array([0, 655.35, 65535, 131070])), [0, 1, 100, 200])
    a, b = np.array([[0]], dtype=np.uint16), np.array([[65535]], dtype=np.uint16)
    assert to_percent(engine.absolute_change(a, b)) == 100
    assert to_percent(engine.absolute_change(b, a)) == 100
    assert to_percent(None) is None
    values = np.array([[0., 1., 5., 100.]])
    np.testing.assert_array_equal(np.ma.getmaskarray(matrix.filtered_difference(values, 25)), np.ma.getmaskarray(matrix.filtered_difference(to_percent(values), 25)))


def test_percent_api_exports_and_raw_compatibility(temporal, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import get_db
    db, cfg, _ = temporal
    run = service.enqueue(db, cfg, wake_scheduler=False)
    service.run_scheduled(run.id); db.expire_all()
    raw = service.summaries(db, run.id)
    def no_images(*args):
        raise AssertionError('Percent display must not reprocess images')
    monkeypatch.setattr(service, 'read_frozen_image', no_images)
    app.dependency_overrides[get_db] = lambda: db
    try:
        client = TestClient(app)
        base = f'/api/temporal-difference/runs/{run.id}'
        assert client.get(base+'/summary').json() == raw
        percentages = client.get(base+'/summary?unit=percent').json()
        for before, after in zip(raw, percentages):
            assert after['pair_count'] == before['pair_count']
            for key in ('median', 'q1', 'q3', 'iqr'):
                assert after[key] == to_percent(before[key])
        pairs = client.get(base+'/pairs?unit=percent').json()['items']
        raw_pairs = client.get(base+'/pairs').json()['items']
        assert pairs[0]['value'] == to_percent(raw_pairs[0]['value'])
        response = client.get(base+'/csv/pairs?unit=percent')
        rows = list(csv.DictReader(response.text.splitlines()))
        assert float(rows[0]['value_percent']) == pairs[0]['value']
        assert rows[0]['role'] in ('Normal', 'Anomalie')
        summary_csv = client.get(base+'/csv/summary?unit=percent').text
        assert 'normal_median_percent' in summary_csv and 'anomaly_iqr_percent' in summary_csv
        assert client.get(base+'/summary?unit=unknown').status_code == 422
        assert service.summaries(db, run.id) == raw
        assert client.get(base+'/csv/pairs').content == service.csv_path(db, run.id, 'pairs').read_bytes()
        plot = client.get(base).json()['plot_settings']
        assert plot['unit_version'] == 2
        assert plot['y_title'] == 'Mittlere Pixeländerung (%)'
        assert client.put(base+'/plot-settings', json={**plot, 'y_range': {'minimum': 0, 'maximum': 3}}).status_code == 200
        assert client.get(base).json()['plot_settings']['y_range']['maximum'] == 3
        directory = service.artifact_dir(run.id)
        (directory/'matrix-reference.json').write_text(json.dumps({'artifact':'old.png', 'config':None, 'warnings':[]}))
        old = matrix.state(db, run.id, 'reference')
        assert old['unit'] == 'raw' and old['render_version'] == 1
    finally:
        app.dependency_overrides.pop(get_db, None)
