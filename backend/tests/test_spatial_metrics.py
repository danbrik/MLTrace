from datetime import datetime, timedelta
import json
import threading

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import models
from app.analysis import spatial_sensitivity as spatial
from app.database import Base
from app.schemas import SpatialSensitivityRunCreate
from app.training.data import ResolvedDatasetImage


def test_four_formulas_transient_events_and_no_mad_scaling():
    normal = np.array([[[8, 8]], [[10, 10]], [[12, 12]]], dtype=np.uint16)
    event = np.array([[[0, 20]], [[20, 0]]], dtype=np.uint16)
    maps = spatial.compute_maps(normal, event)
    np.testing.assert_array_equal(maps['median_normal'], [[10, 10]])
    np.testing.assert_array_equal(maps['mad_normal'], [[2, 2]])
    np.testing.assert_array_equal(maps['D_med'], [[0, 0]])
    np.testing.assert_array_equal(maps['R_med'], [[0, 0]])
    np.testing.assert_array_equal(maps['D_q95'], [[10, 10]])
    np.testing.assert_allclose(maps['R_q95'], [[10 / 3, 10 / 3]])
    assert set(maps) == {'median_normal', 'mad_normal', 'median_event', *spatial.METRIC_TITLES}


def test_linear_q95_and_blockwise_match_numpy(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    normal = rng.integers(0, 65536, size=(7, 6, 8), dtype=np.uint16)
    event = rng.integers(0, 65536, size=(20, 6, 8), dtype=np.uint16)
    median = np.median(normal, axis=0)
    mad = np.median(np.abs(normal.astype(float) - median), axis=0)
    deviations = np.abs(event.astype(float) - median)
    expected = {'median_normal': median, 'mad_normal': mad, 'median_event': np.median(event, axis=0),
                'D_med': np.abs(np.median(event, axis=0) - median),
                'D_q95': np.quantile(deviations, .95, axis=0, method='linear'),
                'R_q95': np.quantile(deviations / (mad + 1), .95, axis=0, method='linear')}
    expected['R_med'] = expected['D_med'] / (mad + 1)
    monkeypatch.setattr(spatial, 'HEIGHT', 6); monkeypatch.setattr(spatial, 'WIDTH', 8)
    monkeypatch.setattr(spatial, 'TILE_ROWS', 2)
    normal_stack = np.lib.format.open_memmap(tmp_path / 'normal.npy', mode='w+', dtype=np.uint16, shape=normal.shape)
    event_stack = np.lib.format.open_memmap(tmp_path / 'event.npy', mode='w+', dtype=np.uint16, shape=event.shape)
    normal_stack[:] = normal; event_stack[:] = event
    abort = threading.Event()
    computed = spatial._compute_normal_staged(normal_stack, abort)
    computed.update(spatial._compute_event_staged(event_stack, computed, abort))
    for name, values in expected.items():
        np.testing.assert_array_equal(computed[name], values.astype(np.float32))
    direct = spatial.compute_maps(normal, event)
    for name in expected: np.testing.assert_array_equal(computed[name], direct[name])


def test_roi_ratios_and_zero_conventions():
    mask = np.array([[True, False]])
    maps = {name: np.array([[3, 1]], dtype=np.float32) for name in spatial.METRIC_TITLES}
    metrics = spatial.map_metrics(maps, mask)
    for name in maps: assert metrics[f'Q_{name}'] == 3
    for name in ('D_med', 'D_q95'): assert metrics[f'P_in_{name}'] == 75
    same = {name: np.ones((1, 2), dtype=np.float32) for name in maps}
    for name in maps: assert spatial.map_metrics(same, mask)[f'Q_{name}'] == 1
    zero = {name: np.zeros((1, 2), dtype=np.float32) for name in maps}
    metrics = spatial.map_metrics(zero, mask)
    assert all(metrics[f'Q_{name}'] == 0 for name in maps)
    assert metrics['P_in_D_med'] == metrics['P_in_D_q95'] == 0
    for name in maps: maps[name][0, 1] = 0
    assert spatial.map_metrics(maps, mask)['Q_R_q95'] == 3 / spatial.RATIO_FLOOR


def test_pooled_color_quantile_is_exact_preserves_unclipped_maps(tmp_path):
    store = np.lib.format.open_memmap(tmp_path / 'maps.npy', mode='w+', dtype=np.float32, shape=(2, 10, 10))
    values = np.arange(200, dtype=np.float32).reshape(2, 10, 10); values[-1, -1, -1] = 10000
    store[:] = values
    scratch = tmp_path / '.quantile.npy'
    vmax = spatial._shared_vmax(store, scratch, threading.Event())
    assert vmax == float(np.quantile(values, .995, method='linear'))
    assert vmax < values.max()
    np.testing.assert_array_equal(store, values)
    assert not scratch.exists()
    store[:] = 0
    assert spatial._shared_vmax(store, scratch, threading.Event()) == 1


def config_for(events, base):
    return {'training_dataset_ids':[1], 'events':events, 'normal_window_hours':1, 'epsilon':1.0,
            'roi_points':[{'x':1,'y':1},{'x':3,'y':1},{'x':3,'y':3},{'x':1,'y':3}],
            'roi_source_dataset_id':1, 'roi_source_timestamp':(base - timedelta(hours=1)).isoformat()}


def test_two_event_integration_artifacts_scales_and_medians(tmp_path, monkeypatch):
    monkeypatch.setattr(spatial, 'HEIGHT', 6); monkeypatch.setattr(spatial, 'WIDTH', 8)
    monkeypatch.setattr(spatial, 'TILE_ROWS', 2)
    base = datetime(2026, 1, 1, 12)
    events = [{'id':f'U{i+1}', 'training_dataset_id':1, 'start':(base + timedelta(minutes=i*120)).isoformat(),
               'end':(base + timedelta(minutes=i*120+20)).isoformat()} for i in range(2)]
    config = config_for(events, base)
    mask = spatial.build_roi_mask(config['roi_points'], (6, 8))
    records = []
    for event_index in range(2):
        for minute, value in ((-60, 8), (-30, 12), (0, 15), (10, 15), (20, 40)):
            image = np.full((6, 8), value if minute < 0 else 10, dtype=np.uint16)
            if minute >= 0: image[mask if event_index == 0 else ~mask] = value
            path = tmp_path / f'image_{event_index}_{minute}.tif'; Image.fromarray(image).save(path)
            records.append(ResolvedDatasetImage(str(path), base + timedelta(minutes=event_index*120+minute), 'dataset', str(tmp_path), 1, '.', path.name))
    monkeypatch.setattr(spatial, 'enumerate_training_dataset_image_records', lambda _: records)
    root = tmp_path / 'artifacts'
    monkeypatch.setattr(spatial, '_artifact_dir', lambda _: root)
    figures = {}
    def capture(fig, path):
        figures[path.name] = [(np.asarray(image.get_array()).copy(), image.get_clim(), len(ax.lines))
                              for ax in fig.axes for image in ax.images]
        if path.name == 'publication_overview': fig.savefig(tmp_path / 'overview.png', dpi=80)
        spatial._plotting().close(fig)
        return []
    monkeypatch.setattr(spatial, '_save_figure', capture)
    run = models.SpatialSensitivityRun(id=1, config=config)
    dataset = models.TrainingDataset(id=1, name='dataset', usage_label='test')
    result, csv, archive, valid, invalid = spatial.calculate(run, {1:dataset}, threading.Event(), lambda *args: None)
    assert valid == 10 and invalid == 0
    assert archive.is_file()
    assert result['analysis_version'] == spatial.ANALYSIS_VERSION
    first, second = result['events']
    for name in ('D_med', 'D_q95'):
        assert first[f'P_in_{name}'] == 100 and second[f'P_in_{name}'] == 0
    assert first['Q_R_q95'] > 1 and second['Q_R_q95'] == 0
    saved = [dict(np.load(root / f'event_{i:03d}_arrays.npz')) for i in (1, 2)]
    with np.load(root / 'aggregate_arrays.npz') as aggregates:
        assert set(aggregates) == set(spatial.AGGREGATE_NAMES.values())
        for name, aggregate in spatial.AGGREGATE_NAMES.items():
            expected = np.mean([event[name] for event in saved], axis=0, dtype=np.float64).astype(np.float32)
            np.testing.assert_array_equal(aggregates[aggregate], expected)
            assert result['vmax'][name] == float(np.quantile([event[name] for event in saved], .995))
            for key in (f'event_001_{name}', f'event_002_{name}', f'events_grid_{name}', aggregate):
                for array, limits, contours in figures[key]:
                    assert array.shape == (6, 8) and np.isfinite(array).all()
                    assert limits == (0, result['vmax'][name]) and contours == 1
    overview = figures['publication_overview']
    assert len(overview) == 13
    for index, name in enumerate(spatial.METRIC_TITLES):
        assert overview[5 + index][1] == overview[9 + index][1] == (0, result['vmax'][name])
    table = pd.read_csv(csv)
    for column in table.select_dtypes(include='number'):
        assert table.iloc[-1][column] == pytest.approx(np.median(table.iloc[:-1][column]))
    manifest = json.loads((root / 'input_manifest.json').read_text())
    assert manifest['epsilon'] == 1 and manifest['ratio_floor'] == 1e-12
    assert manifest['quantile_method'] == 'linear'
    assert not list(root.glob('.*'))


def test_legacy_configuration_does_not_reuse_old_results():
    base = datetime(2026, 1, 1, 12)
    config = config_for([{'id':'U1','training_dataset_id':1,'start':base.isoformat(),'end':base.isoformat()}], base)
    engine = create_engine('sqlite:///:memory:'); Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        legacy = {**config, 'epsilon':2, 'normal_sample_size':2000, 'sampling_seed':99}
        row = models.SpatialSensitivityConfiguration(name='legacy', config=legacy, config_signature='old-signature')
        run = models.SpatialSensitivityRun(status='finished', config=legacy, config_signature='old-signature', dataset_snapshot=[])
        db.add_all([row,run]); db.commit()
        current = spatial.get_configuration(db, row.id)
        assert current.latest_finished_run_id is None
        assert current.config['analysis_version'] == spatial.ANALYSIS_VERSION
        assert current.config['epsilon'] == 1 and current.config['normal_sample_size'] == 1000 and current.config['sampling_seed'] == 42
        assert row.config == legacy and run.config == legacy
        assert current.config_signature != 'old-signature'
    for patch in ({'epsilon':2}, {'sampling_seed':43}, {'normal_sample_size':1001}, {'event_sample_size':1001}):
        with pytest.raises(ValueError): SpatialSensitivityRunCreate(**{**config, **patch})
