"""Result paging must read bounded blocks once and preserve endpoint alignment."""
from collections import Counter
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.time_series import results


@pytest.fixture
def stored_run(tmp_path, monkeypatch):
    monkeypatch.setattr(results, 'artifact_dir', lambda _: tmp_path)
    values = np.arange(40, dtype=float).reshape(20, 2)
    scaled = (values - [10, 100]) / [2, 20]
    times = np.datetime64('2026-01-01T00:00:00', 'ns').astype(np.int64) + np.arange(20) * 300_000_000_000
    groups = np.array([0]*5 + [1]*6 + [2]*9)
    endpoints = np.r_[2:5, 7:11, 13:20]
    np.savez_compressed(tmp_path/'input.npz', values=values, scaled=scaled, timestamps=times,
                        groups=groups, segment_ids=groups, interval_indices=groups, endpoints=endpoints)
    chunks = []
    for part, start in enumerate(range(0, len(endpoints), 4)):
        ends = endpoints[start:start+4]
        name = f'results-{part:05d}.npz'
        np.savez_compressed(tmp_path/name, end_indices=ends, window_score=ends/10, endpoint_score=ends/20,
                            reconstruction=scaled[ends]+1, cascade=scaled[ends]+2, second=scaled[ends]+3,
                            variance=np.full((len(ends),2), .25), latent=np.stack([ends,ends+100], axis=1))
        chunks.append({'file':name,'count':len(ends)})
    (tmp_path/'manifest.json').write_text(json.dumps({'version':1, 'sensor_order':['a','b'], 'chunks':chunks}))
    run = SimpleNamespace(id=1,status='finished',snapshot={'window_length':3, 'preview':{'scaler':{'minimum':[10,100], 'denominator':[2,20]}},
        'split':{'intervals':[{'tags':['train']},{'tags':['validation']},{'tags':['test']}]}})
    return run


def test_series_reads_only_page_blocks_and_decompresses_members_once(stored_run, monkeypatch):
    expected = list(results.result_rows(stored_run, 'test', sensor=1))[1:3]
    counts = Counter()
    real_load = np.load
    class Archive:
        def __init__(self, path):
            self.path, self.archive = path, real_load(path)
            self.files = self.archive.files
        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def __getitem__(self, key):
            counts[(self.path.name,key)] += 1
            return self.archive[key]
    monkeypatch.setattr(results.np, 'load', Archive)
    page = results.series(stored_run, sensor=1, offset=1, limit=2)
    assert page['rows'] == expected
    assert page['total'] == 7
    assert {name for name,key in counts} == {'input.npz','results-00002.npz'}
    assert set(counts.values()) == {1}
    assert all(key != 'latent' for name,key in counts)
    assert page['rows'][0]['original'] == 29
    assert page['rows'][0]['reconstruction'] == 49
    assert page['rows'][0]['std'] == 10


@pytest.mark.parametrize('subset', ['train','validation','test'])
@pytest.mark.parametrize('scaled', [True,False])
def test_pages_equal_full_export_and_preserve_tags_latents(stored_run, subset, scaled):
    full = list(results.result_rows(stored_run, subset, sensor=1, scaled=scaled))
    pages = [results.series(stored_run,subset,sensor=1,scaled=scaled,offset=i,limit=2) for i in range(0,len(full),2)]
    assert [r for page in pages for r in page['rows']] == full
    assert all(page['total'] == len(full) for page in pages)
    for row in results.result_rows(stored_run, subset, include_latent=True):
        assert row['z_sensor'] == [row['endpoint_index'], row['endpoint_index']+100]
        assert row['tags'] == [subset]
        assert row['timestamp'] == row['window_end']
    empty = results.series(stored_run,subset,offset=len(full),limit=2)
    assert empty['rows'] == empty['raw'] == []


def test_time_filters_and_warmup(stored_run):
    full = results.series(stored_run)
    assert [row['warmup'] for row in full['raw'][:3]] == [True,True,False]
    start,end = full['rows'][1]['timestamp'],full['rows'][3]['timestamp']
    page = results.series(stored_run,start=start,end=end)
    assert page['rows'] == full['rows'][1:4]
    assert page['total'] == 3
    assert all(start <= row['timestamp'] <= end for row in page['raw'])
    empty = results.series(stored_run,start='2027-01-01T00:00:00Z')
    assert empty['total'] == 0
    assert empty['rows'] == empty['raw'] == []


@pytest.mark.parametrize('args', [{'sensor':3},{'subset':'invalid'}])
def test_invalid_result_selection(stored_run, args):
    with pytest.raises(ValueError, match='Unbekannte'):
        results.series(stored_run, **args)
