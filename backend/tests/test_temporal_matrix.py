from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from app import models
from app.temporal_difference import engine, matrix, service
from app.temporal_difference.schemas import MatrixConfig
from tests.test_temporal_difference import config, records, period, temporal
from tests.test_reference_image import source


def sampled(**kw):
    return config(selection_version=2,block_seconds=5,deltas_seconds=[1,2],reference=period(0,22),comparison=period(0,22),**kw)


def test_stratified_common_starts_reproducible_and_compact():
    cfg=sampled()
    rows=records(range(23))
    samples,pairs,preview=engine.select_pairs(rows,cfg)
    assert (samples,pairs,preview)==engine.select_pairs(list(reversed(rows))+rows,cfg)
    stats=preview['periods']['reference']
    assert stats['block_count']==4 and stats['selected_start_count']==4
    firsts=[p['first'] for p in pairs if p['role']=='reference' and p['delta_seconds']==1]
    assert firsts==[p['first'] for p in pairs if p['role']=='reference' and p['delta_seconds']==2]
    assert len(samples['reference'])<=12
    cfg.seed=43
    assert engine.select_pairs(rows,cfg)[0]!=samples


def test_24_hours_count_and_inclusive_last_block():
    cfg=config(selection_version=2,block_seconds=300,reference=period(0,86400),comparison=period(0,86400),deltas_seconds=[1,5,15,60,300,900,3600])
    _,pairs,preview=engine.select_pairs(records(range(86401)),cfg)
    assert preview['periods']['reference']['block_count']==276
    assert preview['periods']['reference']['selected_start_count']==276
    assert len(pairs)==276*7*2


def test_tolerance_tie_and_incomplete_candidates():
    cfg=config(selection_version=2,block_seconds=10,reference=period(0,2),comparison=period(0,2),deltas_seconds=[1,2])
    samples,pairs,preview=engine.select_pairs(records([0,.5,1.5,2]),cfg)
    assert not preview['errors']
    chosen=[p for p in pairs if p['role']=='reference']
    assert samples['reference'][chosen[0]['second']]['timestamp'].endswith('00:00:00.500000')
    _,pairs,preview=engine.select_pairs(records([0,.499999,2]),cfg)
    assert not pairs and preview['errors']
    assert preview['periods']['reference']['deltas'][0]['missing_targets']==1


def test_rest_blocks_short_period_and_single_candidate():
    cfg=config(selection_version=2,block_seconds=5,reference=period(0,14),comparison=period(0,14),deltas_seconds=[2])
    _,_,preview=engine.select_pairs(records(range(15)),cfg)
    assert preview['periods']['reference']['block_count']==3
    cfg.reference.end=cfg.reference.start+timedelta(seconds=2)
    _,_,preview=engine.select_pairs(records(range(15)),cfg)
    assert preview['periods']['reference']['selected_start_count']==1
    cfg.reference.end=cfg.reference.start+timedelta(seconds=1)
    assert engine.select_pairs(records(range(15)),cfg)[2]['errors']


def test_top_percent_ties_zero_and_no_filter():
    diff=np.array([[0.,1.],[2.,100.]])
    assert matrix.filtered_difference(diff,None) is diff
    shown=matrix.filtered_difference(diff,25)
    assert shown.count()==1 and shown.compressed()[0]==100
    assert matrix.filtered_difference(np.ones((4,4)),1).count()==16
    assert matrix.filtered_difference(np.zeros((4,4)),100).count()==0


@pytest.mark.parametrize('version',[1,2])
def test_matrix_frozen_pairs_metadata_reopen_failure_and_cleanup(temporal,version,monkeypatch):
    db,cfg,pipeline=temporal
    cfg.deltas_seconds=[2,4];cfg.selection_version=version;cfg.block_seconds=2
    run=service.enqueue(db,cfg,wake_scheduler=False)
    service.run_scheduled(run.id);db.expire_all()
    assert service.get_run(db,run.id).status=='finished'
    choices=matrix.candidates(db,run.id,'comparison',[2,4])
    assert choices['suggested']
    request=MatrixConfig(role='comparison',deltas_seconds=[2,4],start_times=choices['suggested'],top_percent=1)
    old_summary=service.summaries(db,run.id)
    # A later pipeline edit must not affect the frozen renderer.
    pipeline.graph={'nodes':[],'edges':[]};db.commit()
    result=matrix.generate(db,run.id,request)
    path=matrix.png_path(db,run.id,'comparison',result['artifact'])
    with Image.open(path) as image:
        metadata=json.loads(image.info['Description'])
        assert metadata['config']['start_times']==choices['suggested']
        assert metadata['difference_limit']==400
        assert metadata['timestamps'][0]['partners'][1]['delta_seconds']==4
    assert matrix.state(db,run.id,'comparison')==result
    assert service.summaries(db,run.id)==old_summary
    assert matrix.png_path(db,run.id,'comparison','../../outside.png') is None
    assert matrix.state(db,run.id,'reference')['artifact'] is None
    manifest=json.loads((service.artifact_dir(run.id)/'manifest.json').read_text())
    first=next(s for s in manifest['samples']['comparison'] if s['timestamp']==request.start_times[0])
    Path(first['file_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='verändert'):
        matrix.generate(db,run.id,request)
    assert matrix.state(db,run.id,'comparison')==result and path.exists()
    assert service.delete_run(db,run.id)
    assert not path.exists()


def test_render_one_and_three_rows_shared_scale(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    seen=[]; original=Figure.savefig
    def capture(fig,*args,**kwargs):
        seen.append(fig);return original(fig,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',capture)
    for count in (1,3):
        times=[f'2026-01-01T00:00:0{i}' for i in range(count)]
        for row in range(count):
            np.save(tmp_path/f'{row}-base.npy',np.arange(12).reshape(3,4))
            for col in range(3):np.save(tmp_path/f'{row}-{col}.npy',np.full((3,4),row+col+1.))
        matrix._render(tmp_path,MatrixConfig(role='reference',deltas_seconds=[1,2,3],start_times=times),0,11,5,{})
        fig=seen[-1]
        assert len(fig.axes)==count*4+1
        assert all(ax.images[0].get_clim()==(0.,5.) for i,ax in enumerate(fig.axes[:-1]) if i%4)
        assert all(ax.images[0].origin=='upper' and ax.images[0].get_interpolation()=='nearest' for ax in fig.axes[:-1])


def test_matrix_api_validation_download_and_project_isolation(temporal):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import get_db
    from tests.test_training_scheduler import make_db
    db,cfg,_=temporal
    cfg.deltas_seconds=[2]
    run=service.enqueue(db,cfg,wake_scheduler=False);service.run_scheduled(run.id);db.expire_all()
    app.dependency_overrides[get_db]=lambda:db
    try:
        client=TestClient(app);base=f'/api/temporal-difference/runs/{run.id}/matrix'
        assert client.get(base,params={'role':'reference'}).json()['artifact'] is None
        starts=client.get(base+'/start-points',params={'role':'reference','deltas':2}).json()['suggested']
        request=dict(role='reference',deltas_seconds=[2],start_times=starts,top_percent=1)
        response=client.post(base,json=request);assert response.status_code==200,response.text
        artifact=response.json()['artifact']
        image=client.get(base+'/png',params={'role':'reference','artifact':artifact,'download':'true'})
        assert image.status_code==200 and image.headers['content-type']=='image/png'
        assert 'attachment' in image.headers['content-disposition']
        assert client.post(base,json={**request,'top_percent':0}).status_code==422
        assert client.post(base,json={**request,'deltas_seconds':[2,4,6,8]}).status_code==422
        assert client.post(base,json={**request,'start_times':['not a saved time']}).status_code==400
        other=make_db()
        try:
            app.dependency_overrides[get_db]=lambda:other
            assert client.get(base,params={'role':'reference'}).status_code==404
            assert client.get(base+'/png',params={'role':'reference','artifact':artifact}).status_code==404
        finally:other.close()
    finally:app.dependency_overrides.pop(get_db,None)
