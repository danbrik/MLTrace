from datetime import datetime, timedelta
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from app.modeling.defaults import industrial_stae_payload
from app.modeling.stae_reconstruction import build_reconstruction_model
from app.training import data, engine
from app.schemas import PreprocessingGraph
from tests.test_training_gradient import _seed_ae, LOAD_ONLY_GRAPH
from tests.test_training_scheduler import make_db


def records_at(seconds):
    t = datetime(2026, 1, 1)
    return [data.ResolvedDatasetImage(str(i), t+timedelta(seconds=s), 'A', '/', 1, '.', str(i)) for i,s in enumerate(seconds)]


def enumerate_at(monkeypatch, seconds, **kwargs):
    records = records_at(seconds)
    rule = SimpleNamespace(start_timestamp=datetime(2026,1,1), end_timestamp=records[-1].timestamp_parsed)
    monkeypatch.setattr(data, 'enumerate_rule_images', lambda *args: records)
    return data.enumerate_rule_clip_samples(rule, clip_length=kwargs.pop('clip_length',16), future_length=0,
        sequence_contiguity_mode='timestamp_interval', **kwargs)


def test_timestamp_end_anchor_missing_and_custom_interval(monkeypatch):
    result=enumerate_at(monkeypatch, range(81))
    assert len(result.clips)==6
    assert [f.timestamp_parsed.second for f in result.clips[0].input_frames[:3]]==[0,5,10]
    assert result.clips[0].score_timestamp==datetime(2026,1,1)+timedelta(seconds=75)
    assert result.clips[0].future_frames==()
    gappy=enumerate_at(monkeypatch,[i for i in range(81) if i!=40])
    assert gappy.skipped_missing==2 and len(gappy.clips)==4
    custom=enumerate_at(monkeypatch,range(81),frame_interval_seconds=2)
    assert len(custom.clips)==51
    assert (custom.clips[0].clip_end-custom.clips[0].clip_start).total_seconds()==30


@pytest.mark.parametrize('jitter,valid', [(0.5,True),(-0.5,True),(0.500001,False),(-0.500001,False)])
def test_tolerance_is_inclusive(monkeypatch,jitter,valid):
    values=[i*5 for i in range(16)];values[5]+=jitter
    result=enumerate_at(monkeypatch,values)
    assert bool(result.clips)==valid


def test_jitter_at_end_keeps_real_frames_inside_rule(monkeypatch):
    values=[i*5 for i in range(16)]; values[-1]-=0.2
    result=enumerate_at(monkeypatch,values)
    assert len(result.clips)==1
    assert result.clips[0].input_frames[0].timestamp_parsed==datetime(2026,1,1)


def test_ties_earlier_exact_preferred_no_duplicate_frames(monkeypatch):
    result=enumerate_at(monkeypatch,[0,4.5,5.5,10],clip_length=3)
    assert result.clips[0].input_frames[1].timestamp_parsed.second==4
    exact=enumerate_at(monkeypatch,[0,4.5,5,5.5,10],clip_length=3)
    assert exact.clips[0].input_frames[1].timestamp_parsed.second==5
    assert not enumerate_at(monkeypatch,[0,1],clip_length=3,frame_interval_seconds=0.1).clips


def test_rule_boundary_and_sampling(monkeypatch):
    result=enumerate_at(monkeypatch,range(0,81,2))
    assert not result.clips  # every other timestamp required by the 5 s interval is absent
    records=records_at(range(81))
    rule=SimpleNamespace(start_timestamp=records[1].timestamp_parsed,end_timestamp=records[-1].timestamp_parsed)
    monkeypatch.setattr(data,'enumerate_rule_images',lambda *args:records[1:])
    result=data.enumerate_rule_clip_samples(rule,clip_length=16,future_length=0,sequence_contiguity_mode='timestamp_interval')
    assert len(result.clips)==5 and result.clips[0].input_frames[0]==records[1]


def test_full_resolution_forward_and_bottleneck():
    torch.set_num_threads(2)
    model=build_reconstruction_model(torch).eval()
    seen=[]
    hooks=[layer.register_forward_hook(lambda module,args,output: seen.append(tuple(output.shape)))
           for layer in model.encoder if isinstance(layer,torch.nn.MaxPool3d)]
    with torch.no_grad():
        x=torch.zeros(1,1,16,192,352)
        z=model.encode(x)
        y=model.decode(z)
    assert tuple(z.shape)==(1,64,2,24,44)
    assert tuple(y.shape)==tuple(x.shape)
    assert seen==[(1,32,8,96,176),(1,48,4,48,88),(1,64,2,24,44)]
    assert torch.isfinite(y).all() and y.min()>=0 and y.max()<=1
    for hook in hooks:hook.remove()
    convs=[m for m in model.modules() if isinstance(m,(torch.nn.Conv3d,torch.nn.ConvTranspose3d))]
    assert len(convs)==8 and all(m.bias is not None for m in convs)
    assert len([m for m in model.modules() if isinstance(m,torch.nn.BatchNorm3d)])==7
    assert all(m.eps==1e-3 and m.momentum==0.01 for m in model.modules() if isinstance(m,torch.nn.BatchNorm3d))
    assert not any(isinstance(m,torch.nn.Linear) for m in model.modules())
    with torch.no_grad():assert model(torch.zeros(1,1,16,16,16)).shape==(1,1,16,16,16)


def test_real_preset_training_seed_min_delta_and_best_weights(tmp_path,monkeypatch):
    # Actual graph/Adam/BatchNorm training, reduced spatial size for a small CPU test.
    from PIL import Image
    db=make_db()
    try:
        run,method,_=_seed_ae(db,tmp_path,0)
        preset=industrial_stae_payload()
        method.method_type=method.builder_kind='spatiotemporal_autoencoder'
        method.method_config={**preset.method_config,'input_width':16,'input_height':16}
        method.method_graph=preset.method_graph
        run.validation_mode='external';run.shuffle=True;db.commit()
        records=records_at(range(32))
        frames=[]
        for i,r in enumerate(records):
            path=tmp_path/f'{i}.tiff';Image.fromarray(np.full((16,16),i*7,dtype=np.uint8)).save(path)
            frames.append(data.ResolvedDatasetImage(str(path),r.timestamp_parsed,'A',str(tmp_path),1,'.',path.name))
        clips=[data.ResolvedClipSample(tuple(frames[i:i+16]),(),frames[i+15].timestamp_parsed,frames[i].timestamp_parsed,frames[i+15].timestamp_parsed,'A',1) for i in [0,16]]
        params={**preset.training_config,'epochs':4,'batch_size':1,'early_stopping_patience':1}
        losses=iter([1.0,1.0-5e-9])
        def loss_factory(*args):
            return lambda a,b: torch.nn.functional.mse_loss(a,b) if torch.is_grad_enabled() else torch.tensor(next(losses),dtype=torch.float64)
        monkeypatch.setattr(engine,'_loss_fn',loss_factory)
        target=tmp_path/'trained.pt'
        engine.train_spatiotemporal_gradient(db,run,method,clips[:1],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),params,target,threading.Event(),clips[1:])
        db.refresh(run)
        assert run.epochs_completed==2  # improvement below 1e-8 doesn't reset patience
        assert run.selected_epoch==2  # lowest finite loss still determines exported weights
        checkpoint=torch.load(run.checkpoint_path,weights_only=False)
        saved=torch.load(target,weights_only=True)
        assert all(torch.equal(saved[k],checkpoint['best_model_state_dict'][k]) for k in saved)
        model=build_reconstruction_model(torch,preset.method_graph)
        model.load_state_dict(saved);model.eval()
        with torch.no_grad():assert model(torch.zeros(1,1,16,16,16)).shape==(1,1,16,16,16)
        assert checkpoint['rng_state'] is not None
    finally:db.close()


def test_separate_dataset_resolution_uses_configured_seconds(tmp_path):
    from tests.test_training_validation import seed_sets
    from app.training.validation import resolve_selections
    db=make_db()
    try:
        _,method,sets=seed_sets(db,tmp_path)
        method.builder_kind='spatiotemporal_autoencoder'
        method.method_config={**industrial_stae_payload().method_config,'clip_length':2,'frame_interval_seconds':2}
        train,val=resolve_selections(db,[sets[0]],method,'external',[sets[1].id])
        assert len(train)==len(val)==2
        assert all((c.clip_end-c.clip_start).total_seconds()==2 for c in train+val)
        assert max(c.clip_end for c in train)<min(c.clip_start for c in val)
        method.method_config={**method.method_config,'frame_interval_seconds':5}
        with pytest.raises(ValueError,match='keine verwendbaren'):
            resolve_selections(db,[sets[0]],method,'external',[sets[1].id])
    finally:db.close()
