import threading
from pathlib import Path
import pytest
import torch
from app.training.epoch_selection import EpochSelection
from app.training import engine
from app.schemas import PreprocessingGraph
from tests.test_training_gradient import _seed_ae, LOAD_ONLY_GRAPH
from tests.test_training_scheduler import make_db


def test_buffers_ties_and_old_checkpoints():
    model = torch.nn.BatchNorm1d(2)
    selection = EpochSelection({'model_selection':'best_validation'}, True)
    model.running_mean.fill_(5)
    selection.observe(5, 1., model)
    model.running_mean.fill_(10)
    selection.observe(6, 1., model)
    assert selection.finish(model) == 5
    assert model.running_mean.tolist() == [5, 5]
    with pytest.raises(ValueError, match='Checkpoint'):
        selection.restore({'epoch':10,'best_val_loss':1})
    old = EpochSelection({}, True)
    old.restore({'epoch':10,'best_val_loss':1})
    assert old.finish(model) == 10 and old.best_epoch is None
    with pytest.raises(ValueError, match='Validierung'):
        EpochSelection({'model_selection':'best_validation'}, False)


@pytest.mark.parametrize('mode', ['last', 'best_validation'])
@pytest.mark.parametrize('resume', [False, True])
def test_best_fifth_epoch_early_stop_tenth_and_resume(tmp_path, monkeypatch, mode, resume):
    db = make_db()
    try:
        run, method, paths = _seed_ae(db,tmp_path,6)
        run.validation_mode='external'; db.commit()
        params={'epochs':12,'batch_size':2,'num_workers':0,'loss':'mse','model_selection':mode,
                'early_stopping_enabled':True,'early_stopping_patience':5}
        losses=iter([5.,4.,3.,2.,1.,2.,3.,4.,5.,6.])
        def loss_factory(*args):
            def loss(a,b):
                return torch.nn.functional.mse_loss(a,b) if torch.is_grad_enabled() else torch.tensor(next(losses),device=a.device)
            return loss
        monkeypatch.setattr(engine,'_loss_fn',loss_factory)
        actual_save=engine.save_training_checkpoint
        event=threading.Event()
        def interrupt_after_six(*args,**kwargs):
            result=actual_save(*args,**kwargs)
            if kwargs.get('epoch')==6:
                event.set()
            return result
        target=tmp_path/'model.pt'
        def train():
            return engine.train_gradient(db,run,method,paths[:4],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),params,target,event,paths[4:])
        if resume:
            monkeypatch.setattr(engine,'save_training_checkpoint',interrupt_after_six)
            with pytest.raises(engine.AbortedError):train()
            assert not target.exists()
            db.refresh(run)
            run.restart_mode='checkpoint';db.commit()
            event.clear()
            monkeypatch.setattr(engine,'save_training_checkpoint',actual_save)
        train()
        db.refresh(run)
        assert run.epochs_completed==10 and run.best_epoch==5
        assert run.selected_epoch==(5 if mode=='best_validation' else 10)
        checkpoint=torch.load(run.checkpoint_path,weights_only=False)
        expected=checkpoint['best_model_state_dict'] if mode=='best_validation' else checkpoint['model_state_dict']
        exported=torch.load(target,weights_only=True)
        assert all(torch.equal(exported[key],expected[key]) for key in exported)
        if mode=='best_validation':
            assert any(not torch.equal(exported[key],checkpoint['model_state_dict'][key]) for key in exported)
    finally:db.close()


@pytest.mark.parametrize('loss', [float('nan'),float('inf'),float('-inf')])
def test_nonfinite_validation_cannot_be_selected(loss):
    with pytest.raises(ValueError,match='nicht endlich'):
        EpochSelection({'model_selection':'last'},True).observe(1,loss,torch.nn.Linear(1,1))


def test_training_parameter_normalization():
    from types import SimpleNamespace
    from app.services import _merged_training_parameters
    config=SimpleNamespace(method_type='cnn_autoencoder',method_config={},training_config={})
    assert _merged_training_parameters(config,{},active_only=True,validation_mode='external').get('model_selection','last')=='last'
    assert _merged_training_parameters(config,{'model_selection':'best_validation'},active_only=True,validation_mode='none')['model_selection']=='last'
    assert _merged_training_parameters(config,{'model_selection':'best_validation'},active_only=True,validation_mode='external')['model_selection']=='best_validation'
    with pytest.raises(ValueError):
        _merged_training_parameters(config,{'model_selection':'unknown'},active_only=True,validation_mode='external')
