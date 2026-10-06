from datetime import datetime, timedelta
from types import SimpleNamespace
import threading

import pytest
from app import models, services
from app.modeling.registry import registry
from app.modeling.training_ui import active_parameters, training_schema
from app.schemas import TrainingPipelineCreate, TrainingPipelineDryRunRequest, PreprocessingGraph
from app.training.validation import resolve_selections, check_sample_shapes
from app.training.engine import train_gradient
from tests.test_training_gradient import _seed_ae, LOAD_ONLY_GRAPH
from tests.test_testing_service import make_db, write_tiff


def test_active_fields_ignore_hidden_invalid_values():
    definition = registry.get('cnn_autoencoder')
    params = active_parameters(definition, {'loss': 'mse', 'ssim_window_size': 'bad', 'ssim_weight': -10,
        'num_workers': 0, 'prefetch_factor': 'bad', 'early_stopping_enabled': False, 'early_stopping_patience': 'bad'}, {})
    assert not any(key.startswith('ssim_') for key in params)
    assert 'prefetch_factor' not in params
    assert 'early_stopping_patience' not in params
    assert params['optimizer'] == 'adam'
    with pytest.raises(ValueError):
        active_parameters(definition, {'learning_rate': float('nan')}, {})
    for loss in ['ssim', 'mse_ssim', 'mae_ssim']:
        result = active_parameters(definition, {'loss': loss}, {})
        assert 'ssim_window_size' in result
        assert ('ssim_weight' in result) == (loss != 'ssim')


def test_prediction_and_model_capabilities():
    definition = registry.get('spatiotemporal_autoencoder')
    values = active_parameters(definition, {'prediction_loss': 'bad', 'prediction_min_weight': 'bad'}, {'prediction_branch': False})
    assert values['training_objective'] == 'reconstruction'
    assert 'prediction_loss' not in values
    for name in ['mean_image', 'fast_anogan']:
        assert not training_schema(registry.get(name))['supports_validation']


def seed_sets(db, tmp_path):
    run, method, paths = _seed_ae(db, tmp_path, image_count=0)
    root = tmp_path / 'selected'
    begin = datetime(2026, 1, 1)
    for i in range(8):
        write_tiff(root / f'frame_{begin + timedelta(seconds=i):%Y%m%d_%H%M%S}.tif', i * 20, size=(160, 120))
    dataset = models.Dataset(name='Sources', root_path=str(root), status='ready', timestamp_regex=r'(?P<timestamp>\d{8}_\d{6})', timestamp_format='%Y%m%d_%H%M%S')
    db.add(dataset); db.flush()
    folder = models.DatasetFolder(dataset_id=dataset.id, relative_path='.', image_count=8,
        first_timestamp=begin, last_timestamp=begin+timedelta(seconds=7), extension_summary={'.tif':8},
        resolution_summary={'160x120':8}, image_metadata={'channels':1}, cadence_summary={'median_seconds':1})
    db.add(folder); db.flush()
    sets=[]
    for name, start, end in [('Train',0,3), ('Validation',4,7), ('Overlap',3,6)]:
        selected = models.TrainingDataset(name=name, usage_label='train')
        db.add(selected); db.flush()
        db.add(models.TrainingDatasetRule(training_dataset_id=selected.id, folder_id=folder.id,
            start_timestamp=begin+timedelta(seconds=start), end_timestamp=begin+timedelta(seconds=end), stride=1))
        sets.append(selected)
    db.commit()
    return run, method, sets


def test_separate_selection_crud_dry_run_and_dependencies(tmp_path):
    db=make_db()
    try:
        run, method, sets=seed_sets(db,tmp_path)
        train, val = resolve_selections(db,[sets[0],sets[0]],method,'external',[sets[1].id])
        assert len(train)==len(val)==4
        with pytest.raises(ValueError, match='überschneiden'):
            resolve_selections(db,[sets[0]],method,'external',[sets[2].id])
        source=db.get(models.TrainingPipeline, run.training_pipeline_id)
        payload=TrainingPipelineCreate(name='Separate', training_dataset_ids=[sets[0].id],
            preprocessing_pipeline_id=source.preprocessing_pipeline_id, method_configuration_id=method.id,
            validation_mode='external', validation_dataset_ids=[sets[1].id], validation_shuffle=True,
            training_parameters={'loss':'mse','epochs':1,'batch_size':2,'num_workers':0})
        saved=services.create_training_pipeline(db,payload)
        assert saved.validation_datasets[0].training_dataset_id == sets[1].id
        assert saved.validation_shuffle
        assert services.find_training_pipeline_by_signature(db,payload).id==saved.id
        result=services.dry_run_training_pipeline(db,TrainingPipelineDryRunRequest(**payload.model_dump()))
        assert result.valid, result.errors
        assert result.training_sample_count==result.validation_sample_count==4
        from app.training.service import _snapshot
        snapshot=_snapshot(db, db.get(models.TrainingPipeline,saved.id))
        assert snapshot['validation_dataset_ids']==[sets[1].id]
        assert services.training_dataset_update_lock_reasons(db,sets[1].id)
        with pytest.raises(ValueError):
            services.delete_training_dataset(db, sets[1].id)
    finally:
        db.close()


@pytest.mark.parametrize('shuffle', [False, True])
def test_real_training_uses_full_validation_and_sampler(tmp_path, monkeypatch, shuffle):
    import torch
    db=make_db()
    try:
        run,method,paths=_seed_ae(db,tmp_path,7)
        run.validation_mode='external'; run.validation_shuffle=shuffle; run.shuffle=not shuffle
        run.validation_dataset_ids=[]
        db.commit()
        captured=[]
        loader=torch.utils.data.DataLoader
        def capture(*args,**kwargs):
            captured.append((len(args[0]),kwargs['shuffle']))
            return loader(*args,**kwargs)
        monkeypatch.setattr(torch.utils.data,'DataLoader',capture)
        count=train_gradient(db,run,method,paths[:4],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),
            {'epochs':1,'batch_size':2,'num_workers':0,'loss':'mse'},tmp_path/'model.pt',threading.Event(),paths[4:])
        assert count==4 and run.validation_sample_count==3
        assert captured==[(4,not shuffle),(3,shuffle)]
        assert run.val_loss is not None
        # Re-evaluate every validation image individually against the exported weights.
        from app.training.engine import _build_model
        model, _ = _build_model(torch, method)
        model.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True)); model.eval()
        from app.training.engine import _PreprocessedImageDataset
        dataset=_PreprocessedImageDataset(paths[4:],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH))
        with torch.no_grad():
            losses=[float(torch.nn.functional.mse_loss(model(dataset[i].unsqueeze(0))[0],dataset[i].unsqueeze(0))) for i in range(3)]
        assert run.val_loss==pytest.approx(sum(losses)/3,rel=1e-5)
    finally: db.close()


def test_empty_unreadable_validation_fails(tmp_path):
    db=make_db()
    try:
        run,method,paths=_seed_ae(db,tmp_path,2)
        run.validation_mode='external'; db.commit()
        with pytest.raises(ValueError,match='keine lesbaren'):
            train_gradient(db,run,method,paths,PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),
                {'epochs':1,'batch_size':2,'num_workers':0,'loss':'mse'},tmp_path/'model.pt',threading.Event(),[str(tmp_path/'missing.tif')])
    finally: db.close()


def test_clip_overlap_includes_future_targets(tmp_path, monkeypatch):
    from app.training import validation
    from app.training.data import ResolvedClipSample, ResolvedDatasetImage
    t=datetime(2026,1,1)
    def frame(i):
        return ResolvedDatasetImage(str(tmp_path/f'{i}.tif'),t+timedelta(seconds=i),'A',str(tmp_path),1,'.',f'{i}.tif')
    def clip(inputs,future):
        return ResolvedClipSample(tuple(map(frame,inputs)),tuple(map(frame,future)),t,t,t,'A',1)
    normal=clip([0,1],[2]); external=clip([2,3],[4])
    configuration=SimpleNamespace(builder_kind='spatiotemporal_autoencoder',method_type='spatiotemporal_autoencoder')
    monkeypatch.setattr(validation,'datasets_by_ids',lambda db,ids:['validation'])
    monkeypatch.setattr(validation,'resolve_group',lambda sets,config:[normal] if sets==['train'] else [external])
    with pytest.raises(ValueError,match='überschneiden'):
        validation.resolve_selections(None,['train'],configuration,'external',[1])


def test_dry_run_detects_later_size_mismatch(tmp_path):
    a=tmp_path/'a.tif'; b=tmp_path/'b.tif'
    write_tiff(a,20,size=(8,8)); write_tiff(b,20,size=(9,8))
    with pytest.raises(ValueError,match='unterschiedliche'):
        check_sample_shapes([str(a)],[str(b)],SimpleNamespace(builder_kind='form',method_config={}),PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH))


def test_real_vae_external_validation(tmp_path):
    db=make_db()
    try:
        run,method,paths=_seed_ae(db,tmp_path,5)
        method.builder_kind='sequential_variational_autoencoder'; method.method_type='cnn_vae'
        method.method_config={**method.method_config,'kl_weight':.01}
        run.validation_mode='external'; db.commit()
        train_gradient(db,run,method,paths[:3],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),
            {'epochs':1,'batch_size':2,'num_workers':0,'reconstruction_loss':'l1','model_selection':'best_validation'},tmp_path/'vae.pt',threading.Event(),paths[3:])
        assert run.validation_sample_count==2 and run.val_loss is not None
        assert run.best_epoch == run.selected_epoch == 1
    finally: db.close()


def test_real_sequence_training_external_validation(tmp_path):
    from app.training.engine import train_spatiotemporal_gradient
    from app.training.data import ResolvedClipSample, ResolvedDatasetImage
    db=make_db()
    try:
        run,method,paths=_seed_ae(db,tmp_path,8)
        method.builder_kind='spatiotemporal_autoencoder'; method.method_type='spatiotemporal_autoencoder'
        method.method_config={**method.method_config,'clip_length':2,'future_length':0,'prediction_branch':False}
        method.method_graph={'builder_kind':'spatiotemporal_autoencoder',
            'encoder':[{'id':'enc','type':'Conv3d','config':{'out_channels':2,'kernel_size':1,'stride':1,'padding':0}}],
            'decoder':[{'id':'dec','type':'Conv3d','config':{'out_channels':1,'kernel_size':1,'stride':1,'padding':0}}], 'prediction_decoder':[]}
        run.validation_mode='external'; run.validation_shuffle=True; db.commit()
        t=datetime(2026,1,1)
        frames=[ResolvedDatasetImage(path,t+timedelta(seconds=i),'A',str(tmp_path),1,'.',str(i)) for i,path in enumerate(paths)]
        clips=[ResolvedClipSample(tuple(frames[i:i+2]),(),t,t,t,'A',1) for i in range(0,8,2)]
        train_spatiotemporal_gradient(db,run,method,clips[:2],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),
            {'epochs':1,'batch_size':2,'num_workers':0,'reconstruction_loss':'mse','training_objective':'reconstruction','model_selection':'best_validation'},tmp_path/'stae.pt',threading.Event(),clips[2:])
        assert run.validation_sample_count==2 and run.val_loss is not None
        assert run.best_epoch == run.selected_epoch == 1
    finally: db.close()


def test_migration_preserves_legacy_and_separates_projects(tmp_path):
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import create_engine, text, inspect
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for name in ['first','second']:
        url=f'sqlite:///{tmp_path/name}.db'
        config=Config(str(root.parent/'alembic.ini')); config.set_main_option('script_location',str(root/'alembic')); config.set_main_option('sqlalchemy.url',url)
        command.upgrade(config,'0065_temporal_difference')
        engine=create_engine(url)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO training_pipelines (name,shuffle,training_parameters,preprocessing_pipeline_id,method_configuration_id) VALUES ('old',1,'{}',1,1)"))
        command.upgrade(config,'head')
        with engine.connect() as conn:
            row=conn.execute(text('SELECT validation_mode,validation_shuffle FROM training_pipelines')).one()
            assert row==('legacy_fraction',0)
            assert 'training_pipeline_validation_datasets' in inspect(conn).get_table_names()
            columns = {c['name']: c for c in inspect(conn).get_columns('training_runs')}
            assert columns['best_epoch']['nullable'] and columns['selected_epoch']['nullable']
        engine.dispose()


@pytest.mark.parametrize('mode',['none','external'])
def test_early_stopping_and_new_checkpoint_resume(tmp_path,mode):
    from app.training.service import _validate_training_checkpoint, RunConflict
    from app.training import engine as training_engine
    db=make_db()
    try:
        run,method,paths=_seed_ae(db,tmp_path,6)
        run.validation_mode=mode; run.shuffle=False; db.commit()
        params={'epochs':4,'batch_size':4,'num_workers':0,'loss':'mse','learning_rate':0,
                'early_stopping_enabled':True,'early_stopping_patience':1}
        train_gradient(db,run,method,paths[:4],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),params,
                       tmp_path/'model.pt',threading.Event(),paths[4:] if mode=='external' else [])
        assert run.epochs_completed==2
        assert (run.val_loss is not None)==(mode=='external')
        import torch
        checkpoint=torch.load(run.checkpoint_path,weights_only=False)
        assert checkpoint['validation']['validation_mode']==mode
        assert checkpoint['best_stop_metric']==pytest.approx(run.val_loss if mode=='external' else run.train_loss)
        # New checkpoint sources include the validation set, and cannot be changed on resume.
        run.restart_mode='checkpoint'; db.commit()
        train_gradient(db,run,method,paths[:4],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),params,
                       tmp_path/'resumed.pt',threading.Event(),paths[4:] if mode=='external' else [])
        run.validation_shuffle=True
        with pytest.raises(Exception,match='signature|match|changed|incompatible'):
            train_gradient(db,run,method,paths[:4],PreprocessingGraph.model_validate(LOAD_ONLY_GRAPH),params,
                           tmp_path/'invalid.pt',threading.Event(),paths[4:] if mode=='external' else [])
    finally: db.close()
