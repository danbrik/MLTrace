from types import SimpleNamespace

import numpy as np
import pytest

from app.modeling.architectures.statistical_reference import StatisticalReferenceArchitecture
from app.modeling.statistical_reference import reference_statistics, anomaly_map
from app.testing.service import ArtifactEvaluator, _effective_inference_config


def test_population_statistics_uint16_and_fractional_mean():
    images = [np.array([[65530, 0], [10, 4]], dtype=np.uint16), np.array([[65531, 2], [14, 4]], dtype=np.uint16)]
    mean, std, count = reference_statistics(iter(images))
    assert count == 2
    assert mean.dtype == std.dtype == np.float64
    np.testing.assert_array_equal(mean, [[65530.5, 1], [12, 4]])
    np.testing.assert_array_equal(std, [[.5, 1], [2, 0]])
    np.testing.assert_allclose(anomaly_map(mean + 2, mean, std, .5), [[2, 4/3], [.8, 4]])


def test_single_image_stable_and_zero_reference():
    mean, std, _ = reference_statistics([np.full((2, 3), 50000, dtype=np.uint16)])
    np.testing.assert_array_equal(std, np.zeros((2, 3)))
    assert anomaly_map(mean, mean, std, 1e-6).sum() == 0
    assert anomaly_map(mean + 1, mean, std, 1e-6)[0, 0] == 1e6


@pytest.mark.parametrize('images', [[], [np.zeros((2, 2)), np.zeros((2, 3))], [np.array([[np.nan]])], [np.zeros((2, 2, 3))]])
def test_invalid_reference_images(images):
    with pytest.raises(ValueError):
        reference_statistics(images)


@pytest.mark.parametrize('epsilon', [0, -1, float('nan'), float('inf')])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        StatisticalReferenceArchitecture().validate_config({}, {'epsilon': epsilon})


def test_artifact_evaluator_scores_and_heatmaps_share_normalization(tmp_path):
    path = tmp_path / 'reference.npz'
    np.savez(path, mean=np.array([[10., 10.]]), std=np.array([[0., 3.]]), epsilon=1., count=2)
    config = SimpleNamespace(inference_config={'frame_score_aggregation': 'mean'}, builder_kind='form')
    run = SimpleNamespace(training_pipeline=SimpleNamespace(method_configuration=config), artifact_path=str(path), artifact_kind='statistical_reference')
    evaluator = ArtifactEvaluator(run, {'error_metric': 'mse'})
    image = np.array([[14, 14]], dtype=np.uint16)
    assert evaluator.score(image, None)[0] == 2.5
    assert evaluator.score_batch([image, image], None)[1][0] == 2.5
    np.testing.assert_array_equal(evaluator.pixel_error_map(image, evaluator.reconstruct(image)), [[4, 1]])
    evaluator.inference_config['frame_score_aggregation'] = 'max'
    assert evaluator.score(image, None)[0] == 4
    effective = _effective_inference_config(run, {'error_metric': 'ssim_distance', 'frame_score_aggregation': 'p95'})
    assert effective == {'error_metric': 'normalized_deviation', 'frame_score_aggregation': 'p95'}


def test_method_api_supports_new_baseline():
    from test_modeling import make_client
    clients = make_client()
    client = next(clients)
    try:
        definitions = client.get('/api/methods/definitions').json()
        assert any(d['type'] == 'statistical_reference' for d in definitions)
        response = client.post('/api/methods/configurations', json={
            'name': 'Statistical reference', 'method_type': 'statistical_reference',
            'method_graph': {}, 'method_config': {'epsilon': .001}, 'training_config': {}, 'inference_config': {},
        })
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['artifact_kind'] == 'statistical_reference'
        assert result['training_mode'] == 'fit'
        assert result['inference_config']['error_metric'] == 'normalized_deviation'
        assert result['diagram']['nodes'][1]['label'] == 'Pixel-wise mean and standard deviation'
    finally:
        clients.close()


def test_numerical_stability_large_offset():
    values = [np.full((2, 2), 1e12 + delta) for delta in (0, 1, 2, 3)]
    mean, std, _ = reference_statistics(values)
    np.testing.assert_array_equal(mean, np.full((2, 2), 1e12 + 1.5))
    np.testing.assert_allclose(std, np.sqrt(1.25))


@pytest.mark.parametrize('aborted', [False, True])
def test_training_worker_fits_and_reopens_artifact(tmp_path, monkeypatch, aborted):
    import threading
    from sqlalchemy.orm import sessionmaker
    from app import models
    from app.training import engine
    from test_testing_service import make_db
    from test_training_gradient import _seed_ae
    db = make_db()
    try:
        run, method, paths = _seed_ae(db, tmp_path, image_count=3)
        method.method_type = 'statistical_reference'
        method.builder_kind = 'form'
        method.method_config = {'epsilon': 1.0}
        method.inference_config = {'frame_score_aggregation': 'mean'}
        run.validation_mode = 'legacy_fraction'
        db.commit()
        run_id = run.id
        monkeypatch.setattr(engine, 'SessionLocal', sessionmaker(bind=db.get_bind()))
        monkeypatch.setattr(engine, 'enumerate_or_fail', lambda *_: paths)
        monkeypatch.setattr(engine, '_run_artifact_dir', lambda *_: tmp_path)
        monkeypatch.setattr(engine, '_clear_successful_checkpoint', lambda *_: None)
        event = threading.Event()
        if aborted:
            event.set()
        engine.run_training(run_id, event)
        db.expire_all()
        saved = db.get(models.TrainingRun, run_id)
        assert saved.status == ('aborted' if aborted else 'finished'), saved.error_message
        if not aborted:
            assert saved.artifact_kind == 'statistical_reference'
            assert saved.image_count == 3
            evaluator = ArtifactEvaluator(saved)
            np.testing.assert_allclose(evaluator.mean_image, 90)
            np.testing.assert_allclose(evaluator.statistical_reference[1], np.sqrt(200/3))
            assert evaluator.score(np.full((120, 160), 100, dtype=np.uint16), None)[0] == pytest.approx(10 / (np.sqrt(200/3) + 1))
    finally:
        db.close()
