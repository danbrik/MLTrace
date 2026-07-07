from app.optimization.service import _pr_auc, _random_sample, _roc_auc


def test_optimization_auc_metrics_rank_anomalies_higher() -> None:
    normal = [0.1, 0.2, 0.3]
    anomaly = [0.8, 0.9]

    assert _roc_auc(normal, anomaly) == 1.0
    assert _pr_auc(normal, anomaly) == 1.0


def test_optimization_random_sampler_is_deterministic() -> None:
    distributions = {
        "method_configuration_id": {"kind": "categorical", "choices": [1, 2]},
        "method_config.latent_dim": {"kind": "categorical", "choices": [64, 128]},
        "training_parameters.learning_rate": {"kind": "float", "low": 0.00001, "high": 0.001, "log": True},
        "training_parameters.batch_size": {"kind": "int", "low": 8, "high": 32, "step": 8},
    }

    first = _random_sample(distributions, seed=42)
    second = _random_sample(distributions, seed=42)

    assert first == second
    assert first["method_config.latent_dim"] in {64, 128}
    assert 0.00001 <= first["training_parameters.learning_rate"] <= 0.001
    assert first["training_parameters.batch_size"] in {8, 16, 24, 32}
