from fastapi.testclient import TestClient
from types import SimpleNamespace
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.database import Base, get_db
from app.main import app
from app.modeling.architectures.cnn_autoencoder import CnnAutoencoderArchitecture
from app.modeling.defaults import _old_cnn_ae_payload, _old_industrial_stae_payload, _old_paper_payloads_by_name, default_method_payloads, ensure_default_method_configurations
from app.modeling.fast_anogan import build_fast_anogan_modules, fast_anogan_forward
from app.modeling.registry import MethodRegistry, registry
from app.services import create_method_configuration, validate_method_configuration
from app.training.engine import _build_model, _prediction_horizon_weights, _prediction_weight_for_epoch


def make_client():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db: Session = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def autoencoder_payload(name: str = "CNN-AE small 160x120") -> dict:
    return {
        "name": name,
        "description": "Small test method",
        "method_type": "cnn_autoencoder",
        "method_config": {
            "input_channels": 1,
            "input_width": 160,
            "input_height": 120,
            "latent_dim": 64,
            "output_activation": "sigmoid",
        },
        "training_config": {"epochs": 5, "batch_size": 2, "learning_rate": 0.001, "loss": "mse"},
        "inference_config": {"error_metric": "mse"},
        "method_graph": {
            "builder_kind": "sequential_autoencoder",
            "encoder": [
                {
                    "id": "enc-conv",
                    "type": "Conv2d",
                    "config": {"out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1},
                },
                {"id": "enc-act", "type": "ReLU", "config": {}},
                {"id": "enc-flat", "type": "Flatten", "config": {}},
                {"id": "enc-linear", "type": "Linear", "config": {"out_features": 64}},
            ],
            "latent": {"latent_dim": 64},
            "decoder": [
                {"id": "dec-linear", "type": "Linear", "config": {"out_features": 38400}},
                {"id": "dec-unflat", "type": "Unflatten", "config": {"channels": 8, "height": 60, "width": 80}},
                {
                    "id": "dec-conv",
                    "type": "ConvTranspose2d",
                    "config": {"out_channels": 1, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
                },
            ],
        },
    }


def mean_image_payload(name: str = "Mean image baseline") -> dict:
    return {
        "name": name,
        "description": "Baseline",
        "method_type": "mean_image",
        "method_graph": {},
        "method_config": {
            "aggregation": "mean",
            "accumulator_dtype": "float32",
            "output_dtype_policy": "source",
            "normalization_mode": "none",
        },
        "training_config": {},
        "inference_config": {"error_metric": "mse"},
    }


def test_method_registry_discovers_core_methods() -> None:
    method_types = {definition.type for definition in registry.list_definitions()}

    assert {"ae_dense", "ae_spatial", "cnn_autoencoder", "cnn_vae", "mean_image"}.issubset(method_types)


def test_duplicate_method_registration_rejected() -> None:
    local_registry = MethodRegistry()
    local_registry.register(CnnAutoencoderArchitecture())

    with pytest.raises(ValueError, match="already registered"):
        local_registry.register(CnnAutoencoderArchitecture())


def test_method_layer_catalog_api_returns_v1_layers_and_model_alias_works() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        response = client.get("/api/methods/layers")
        assert response.status_code == 200
        layer_types = {layer["type"] for layer in response.json()}
        assert {
            "Conv2d",
            "ConvTranspose2d",
            "BatchNorm2d",
            "MaxPool2d",
            "Upsample",
            "Dropout2d",
            "Flatten",
            "Unflatten",
            "Linear",
            "ReLU",
            "LeakyReLU",
            "GELU",
            "Sigmoid",
            "Tanh",
        }.issubset(layer_types)

        alias_response = client.get("/api/models/layers")
        assert alias_response.status_code == 200
        assert {layer["type"] for layer in alias_response.json()} == layer_types
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_method_configuration_crud_and_parameter_index() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        definitions = client.get("/api/methods/definitions")
        assert definitions.status_code == 200
        definitions_by_type = {definition["type"]: definition for definition in definitions.json()}
        assert "mean_image" in definitions_by_type
        assert definitions_by_type["cnn_autoencoder"]["training_mode"] == "gradient"
        assert definitions_by_type["mean_image"]["training_mode"] == "fit"

        created = client.post("/api/methods/configurations", json=autoencoder_payload())
        assert created.status_code == 200
        method = created.json()
        assert method["method_type"] == "cnn_autoencoder"
        assert method["architecture_type"] == "cnn_autoencoder"
        assert method["method_family"] == "neural_reconstruction"
        assert method["training_mode"] == "gradient"
        assert method["artifact_kind"] == "weights"
        assert method["requires_training"] is True
        assert method["supports_training_pipeline"] is True
        assert method["builder_kind"] == "sequential_autoencoder"
        assert method["diagram"]["nodes"][0]["label"] == "Input"
        assert method["validation"]["valid"] is True
        assert method["validation"]["layer_specs"]
        assert any(parameter["path"] == "method_config.latent_dim" for parameter in method["parameters"])

        listed = client.get("/api/methods/configurations")
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        assert client.get("/api/models/configurations").status_code == 200

        duplicate = client.post("/api/methods/configurations", json=autoencoder_payload("cnn-ae SMALL 160x120"))
        assert duplicate.status_code == 400

        update_payload = autoencoder_payload("CNN-AE updated")
        update_payload["method_config"]["latent_dim"] = 32
        update_payload["method_graph"]["latent"]["latent_dim"] = 32
        update_payload["method_graph"]["encoder"][-1]["config"]["out_features"] = 32
        updated = client.put(f"/api/methods/configurations/{method['id']}", json=update_payload)
        assert updated.status_code == 200
        updated_body = updated.json()
        assert updated_body["name"] == "CNN-AE updated"
        assert updated_body["method_config"]["latent_dim"] == 32
        assert updated_body["model_config"]["latent_dim"] == 32
        latent_params = [p for p in updated_body["parameters"] if p["path"] == "method_config.latent_dim"]
        assert len(latent_params) == 1
        assert latent_params[0]["value_number"] == 32

        loaded = client.get(f"/api/methods/configurations/{method['id']}")
        assert loaded.status_code == 200
        assert loaded.json()["name"] == "CNN-AE updated"

        deleted = client.delete(f"/api/methods/configurations/{method['id']}")
        assert deleted.status_code == 204
        assert client.get("/api/methods/configurations").json() == []
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_mean_image_saved_with_training_pipeline_filter_flags() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        created = client.post("/api/methods/configurations", json=mean_image_payload())
        assert created.status_code == 200
        body = created.json()
        assert body["method_type"] == "mean_image"
        assert body["architecture_type"] == "mean_image"
        assert body["method_family"] == "statistical_baseline"
        assert body["training_mode"] == "fit"
        assert body["artifact_kind"] == "mean_image"
        assert body["requires_training"] is True
        assert body["supports_training_pipeline"] is True
        assert body["builder_kind"] == "form"
        assert body["diagram"]["nodes"][1]["label"] == "Aggregate mean image"
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_method_configuration_validation_rejects_invalid_payloads() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        invalid_method = autoencoder_payload()
        invalid_method["method_type"] = "unknown"
        response = client.post("/api/methods/configurations", json=invalid_method)
        assert response.status_code == 400

        invalid_layer = autoencoder_payload("Invalid layer")
        invalid_layer["method_graph"]["encoder"][0]["type"] = "ArbitraryPythonImport"
        response = client.post("/api/methods/configurations", json=invalid_layer)
        assert response.status_code == 400
        assert "Unknown layer type" in response.json()["detail"]

        missing_encoder = autoencoder_payload("Missing encoder")
        missing_encoder["method_graph"]["encoder"] = []
        response = client.post("/api/methods/configurations", json=missing_encoder)
        assert response.status_code == 400
        assert "encoder layer" in response.json()["detail"]

        invalid_enum = autoencoder_payload("Invalid enum")
        invalid_enum["method_config"]["output_activation"] = "softmax"
        response = client.post("/api/methods/configurations", json=invalid_enum)
        assert response.status_code == 400
        assert "output_activation" in response.json()["detail"]

        mean_with_layers = mean_image_payload("Mean with layers")
        mean_with_layers["method_graph"] = {"encoder": [{"id": "x", "type": "Conv2d", "config": {}}]}
        response = client.post("/api/methods/configurations", json=mean_with_layers)
        assert response.status_code == 400
        assert "cannot contain a layer graph" in response.json()["detail"]
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_method_configuration_diagram_endpoint_returns_static_shape_flow_without_torch() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        response = client.post("/api/methods/configurations/diagram", json=autoencoder_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert body["errors"] == []
        assert body["layer_specs"][0]["input_label"] == "N,1,120,160"
        assert body["layer_specs"][-1]["output_label"] == "N,1,120,160"
        assert body["torch_check"] is None
        assert [node["section"] for node in body["diagram"]["nodes"]][:2] == ["input", "encoder"]
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_method_torch_check_endpoint_runs_only_when_requested() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        response = client.post("/api/methods/configurations/torch-check", json=autoencoder_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"available", "missing"}
        assert body["logs"][0] == "Building encoder"
        if body["status"] == "available":
            assert body["valid"] is True
            assert body["logs"][-1] == "Passed"
            assert body["torch_check"]["output_shape"] == [1, 1, 120, 160]
        else:
            assert body["valid"] is False
            assert "Torch is not installed" in " ".join(body["warnings"])
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_cnn_static_validation_rejects_bad_shapes() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        latent_mismatch = autoencoder_payload("Latent mismatch")
        latent_mismatch["method_config"]["latent_dim"] = 32
        latent_mismatch["method_graph"]["latent"]["latent_dim"] = 32
        response = client.post("/api/methods/configurations", json=latent_mismatch)
        assert response.status_code == 400
        assert "latent_dim must match the final encoder output feature count" in response.json()["detail"]

        output_mismatch = autoencoder_payload("Output mismatch")
        output_mismatch["method_graph"]["decoder"][-1]["config"]["out_channels"] = 2
        response = client.post("/api/methods/configurations", json=output_mismatch)
        assert response.status_code == 400
        assert "Decoder output must match input shape" in response.json()["detail"]

        bad_batch_norm = autoencoder_payload("Bad BatchNorm")
        bad_batch_norm["method_graph"]["encoder"].insert(
            1,
            {"id": "bad-bn", "type": "BatchNorm2d", "config": {"num_features": 99}},
        )
        response = client.post("/api/methods/configurations/validate", json=bad_batch_norm)
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is False
        assert "num_features must match input channels" in " ".join(body["errors"])

        bad_pool = autoencoder_payload("Bad Pool")
        bad_pool["method_graph"]["encoder"].insert(
            1,
            {"id": "bad-pool", "type": "MaxPool2d", "config": {"kernel_size": 999}},
        )
        response = client.post("/api/methods/configurations/validate", json=bad_pool)
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is False
        assert "output spatial dimensions must be positive" in " ".join(body["errors"])
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass


def test_default_method_bootstrap_is_idempotent() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        assert ensure_default_method_configurations(db) == len(default_method_payloads())
        assert ensure_default_method_configurations(db) == 0
        rows = db.scalars(select(models.MethodConfiguration.name)).all()
        names = set(rows)
        assert {
            "AEDense d64 default",
            "AEDense d256 default",
            "AESpatial c16 default",
            "AESpatial c64 default",
            "AEDense d256 384x240 default",
            "CNN-AE 704x384",
            "CNN-AE 352x192",
            "CNN-VAE 704x384",
            "CNN-VAE 352x192",
            "Simple Original CNN AE 352x192 Latent 300",
            "AESpatial c64 384x240 default",
            "VAE Baur d128 default",
            "STAE-3D Reconstruction 352x192",
            "STAE reconstruction prediction default",
            "STAE Reconstruction paper default",
            "STAE Reconstruction + Future Prediction paper default",
            "fastAnoGAN paper default",
        } == names
        assert len(rows) == len(default_method_payloads())
    finally:
        db.close()


@pytest.mark.parametrize("customized", [False, True])
def test_industrial_stae_bootstrap_updates_only_unchanged_old_default(customized) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = Session(engine)
    try:
        payload = next(item for item in default_method_payloads() if item.name == "STAE-3D Reconstruction 352x192")
        existing = create_method_configuration(db, _old_industrial_stae_payload(payload))
        existing = db.get(models.MethodConfiguration, existing.id)
        if customized:
            existing.method_config = {**existing.method_config, "frame_interval_seconds": 7.0}
            db.commit()
        assert ensure_default_method_configurations(db) == len(default_method_payloads()) - 1
        db.refresh(existing)
        assert existing.method_config["sequence_contiguity_mode"] == ("timestamp_interval" if customized else "ordered_index")
        assert existing.method_config.get("frame_interval_seconds") == (7.0 if customized else None)
        assert ensure_default_method_configurations(db) == 0
        db.refresh(existing)
        assert existing.method_config["sequence_contiguity_mode"] == ("timestamp_interval" if customized else "ordered_index")
    finally:
        db.close()


def test_cnn_ae_bootstrap_updates_only_unchanged_old_defaults() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = Session(engine)
    try:
        defaults = {payload.name: payload for payload in default_method_payloads()}
        original = create_method_configuration(db, _old_cnn_ae_payload(defaults["CNN-AE 704x384"]))
        customized = create_method_configuration(db, _old_cnn_ae_payload(defaults["CNN-AE 352x192"]))
        customized = db.get(models.MethodConfiguration, customized.id)
        customized.training_config = {**customized.training_config, "batch_size": 4}
        db.commit()

        assert ensure_default_method_configurations(db) == len(defaults) - 2
        original = db.get(models.MethodConfiguration, original.id)
        assert original.training_config["batch_size"] == 8
        assert original.training_config["early_stopping_patience"] == 5
        assert original.training_config["loss"] == "mse"
        assert original.method_graph["encoder"][1]["config"]["eps"] == 0.001
        assert original.method_graph["encoder"][2]["config"]["negative_slope"] == 0.3
        assert customized.training_config["batch_size"] == 4
        assert customized.method_graph["encoder"][1]["config"]["eps"] == 0.00001
        assert ensure_default_method_configurations(db) == 0
    finally:
        db.close()


@pytest.mark.parametrize('name,height,width', [
    ('CNN-VAE 704x384', 384, 704),
    ('CNN-VAE 352x192', 192, 352),
])
def test_cnn_vae_presets_forward_and_sampling(name, height, width) -> None:
    torch = pytest.importorskip('torch')
    payload = next(item for item in default_method_payloads() if item.name == name)
    configuration = SimpleNamespace(builder_kind=payload.method_graph['builder_kind'],
                                    method_graph=payload.method_graph, method_config=payload.method_config)
    model, is_vae = _build_model(torch, configuration)
    model.eval()
    sample = torch.zeros(1, 1, height, width)
    with torch.no_grad():
        first, (mu, logvar) = model(sample)
        second, _ = model(sample)
        model.deterministic_vae = True
        mean_first, _ = model(sample)
        mean_second, _ = model(sample)
    assert is_vae
    assert first.shape == second.shape == mean_first.shape == sample.shape
    assert mu.shape == logvar.shape == (1, 128)
    assert not torch.allclose(first, second)
    assert torch.equal(mean_first, mean_second)


def test_cnn_vae_pixel_sum_accepts_l1_but_rejects_ssim() -> None:
    payload = next(item for item in default_method_payloads() if item.name == 'CNN-VAE 352x192')
    l1 = payload.model_copy(deep=True)
    l1.training_config['reconstruction_loss'] = 'l1'
    assert validate_method_configuration(l1).valid
    ssim = payload.model_copy(deep=True)
    ssim.training_config['reconstruction_loss'] = 'ssim'
    result = validate_method_configuration(ssim)
    assert not result.valid
    assert any('only mse or l1' in error for error in result.errors)


def test_default_method_payloads_validate_statically() -> None:
    for payload in default_method_payloads():
        result = validate_method_configuration(payload)
        assert result.valid is True, payload.name
        assert result.errors == []
        if payload.method_type != "fast_anogan" and payload.name != "Simple Original CNN AE 352x192 Latent 300":
            assert payload.training_config["optimizer"] == "adam"
            assert payload.training_config["learning_rate"] == 0.0001
            assert payload.training_config["weight_decay"] == (0.0 if payload.name == "STAE-3D Reconstruction 352x192" else 0.00001)
            assert payload.training_config["early_stopping_enabled"] is True
            assert payload.training_config["early_stopping_patience"] == (5 if payload.name in {"STAE-3D Reconstruction 352x192", "CNN-AE 704x384", "CNN-AE 352x192", "CNN-VAE 704x384", "CNN-VAE 352x192"} else 10)
        if payload.name in {"CNN-VAE 704x384", "CNN-VAE 352x192"}:
            height, width = ((384, 704) if payload.name == "CNN-VAE 704x384" else (192, 352))
            graph = payload.method_graph
            assert payload.method_type == "cnn_vae"
            assert payload.method_config["output_activation"] == "none"
            assert payload.method_config["kl_weight"] == 1.0
            assert payload.training_config["batch_size"] == 8
            assert payload.training_config["reconstruction_loss"] == "mse"
            assert payload.training_config["reconstruction_reduction"] == "pixel_sum"
            assert payload.inference_config["sample_count"] == 1
            assert graph["builder_kind"] == "sequential_variational_autoencoder"
            assert graph["encoder"][-1]["type"] == "Flatten"
            assert graph["decoder"][0]["config"] == {"channels": 16, "height": height // 16, "width": width // 16}
            assert [layer["config"]["out_channels"] for layer in graph["encoder"] if layer["type"] == "Conv2d"] == [32, 64, 128, 128, 16]
            assert [layer["config"]["out_channels"] for layer in graph["decoder"] if layer["type"] == "ConvTranspose2d"] == [128, 64, 32, 32]
            assert graph["decoder"][-1]["config"]["out_channels"] == 1
            assert result.layer_specs[-1]["output_label"] == f"N,1,{height},{width}"
        if payload.name == "Simple Original CNN AE 352x192 Latent 300":
            assert payload.method_type == "cnn_autoencoder"
            assert payload.method_config == {
                "input_channels": 1, "input_width": 352, "input_height": 192,
                "latent_dim": 300, "output_activation": "sigmoid",
            }
            assert payload.training_config["batch_size"] == 16
            assert payload.training_config["learning_rate"] == 0.001
            assert payload.training_config["loss"] == "mse"
            assert "early_stopping_enabled" not in payload.training_config
            assert [layer["config"]["out_channels"] for layer in payload.method_graph["encoder"] if layer["type"] == "Conv2d"] == [32, 64, 128]
            assert [layer["config"]["out_channels"] for layer in payload.method_graph["decoder"] if layer["type"] == "ConvTranspose2d"] == [128, 64, 32]
            assert payload.method_graph["encoder"][-1]["config"]["out_features"] == 300
            assert payload.method_graph["decoder"][0]["config"]["out_features"] == 135168
            assert payload.method_graph["decoder"][1]["config"] == {"channels": 128, "height": 24, "width": 44}
            assert payload.method_graph["decoder"][-1]["config"] == {"out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1, "bias": True}
            assert result.layer_specs[-1]["output_label"] == "N,1,192,352"
        if payload.name == "VAE Baur d128 default":
            assert payload.method_type == "cnn_vae"
            assert payload.method_config["latent_dim"] == 128
            assert payload.method_config["kl_weight"] == 1.0
            assert payload.training_config["reconstruction_loss"] == "l1"
            assert "loss" not in payload.training_config
        if payload.name in {"CNN-AE 704x384", "CNN-AE 352x192"}:
            height, width = ((384, 704) if payload.name == "CNN-AE 704x384" else (192, 352))
            assert payload.method_type == "cnn_autoencoder"
            assert payload.method_config["output_activation"] == "none"
            assert payload.training_config["loss"] == "mse"
            assert payload.training_config["batch_size"] == 8
            assert payload.method_config["input_height"] == height
            assert payload.method_config["input_width"] == width
            encoder = payload.method_graph["encoder"]
            decoder = payload.method_graph["decoder"]
            layers = encoder + decoder
            assert all(layer["config"]["eps"] == 0.001 and layer["config"]["momentum"] == 0.01 for layer in layers if layer["type"] == "BatchNorm2d")
            assert all(layer["config"]["negative_slope"] == 0.3 for layer in layers if layer["type"] == "LeakyReLU")
            assert [layer["config"]["out_channels"] for layer in encoder if layer["type"] == "Conv2d"] == [32, 64, 128, 128, 16]
            assert [layer["config"]["out_channels"] for layer in decoder if layer["type"] == "ConvTranspose2d"] == [128, 64, 32, 32]
            assert decoder[0]["config"]["out_features"] == 16 * (height // 16) * (width // 16)
            assert decoder[1]["config"] == {"channels": 16, "height": height // 16, "width": width // 16}
            assert encoder[-1]["config"]["out_features"] == 128
            assert result.layer_specs[-1]["output_label"] == f"N,1,{height},{width}"
        if payload.name == "STAE reconstruction prediction default":
            assert payload.method_type == "spatiotemporal_autoencoder"
            assert payload.method_config["clip_length"] == 8
            assert payload.method_config["future_length"] == 1
            assert payload.training_config["training_objective"] == "reconstruction_prediction"
        if payload.name == "STAE Reconstruction paper default":
            assert payload.method_type == "spatiotemporal_autoencoder"
            assert payload.method_config["clip_length"] == 16
            assert payload.method_config["future_length"] == 0
            assert payload.method_config["prediction_branch"] is False
            assert payload.training_config["training_objective"] == "reconstruction"
            assert payload.inference_config["score_mode"] == "reconstruction_only"
        if payload.name == "STAE Reconstruction + Future Prediction paper default":
            assert payload.method_type == "spatiotemporal_autoencoder"
            assert payload.method_config["clip_length"] == 16
            assert payload.method_config["future_length"] == 16
            assert payload.method_config["prediction_branch"] is True
            assert payload.training_config["training_objective"] == "reconstruction_prediction"
            assert payload.training_config["prediction_weight_schedule"] == "linear_decay"
            assert payload.training_config["prediction_min_weight"] == 0.0
            assert payload.training_config["prediction_horizon_weight_schedule"] == "linear_decay"
        if payload.name == "fastAnoGAN paper default":
            assert payload.method_type == "fast_anogan"
            assert payload.method_config["input_width"] == 64
            assert payload.method_config["input_height"] == 64
            assert payload.method_config["latent_dim"] == 128
            assert payload.training_config["critic_updates_per_generator"] == 5
            assert payload.training_config["encoder_training_mode"] == "izif"
            assert [block["out_channels"] for block in payload.method_graph["generator_blocks"]] == [512, 256, 128, 64]
            assert [block["normalization"] for block in payload.method_graph["critic_blocks"]] == ["layer_norm"] * 4
            assert [block["direction"] for block in payload.method_graph["encoder_blocks"]] == ["down"] * 4


def test_spatial_autoencoder_rejects_flat_bottleneck() -> None:
    payload = next(item for item in default_method_payloads() if item.name == "AESpatial c16 default").model_copy(deep=True)
    payload.method_graph["encoder"].append({"id": "bad-flat", "type": "Flatten", "config": {}})
    result = validate_method_configuration(payload)
    assert result.valid is False
    assert "must remain spatial rank 4" in " ".join(result.errors)


def test_fast_anogan_rejects_batch_norm_critic_blocks() -> None:
    payload = next(item for item in default_method_payloads() if item.name == "fastAnoGAN paper default").model_copy(deep=True)
    payload.method_graph["critic_blocks"][0]["normalization"] = "batch_norm"
    result = validate_method_configuration(payload)
    assert result.valid is False
    assert "batch_norm" in " ".join(result.errors)


def test_fast_anogan_torch_forward_shapes_and_feature_score() -> None:
    torch = pytest.importorskip("torch")
    payload = next(item for item in default_method_payloads() if item.name == "fastAnoGAN paper default")
    generator, critic, encoder = build_fast_anogan_modules(torch, payload.method_graph, payload.method_config)
    x = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
    z = torch.zeros((2, 128), dtype=torch.float32)

    generated = generator(z)
    output = fast_anogan_forward(generator, critic, encoder, x)
    feature_score = ((output.real_features - output.reconstruction_features) ** 2).flatten(1).mean(dim=1)

    assert tuple(generated.shape) == (2, 1, 64, 64)
    assert tuple(output.reconstruction.shape) == (2, 1, 64, 64)
    assert tuple(output.latent.shape) == (2, 128)
    assert tuple(feature_score.shape) == (2,)


def test_spatiotemporal_default_validates_reconstruction_and_prediction() -> None:
    payload = next(item for item in default_method_payloads() if item.name == "STAE reconstruction prediction default")
    result = validate_method_configuration(payload)
    assert result.valid is True
    sections = {spec["section"] for spec in result.layer_specs}
    assert {"encoder", "decoder", "prediction_decoder"}.issubset(sections)
    decoder_outputs = [spec["output_label"] for spec in result.layer_specs if spec["section"] == "decoder"]
    prediction_outputs = [spec["output_label"] for spec in result.layer_specs if spec["section"] == "prediction_decoder"]
    assert decoder_outputs[-1] == "N,1,8,256,256"
    assert prediction_outputs[-1] == "N,1,1,256,256"


def test_paper_stae_defaults_validate_expected_shapes() -> None:
    reconstruction = next(item for item in default_method_payloads() if item.name == "STAE Reconstruction paper default")
    reconstruction_result = validate_method_configuration(reconstruction)
    assert reconstruction_result.valid is True
    reconstruction_sections = {spec["section"] for spec in reconstruction_result.layer_specs}
    assert "prediction_decoder" not in reconstruction_sections
    reconstruction_decoder_outputs = [
        spec["output_label"] for spec in reconstruction_result.layer_specs if spec["section"] == "decoder"
    ]
    assert reconstruction_decoder_outputs[-1] == "N,1,16,128,128"

    prediction = next(
        item for item in default_method_payloads() if item.name == "STAE Reconstruction + Future Prediction paper default"
    )
    prediction_result = validate_method_configuration(prediction)
    assert prediction_result.valid is True
    prediction_decoder_outputs = [
        spec["output_label"] for spec in prediction_result.layer_specs if spec["section"] == "prediction_decoder"
    ]
    reconstruction_decoder_outputs = [
        spec["output_label"] for spec in prediction_result.layer_specs if spec["section"] == "decoder"
    ]
    assert reconstruction_decoder_outputs[-1] == "N,1,16,128,128"
    assert prediction_decoder_outputs[-1] == "N,1,16,128,128"


def test_paper_stae_defaults_use_zhao_like_encoder_channels() -> None:
    payload = next(item for item in default_method_payloads() if item.name == "STAE Reconstruction paper default")
    conv_channels = [
        layer["config"]["out_channels"]
        for layer in payload.method_graph["encoder"]
        if layer["type"] == "Conv3d"
    ]
    pool_count = sum(1 for layer in payload.method_graph["encoder"] if layer["type"] == "MaxPool3d")
    assert conv_channels == [32, 48, 64, 64]
    assert pool_count == 3


def test_paper_stae_prediction_horizon_decay_weights_near_future_more_strongly() -> None:
    torch = pytest.importorskip("torch")
    training_parameters = {
        "prediction_weight": 1.0,
        "prediction_weight_schedule": "linear_decay",
        "prediction_min_weight": 0.0,
        "prediction_horizon_weight_schedule": "linear_decay",
    }
    weights = _prediction_horizon_weights(torch, training_parameters, 16, torch.device("cpu"))
    assert weights[0].item() == pytest.approx(1.0)
    assert weights[-1].item() == pytest.approx(0.0)
    assert weights[0].item() > weights[8].item() > weights[-1].item()
    assert _prediction_weight_for_epoch(training_parameters, epoch=999, epochs=1000) == pytest.approx(1.0)


def test_default_bootstrap_repairs_only_unmodified_old_paper_stae_defaults() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        old_payloads = _old_paper_payloads_by_name()
        create_method_configuration(db, old_payloads["STAE Reconstruction + Future Prediction paper default"])
        assert ensure_default_method_configurations(db) == len(default_method_payloads()) - 1
        repaired = db.scalar(
            select(models.MethodConfiguration).where(
                models.MethodConfiguration.name == "STAE Reconstruction + Future Prediction paper default"
            )
        )
        assert repaired is not None
        assert repaired.method_config["future_length"] == 16
        conv_channels = [
            layer["config"]["out_channels"]
            for layer in repaired.method_graph["encoder"]
            if layer["type"] == "Conv3d"
        ]
        assert conv_channels == [32, 48, 64, 64]
    finally:
        db.close()

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        modified = _old_paper_payloads_by_name()["STAE Reconstruction + Future Prediction paper default"].model_copy(deep=True)
        modified.description = "User modified one-step STAE"
        create_method_configuration(db, modified)
        assert ensure_default_method_configurations(db) == len(default_method_payloads()) - 1
        preserved = db.scalar(
            select(models.MethodConfiguration).where(
                models.MethodConfiguration.name == "STAE Reconstruction + Future Prediction paper default"
            )
        )
        assert preserved is not None
        assert preserved.description == "User modified one-step STAE"
        assert preserved.method_config["future_length"] == 1
    finally:
        db.close()


def test_valid_cnn_vae_returns_layer_specs() -> None:
    client_iter = make_client()
    client = next(client_iter)
    try:
        payload = autoencoder_payload("CNN-VAE latent64 beta1")
        payload["method_type"] = "cnn_vae"
        payload["method_config"]["kl_weight"] = 1.0
        payload["method_graph"]["builder_kind"] = "sequential_variational_autoencoder"
        payload["method_graph"]["latent"] = {
            "latent_dim": 64,
            "kl_weight": 1.0,
            "reparameterization": True,
        }
        response = client.post("/api/methods/configurations/validate", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert any(node["label"] == "Mu/logvar projection" for node in body["diagram"]["nodes"])
        assert any(node["label"] == "Decoder seed projection" for node in body["diagram"]["nodes"])
        assert len(body["layer_specs"]) == 7
    finally:
        try:
            next(client_iter)
        except StopIteration:
            pass
