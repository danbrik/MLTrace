from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.modeling.architectures.cnn_autoencoder import CnnAutoencoderArchitecture
from app.modeling.registry import MethodRegistry, registry


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

    assert {"cnn_autoencoder", "cnn_vae", "mean_image"}.issubset(method_types)


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
