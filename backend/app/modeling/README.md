# MLTrace Modeling Extension Guide

This directory defines the reusable anomaly-detection methods that can be saved in MLTrace.
A method can be a neural architecture, a statistical baseline, or a later non-neural approach such as PatchCore or Optical Flow.

## Directory Layout

```text
backend/app/modeling/
  base.py                         # BaseMethodDefinition / BaseModelArchitecture
  registry.py                     # Global method registry
  validation.py                   # Static CNN tensor contract validation
  architectures/
    __init__.py                   # Auto-discovers architecture classes
    cnn_autoencoder.py            # Example: trainable neural method
    cnn_vae.py                    # Example: trainable variational neural method
    mean_image.py                 # Example: fit/training statistical method
  layers/
    catalog.py                    # Curated layer catalog for sequential CNN builders
```

## Which Class to Inherit From

Create one Python file per method under `backend/app/modeling/architectures/`.
Inside that file, define one class that inherits from `BaseMethodDefinition`.

For backwards compatibility the code also exposes `BaseModelArchitecture` as an alias, so existing classes currently inherit from:

```python
from app.modeling.base import BaseModelArchitecture


class MyMethodArchitecture(BaseModelArchitecture):
    ...
```

For new code, prefer this wording when possible:

```python
from app.modeling.base import BaseMethodDefinition


class MyMethod(BaseMethodDefinition):
    ...
```

Both forms work because `BaseModelArchitecture = BaseMethodDefinition`.

## Auto-Discovery

`backend/app/modeling/architectures/__init__.py` automatically imports every Python module in `architectures/` except files whose name starts with `_` and `common.py`.
Every non-abstract class that inherits from `BaseModelArchitecture` is instantiated and registered.

To add a method:

1. Add `backend/app/modeling/architectures/my_method.py`.
2. Define exactly one concrete method class in that file.
3. Give it a globally unique `type`.
4. Restart the backend or reload the app process.
5. Confirm `GET /api/methods/definitions` or `GET /api/models/architectures` includes it.

Do not manually import the new file in the registry.

## Required Metadata

Every method class should define these class attributes:

```python
type = "my_method"                         # Stable snake_case id, never reuse for a different meaning
label = "My Method"                        # User-facing name
category = "Neural reconstruction"         # UI grouping
description = "Short description."
framework = "torch_optional"               # Example values: torch_optional, numpy, opencv, generic
method_family = "neural_reconstruction"    # Broad family for filtering
method_version = "1"                       # Increment when config semantics change
training_mode = "gradient"                 # gradient | fit | none
requires_training = True
supports_training_pipeline = True
artifact_kind = "weights"                  # weights | mean_image | memory_bank | flow_reference | ...
builder_kind = "sequential_autoencoder"    # sequential_autoencoder | sequential_variational_autoencoder | form | ...
capabilities = {}
```

Important rule: `supports_training_pipeline` should be `True` when the method needs a later artifact-building step.
That includes gradient-trained neural methods and fit-based methods such as Mean Image.
Use `training_mode` to distinguish how the artifact is produced.

Use these meanings consistently:

- `training_mode = "gradient"`: the method needs gradient-based training, usually with Torch weights.
- `training_mode = "fit"`: the method trains/fits a reference artifact from normal data, but does not train with gradients.
- `training_mode = "none"`: the method has no training or fitting phase.
- `artifact_kind = "weights"`: a neural model weight artifact will exist later.
- `artifact_kind = "mean_image"`: a mean/reference image artifact will exist later.
- Future artifact kinds can be added without changing the DB shape.

## Configuration Schemas

Methods expose three schema/default pairs:

```python
method_schema = {"type": "object", "properties": {...}}
training_schema = {"type": "object", "properties": {...}}
inference_schema = {"type": "object", "properties": {...}}

default_method_config = {...}
default_training_config = {...}
default_inference_config = {...}
```

The current schema validator supports a small JSON-schema-like subset:

- property types: `integer`, `number`, `string`, `boolean`
- `required`
- `enum`
- `minimum`
- `maximum`
- `default`
- `label`

Example:

```python
default_method_config = {
    "input_channels": 1,
    "input_width": 160,
    "input_height": 120,
    "latent_dim": 64,
    "output_activation": "sigmoid",
}

method_schema = {
    "type": "object",
    "required": ["input_channels", "input_width", "input_height", "latent_dim"],
    "properties": {
        "input_channels": {"type": "integer", "label": "Input channels", "minimum": 1, "default": 1},
        "input_width": {"type": "integer", "label": "Input width", "minimum": 1, "default": 160},
        "input_height": {"type": "integer", "label": "Input height", "minimum": 1, "default": 120},
        "latent_dim": {"type": "integer", "label": "Latent dim", "minimum": 1, "default": 64},
        "output_activation": {
            "type": "string",
            "label": "Output activation",
            "enum": ["none", "sigmoid", "tanh"],
            "default": "sigmoid",
        },
    },
}
```

Keep top-level config values scalar when they should be filterable later.
MLTrace writes scalar values into `method_configuration_parameters` so future UI filters can search by paths such as `method_config.latent_dim`.

## Builder Kinds

### `form`

Use `builder_kind = "form"` for methods without a layer graph.
Examples: Mean Image, later simple Optical Flow references, threshold baselines, or non-neural fit methods.

Rules:

- The UI renders schema fields instead of layer controls.
- `_normalize_method_payload()` stores an empty graph for form builders.
- `validate_config()` should reject unexpected graph content if the method must remain form-only.

### `sequential_autoencoder`

Use this for CNN autoencoders with:

```text
input -> encoder layers -> latent bridge -> decoder layers -> output
```

Rules:

- `method_graph.encoder` must be a non-empty list.
- `method_graph.decoder` must be a non-empty list.
- Each layer type must exist in `backend/app/modeling/layers/catalog.py`.
- Static shape validation must end with decoder output matching `input_channels,input_height,input_width`.

### `sequential_variational_autoencoder`

Same as `sequential_autoencoder`, but the latent bridge represents VAE `mu/logvar` metadata.
The static validator uses the deterministic latent path for shape checking.

## Validation Pattern

Always override `validate_config()` when the method has graph-specific rules.
Call `super().validate_config(...)` first so schema validation still runs.

Form example:

```python
def validate_config(self, method_graph, method_config, training_config=None, inference_config=None) -> None:
    super().validate_config(method_graph, method_config, training_config, inference_config)
    graph = method_graph or {}
    if graph.get("encoder") or graph.get("decoder") or graph.get("nodes") or graph.get("edges"):
        raise ValueError("My Method uses a form builder and cannot contain a layer graph.")
```

Sequential CNN example:

```python
from app.modeling.architectures.common import validate_sequential_model_graph


def validate_config(self, method_graph, method_config, training_config=None, inference_config=None) -> None:
    super().validate_config(method_graph, method_config, training_config, inference_config)
    validate_sequential_model_graph(method_graph, self.builder_kind)
```

For non-form builders, `app.services._normalize_method_payload()` also runs `validate_cnn_tensor_contract()`.
That validator tracks rank, channels, height, width, and features through the configured layer list.
It rejects invalid rank transitions, bad BatchNorm feature counts, impossible spatial sizes, bad Unflatten products, and output-size mismatches.

If Torch is installed, the validator also runs a CPU dummy-forward check.
Missing Torch is not a blocker; a Torch failure is a hard validation error.

## Adding Layer Types

For sequential CNN builders, add layer types to `backend/app/modeling/layers/catalog.py`.

Each layer definition needs:

- `type`
- `label`
- `category`
- `config_schema`
- `default_config`
- `input_rank`
- `output_rank`
- optional `shape_notes`

If the layer changes tensor shape, also update `backend/app/modeling/validation.py` so `_infer_layer_output()` can infer the output spec.
If Torch dummy-forward support exists for that layer, update the Torch construction path there as well.

Do not expose arbitrary Python imports from the UI.
The catalog is deliberately curated so saved configs remain stable and safe to validate.

## SSIM Losses And Scores

MLTrace supports SSIM as an optional reconstruction loss and scoring metric.
It does not replace MSE or MAE; method schemas can expose all of these options:

- `mse`
- `l1` / MAE
- `smooth_l1`
- `ssim`
- `mae_ssim`
- `mse_ssim`

Combined losses use:

```text
(1 - ssim_weight) * pixel_loss + ssim_weight * ssim_loss
```

where `pixel_loss` is either MAE or MSE. `ssim_weight` is only meaningful for
the combined modes.

For inference and analysis, `error_metric=ssim_distance` means local
`1 - SSIM`. Heatmaps can also use `residual_source=ssim_residual`.

Important constant convention: `ssim_k1` and `ssim_k2` are standard SSIM
K constants, not direct C constants. The implementation computes:

```text
C1 = (ssim_k1 * ssim_data_range)^2
C2 = (ssim_k2 * ssim_data_range)^2
```

With the default `ssim_data_range=1.0`, `ssim_k1=0.01` and `ssim_k2=0.03`
therefore produce `C1=0.0001` and `C2=0.0009`. Do not document these as
direct `C1=0.01` / `C2=0.03` values.

## fastAnoGAN Method

`fast_anogan` is not a sequential encoder/decoder method. It stores a block graph
with three sections:

- `generator_blocks`: residual upsampling blocks, `z -> image`.
- `critic_blocks`: residual downsampling blocks, `image -> critic/features`.
- `encoder_blocks`: residual downsampling blocks, `image -> z`.

The paper-near default uses `64x64x1` input, `latent_dim=128`, generator
channels `[512, 256, 128, 64]`, critic channels `[128, 256, 512, 512]`, and
`kappa=1.0`.

The critic must not use BatchNorm. WGAN-GP applies a per-sample gradient
penalty, and BatchNorm couples samples in a batch. MLTrace therefore validates
critic blocks so `normalization=batch_norm` is rejected. Use `layer_norm`
implemented as `GroupNorm(1, C)` for 2D feature maps, or `none`.

Training writes a `gan_bundle` artifact containing generator, critic, and
encoder weights. Testing computes:

```text
z = E(x)
x_hat = G(z)
residual_score = mean((x - x_hat)^2)
feature_score = mean((D_feat(x) - D_feat(x_hat))^2)
combined_score = residual_score + kappa * feature_score
```

The feature mean is computed over all non-batch feature elements, matching the
paper's `1 / n_d` normalization.

## Persistence Model

Saved methods are stored in a generic table:

```text
method_configurations
```

It stores:

- method identity metadata
- `method_graph`
- `method_config`
- `training_config`
- `inference_config`
- generated `diagram`
- validation result

Scalar values are indexed in:

```text
method_configuration_parameters
```

Do not add one database table per method unless there is a strong reason.
Different method-specific parameters belong in JSON config plus the scalar index.

## API Surface

Primary endpoints:

```text
GET    /api/methods/definitions
GET    /api/methods/definitions/{method_type}
GET    /api/methods/layers
GET    /api/methods/configurations
POST   /api/methods/configurations
GET    /api/methods/configurations/{id}
PUT    /api/methods/configurations/{id}
DELETE /api/methods/configurations/{id}
POST   /api/methods/configurations/validate
POST   /api/methods/configurations/diagram
```

Temporary compatibility aliases under `/api/models/...` still exist.
New code should use `/api/methods/...`.

## Checklist for a New Method

Before considering a new method complete:

1. Add one class in `backend/app/modeling/architectures/<method>.py`.
2. Use a stable unique `type`.
3. Choose `training_mode`, `artifact_kind`, and `builder_kind` deliberately.
4. Define `method_schema` and defaults.
5. Keep filter-relevant values as top-level scalar config keys.
6. Override `validate_config()` and call `super()`.
7. For layer builders, add missing layer catalog entries and shape inference.
8. Add backend tests for discovery, validation success, validation failures, and save/load round-trip.
9. Run backend tests and frontend build.

## Minimal Form Method Template

```python
from __future__ import annotations

from app.modeling.base import BaseMethodDefinition


class MyBaselineMethod(BaseMethodDefinition):
    type = "my_baseline"
    label = "My Baseline"
    category = "Baseline reconstruction"
    description = "Stores configuration for a simple baseline method."
    framework = "numpy"
    method_family = "statistical_baseline"
    method_version = "1"
    training_mode = "fit"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "reference"
    builder_kind = "form"
    capabilities = {
        "input_kind": "image_collection",
        "output_kind": "reference",
        "supports_layer_builder": False,
        "supports_training": True,
    }
    default_method_config = {"aggregation": "mean"}
    method_schema = {
        "type": "object",
        "required": ["aggregation"],
        "properties": {
            "aggregation": {
                "type": "string",
                "label": "Aggregation",
                "enum": ["mean"],
                "default": "mean",
            },
        },
    }

    def validate_config(self, method_graph, method_config, training_config=None, inference_config=None) -> None:
        super().validate_config(method_graph, method_config, training_config, inference_config)
        graph = method_graph or {}
        if graph:
            raise ValueError("My Baseline uses a form builder and cannot contain a graph.")
```

## Minimal Sequential CNN Method Template

```python
from __future__ import annotations

from app.modeling.architectures.common import validate_sequential_model_graph
from app.modeling.base import BaseMethodDefinition


class MyCnnMethod(BaseMethodDefinition):
    type = "my_cnn_method"
    label = "My CNN Method"
    category = "Neural reconstruction"
    description = "Sequential CNN reconstruction method."
    framework = "torch_optional"
    method_family = "neural_reconstruction"
    method_version = "1"
    training_mode = "gradient"
    requires_training = True
    supports_training_pipeline = True
    artifact_kind = "weights"
    builder_kind = "sequential_autoencoder"
    capabilities = {
        "input_kind": "image",
        "output_kind": "reconstruction",
        "supports_layer_builder": True,
        "supports_training": True,
    }
    default_method_config = {
        "input_channels": 1,
        "input_width": 160,
        "input_height": 120,
        "latent_dim": 64,
        "output_activation": "sigmoid",
    }
    method_schema = {
        "type": "object",
        "required": ["input_channels", "input_width", "input_height", "latent_dim", "output_activation"],
        "properties": {
            "input_channels": {"type": "integer", "label": "Input channels", "minimum": 1, "default": 1},
            "input_width": {"type": "integer", "label": "Input width", "minimum": 1, "default": 160},
            "input_height": {"type": "integer", "label": "Input height", "minimum": 1, "default": 120},
            "latent_dim": {"type": "integer", "label": "Latent dim", "minimum": 1, "default": 64},
            "output_activation": {
                "type": "string",
                "label": "Output activation",
                "enum": ["none", "sigmoid", "tanh"],
                "default": "sigmoid",
            },
        },
    }

    def validate_config(self, method_graph, method_config, training_config=None, inference_config=None) -> None:
        super().validate_config(method_graph, method_config, training_config, inference_config)
        validate_sequential_model_graph(method_graph, self.builder_kind)
```
