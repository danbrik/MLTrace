from __future__ import annotations

from dataclasses import dataclass

from app.modeling.base import merge_defaults, validate_schema_values


@dataclass(frozen=True)
class LayerDefinition:
    type: str
    label: str
    category: str
    config_schema: dict
    default_config: dict
    input_rank: int | None
    output_rank: int | None
    shape_notes: str | None = None


LAYER_DEFINITIONS: tuple[LayerDefinition, ...] = (
    LayerDefinition(
        type="Conv2d",
        label="Conv2d",
        category="Convolution",
        input_rank=4,
        output_rank=4,
        default_config={"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "bias": True},
        config_schema={
            "type": "object",
            "required": ["out_channels", "kernel_size"],
            "properties": {
                "out_channels": {"type": "integer", "label": "Out channels", "minimum": 1, "default": 16},
                "kernel_size": {"type": "integer", "label": "Kernel size", "minimum": 1, "default": 3},
                "stride": {"type": "integer", "label": "Stride", "minimum": 1, "default": 1},
                "padding": {"type": "integer", "label": "Padding", "minimum": 0, "default": 1},
                "bias": {"type": "boolean", "label": "Bias", "default": True},
            },
        },
        shape_notes="Output size follows torch.nn.Conv2d semantics.",
    ),
    LayerDefinition(
        type="ConvTranspose2d",
        label="ConvTranspose2d",
        category="Convolution",
        input_rank=4,
        output_rank=4,
        default_config={
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "output_padding": 0,
            "bias": True,
        },
        config_schema={
            "type": "object",
            "required": ["out_channels", "kernel_size"],
            "properties": {
                "out_channels": {"type": "integer", "label": "Out channels", "minimum": 1, "default": 16},
                "kernel_size": {"type": "integer", "label": "Kernel size", "minimum": 1, "default": 3},
                "stride": {"type": "integer", "label": "Stride", "minimum": 1, "default": 1},
                "padding": {"type": "integer", "label": "Padding", "minimum": 0, "default": 1},
                "output_padding": {"type": "integer", "label": "Output padding", "minimum": 0, "default": 0},
                "bias": {"type": "boolean", "label": "Bias", "default": True},
            },
        },
        shape_notes="Output size follows torch.nn.ConvTranspose2d semantics.",
    ),
    LayerDefinition(
        type="BatchNorm2d",
        label="BatchNorm2d",
        category="Normalization",
        input_rank=4,
        output_rank=4,
        default_config={"num_features": 16, "eps": 0.00001, "momentum": 0.1},
        config_schema={
            "type": "object",
            "required": ["num_features"],
            "properties": {
                "num_features": {"type": "integer", "label": "Features", "minimum": 1, "default": 16},
                "eps": {"type": "number", "label": "Epsilon", "minimum": 0, "default": 0.00001},
                "momentum": {"type": "number", "label": "Momentum", "minimum": 0, "maximum": 1, "default": 0.1},
            },
        },
    ),
    LayerDefinition(
        type="MaxPool2d",
        label="MaxPool2d",
        category="Pooling",
        input_rank=4,
        output_rank=4,
        default_config={"kernel_size": 2, "stride": 2, "padding": 0},
        config_schema={
            "type": "object",
            "required": ["kernel_size"],
            "properties": {
                "kernel_size": {"type": "integer", "label": "Kernel size", "minimum": 1, "default": 2},
                "stride": {"type": "integer", "label": "Stride", "minimum": 1, "default": 2},
                "padding": {"type": "integer", "label": "Padding", "minimum": 0, "default": 0},
            },
        },
    ),
    LayerDefinition(
        type="Upsample",
        label="Upsample",
        category="Resampling",
        input_rank=4,
        output_rank=4,
        default_config={"scale_factor": 2, "mode": "nearest"},
        config_schema={
            "type": "object",
            "required": ["scale_factor", "mode"],
            "properties": {
                "scale_factor": {"type": "number", "label": "Scale factor", "minimum": 0.01, "default": 2},
                "mode": {"type": "string", "label": "Mode", "enum": ["nearest", "bilinear"], "default": "nearest"},
            },
        },
    ),
    LayerDefinition(
        type="Dropout2d",
        label="Dropout2d",
        category="Regularization",
        input_rank=4,
        output_rank=4,
        default_config={"p": 0.1},
        config_schema={
            "type": "object",
            "properties": {"p": {"type": "number", "label": "Probability", "minimum": 0, "maximum": 1, "default": 0.1}},
        },
    ),
    LayerDefinition(
        type="Flatten",
        label="Flatten",
        category="Shape",
        input_rank=None,
        output_rank=2,
        default_config={"start_dim": 1, "end_dim": -1},
        config_schema={
            "type": "object",
            "properties": {
                "start_dim": {"type": "integer", "label": "Start dim", "default": 1},
                "end_dim": {"type": "integer", "label": "End dim", "default": -1},
            },
        },
    ),
    LayerDefinition(
        type="Unflatten",
        label="Unflatten",
        category="Shape",
        input_rank=2,
        output_rank=4,
        default_config={"channels": 16, "height": 10, "width": 10},
        config_schema={
            "type": "object",
            "required": ["channels", "height", "width"],
            "properties": {
                "channels": {"type": "integer", "label": "Channels", "minimum": 1, "default": 16},
                "height": {"type": "integer", "label": "Height", "minimum": 1, "default": 10},
                "width": {"type": "integer", "label": "Width", "minimum": 1, "default": 10},
            },
        },
    ),
    LayerDefinition(
        type="Linear",
        label="Linear",
        category="Dense",
        input_rank=2,
        output_rank=2,
        default_config={"out_features": 128, "bias": True},
        config_schema={
            "type": "object",
            "required": ["out_features"],
            "properties": {
                "out_features": {"type": "integer", "label": "Out features", "minimum": 1, "default": 128},
                "bias": {"type": "boolean", "label": "Bias", "default": True},
            },
        },
    ),
    LayerDefinition(
        type="ReLU",
        label="ReLU",
        category="Activation",
        input_rank=None,
        output_rank=None,
        default_config={"inplace": False},
        config_schema={"type": "object", "properties": {"inplace": {"type": "boolean", "label": "Inplace", "default": False}}},
    ),
    LayerDefinition(
        type="LeakyReLU",
        label="LeakyReLU",
        category="Activation",
        input_rank=None,
        output_rank=None,
        default_config={"negative_slope": 0.01, "inplace": False},
        config_schema={
            "type": "object",
            "properties": {
                "negative_slope": {"type": "number", "label": "Negative slope", "minimum": 0, "default": 0.01},
                "inplace": {"type": "boolean", "label": "Inplace", "default": False},
            },
        },
    ),
    LayerDefinition(
        type="GELU",
        label="GELU",
        category="Activation",
        input_rank=None,
        output_rank=None,
        default_config={},
        config_schema={"type": "object", "properties": {}},
    ),
    LayerDefinition(
        type="Sigmoid",
        label="Sigmoid",
        category="Activation",
        input_rank=None,
        output_rank=None,
        default_config={},
        config_schema={"type": "object", "properties": {}},
    ),
    LayerDefinition(
        type="Tanh",
        label="Tanh",
        category="Activation",
        input_rank=None,
        output_rank=None,
        default_config={},
        config_schema={"type": "object", "properties": {}},
    ),
)


LAYER_BY_TYPE = {definition.type: definition for definition in LAYER_DEFINITIONS}


def list_layer_definitions() -> list[LayerDefinition]:
    return sorted(LAYER_DEFINITIONS, key=lambda item: (item.category, item.label))


def get_layer_definition(layer_type: str) -> LayerDefinition:
    try:
        return LAYER_BY_TYPE[layer_type]
    except KeyError as exc:
        raise ValueError(f"Unknown layer type: {layer_type}") from exc


def validate_layer(layer: dict, prefix: str) -> dict:
    layer_type = layer.get("type")
    if not isinstance(layer_type, str) or not layer_type:
        raise ValueError(f"{prefix}.type is required.")
    definition = get_layer_definition(layer_type)
    config = merge_defaults(definition.default_config, layer.get("config") or {})
    validate_schema_values(definition.config_schema, config, f"{prefix}.{layer_type}")
    return {"id": layer.get("id"), "type": layer_type, "config": config}
