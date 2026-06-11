from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import Any


def merge_defaults(defaults: dict, values: dict | None) -> dict:
    merged = deepcopy(defaults)
    merged.update(values or {})
    return merged


def validate_schema_values(schema: dict, values: dict, prefix: str) -> None:
    """Validate a small JSON-schema-like config contract used by MLTrace builders."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for key in required:
        if key not in values or values[key] is None:
            raise ValueError(f"{prefix}.{key} is required.")

    for key, prop in properties.items():
        if key not in values or values[key] is None:
            continue
        value = values[key]
        expected_type = prop.get("type")
        if expected_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{prefix}.{key} must be an integer, got {value!r}.")
        elif expected_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{prefix}.{key} must be a number, got {value!r}.")
        elif expected_type == "string":
            if not isinstance(value, str):
                raise ValueError(f"{prefix}.{key} must be a string, got {value!r}.")
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{prefix}.{key} must be a boolean, got {value!r}.")

        enum = prop.get("enum")
        if enum is not None and value not in enum:
            raise ValueError(f"{prefix}.{key} must be one of {enum}, got {value!r}.")
        minimum = prop.get("minimum")
        maximum = prop.get("maximum")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if minimum is not None and value < minimum:
                raise ValueError(f"{prefix}.{key} must be >= {minimum}, got {value}.")
            if maximum is not None and value > maximum:
                raise ValueError(f"{prefix}.{key} must be <= {maximum}, got {value}.")


class BaseMethodDefinition(ABC):
    """Base class for saved anomaly-detection method definitions.

    A method may be a neural model, statistical baseline, feature memory, temporal
    algorithm, or any later anomaly-scoring approach.
    """

    type: str
    label: str
    category: str
    description: str
    framework: str = "generic"
    method_family: str
    method_version: str = "1"
    training_mode: str
    requires_training: bool
    supports_training_pipeline: bool
    artifact_kind: str
    builder_kind: str
    capabilities: dict[str, Any] = {}
    method_schema: dict = {"type": "object", "properties": {}}
    training_schema: dict = {"type": "object", "properties": {}}
    inference_schema: dict = {"type": "object", "properties": {}}
    default_method_config: dict = {}
    default_training_config: dict = {}
    default_inference_config: dict = {}

    @property
    def architecture_version(self) -> str:
        return self.method_version

    @property
    def model_schema(self) -> dict:
        return self.method_schema

    @property
    def default_model_config(self) -> dict:
        return self.default_method_config

    def merged_method_config(self, config: dict | None) -> dict:
        return merge_defaults(self.default_method_config, config)

    def merged_model_config(self, config: dict | None) -> dict:
        return self.merged_method_config(config)

    def merged_training_config(self, config: dict | None) -> dict:
        return merge_defaults(self.default_training_config, config)

    def merged_inference_config(self, config: dict | None) -> dict:
        return merge_defaults(self.default_inference_config, config)

    def validate_config(
        self,
        method_graph: dict | None,
        method_config: dict | None,
        training_config: dict | None = None,
        inference_config: dict | None = None,
    ) -> None:
        validate_schema_values(
            self.method_schema,
            self.merged_method_config(method_config),
            f"{self.type}.method_config",
        )
        validate_schema_values(
            self.training_schema,
            self.merged_training_config(training_config),
            f"{self.type}.training_config",
        )
        validate_schema_values(
            self.inference_schema,
            self.merged_inference_config(inference_config),
            f"{self.type}.inference_config",
        )


BaseModelArchitecture = BaseMethodDefinition
