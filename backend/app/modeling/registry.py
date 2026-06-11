from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class MethodDefinition(Protocol):
    type: str
    label: str
    category: str
    description: str
    framework: str
    method_family: str
    method_version: str
    training_mode: str
    requires_training: bool
    supports_training_pipeline: bool
    artifact_kind: str
    builder_kind: str
    capabilities: dict
    method_schema: dict
    training_schema: dict
    inference_schema: dict
    default_method_config: dict
    default_training_config: dict
    default_inference_config: dict

    def merged_method_config(self, config: dict | None) -> dict:
        ...

    def merged_training_config(self, config: dict | None) -> dict:
        ...

    def merged_inference_config(self, config: dict | None) -> dict:
        ...

    def validate_config(
        self,
        method_graph: dict | None,
        method_config: dict | None,
        training_config: dict | None = None,
        inference_config: dict | None = None,
    ) -> None:
        ...


@dataclass(frozen=True)
class MethodDefinitionRead:
    type: str
    label: str
    category: str
    description: str
    framework: str
    method_family: str
    method_version: str
    training_mode: str
    requires_training: bool
    supports_training_pipeline: bool
    artifact_kind: str
    builder_kind: str
    capabilities: dict
    method_schema: dict
    training_schema: dict
    inference_schema: dict
    default_method_config: dict
    default_training_config: dict
    default_inference_config: dict

    @property
    def architecture_version(self) -> str:
        return self.method_version

    @property
    def model_schema(self) -> dict:
        return self.method_schema

    @property
    def default_model_config(self) -> dict:
        return self.default_method_config


class MethodRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, MethodDefinition] = {}

    def register(self, method: MethodDefinition) -> None:
        if method.type in self._methods:
            raise ValueError(f"Method is already registered: {method.type}")
        self._methods[method.type] = method

    def get(self, method_type: str) -> MethodDefinition:
        try:
            return self._methods[method_type]
        except KeyError as exc:
            raise ValueError(f"Unknown method: {method_type}") from exc

    def list_definitions(self) -> list[MethodDefinitionRead]:
        return [
            MethodDefinitionRead(
                type=method.type,
                label=method.label,
                category=method.category,
                description=method.description,
                framework=method.framework,
                method_family=method.method_family,
                method_version=method.method_version,
                training_mode=method.training_mode,
                requires_training=method.requires_training,
                supports_training_pipeline=method.supports_training_pipeline,
                artifact_kind=method.artifact_kind,
                builder_kind=method.builder_kind,
                capabilities=method.capabilities,
                method_schema=method.method_schema,
                training_schema=method.training_schema,
                inference_schema=method.inference_schema,
                default_method_config=method.default_method_config,
                default_training_config=method.default_training_config,
                default_inference_config=method.default_inference_config,
            )
            for method in sorted(self._methods.values(), key=lambda item: (item.category, item.label))
        ]


ModelArchitecture = MethodDefinition
ArchitectureDefinition = MethodDefinitionRead
ModelArchitectureRegistry = MethodRegistry

registry = MethodRegistry()
