from __future__ import annotations

import importlib
import inspect
import pkgutil

from app.modeling.base import BaseModelArchitecture
from app.modeling.registry import registry


def discover_architecture_classes() -> list[type[BaseModelArchitecture]]:
    architecture_classes: list[type[BaseModelArchitecture]] = []
    package_name = __name__
    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):  # type: ignore[name-defined]
        if module_info.name.startswith("_") or module_info.name == "common":
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                candidate is not BaseModelArchitecture
                and issubclass(candidate, BaseModelArchitecture)
                and not inspect.isabstract(candidate)
            ):
                architecture_classes.append(candidate)
    return architecture_classes


def register_discovered_architectures() -> None:
    for architecture_cls in discover_architecture_classes():
        try:
            registry.register(architecture_cls())
        except ValueError:
            pass


register_discovered_architectures()
