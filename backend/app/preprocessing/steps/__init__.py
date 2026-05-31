from __future__ import annotations

import importlib
import inspect
import pkgutil

from app.preprocessing.base import BasePreprocessingStep
from app.preprocessing.registry import registry


def discover_step_classes() -> list[type[BasePreprocessingStep]]:
    step_classes: list[type[BasePreprocessingStep]] = []
    package_name = __name__
    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):  # type: ignore[name-defined]
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                candidate is not BasePreprocessingStep
                and issubclass(candidate, BasePreprocessingStep)
                and not inspect.isabstract(candidate)
            ):
                step_classes.append(candidate)
    return step_classes


def register_discovered_steps() -> None:
    for step_cls in discover_step_classes():
        try:
            registry.register(step_cls())
        except ValueError:
            pass


register_discovered_steps()
