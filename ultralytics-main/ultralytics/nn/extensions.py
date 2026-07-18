"""Narrow runtime extension registry for project-defined YAML modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelModuleSpec:
    name: str
    cls: type
    inject_channels: bool = True
    internal_repeat: bool = False
    multi_input_channels: bool = False
    detection_head: bool = False


_MODEL_MODULES: dict[str, ModelModuleSpec] = {}
_MODEL_MODULES_BY_CLASS: dict[type, ModelModuleSpec] = {}
_DETECTION_LOSS_FACTORY = None


def register_model_module(
    name: str,
    cls: type,
    *,
    inject_channels: bool = True,
    internal_repeat: bool = False,
    multi_input_channels: bool = False,
    detection_head: bool = False,
) -> ModelModuleSpec:
    if inject_channels and multi_input_channels:
        raise ValueError("A model extension cannot request both single-input and multi-input channel injection")
    spec = ModelModuleSpec(name, cls, inject_channels, internal_repeat, multi_input_channels, detection_head)
    current = _MODEL_MODULES.get(name)
    if current is not None and current != spec:
        raise ValueError(f"Model module {name!r} is already registered with a different specification")
    class_current = _MODEL_MODULES_BY_CLASS.get(cls)
    if class_current is not None and class_current != spec:
        raise ValueError(f"Model module class {cls!r} is already registered under {class_current.name!r}")
    _MODEL_MODULES[name] = spec
    _MODEL_MODULES_BY_CLASS[cls] = spec
    return spec


def get_model_module(name: str) -> ModelModuleSpec | None:
    return _MODEL_MODULES.get(name)


def get_model_module_by_class(cls: type) -> ModelModuleSpec | None:
    return _MODEL_MODULES_BY_CLASS.get(cls)


def registered_model_modules() -> dict[str, ModelModuleSpec]:
    return dict(_MODEL_MODULES)


def register_detection_loss_factory(factory) -> None:
    global _DETECTION_LOSS_FACTORY
    if not callable(factory):
        raise TypeError("Detection loss factory must be callable")
    _DETECTION_LOSS_FACTORY = factory


def clear_detection_loss_factory() -> None:
    global _DETECTION_LOSS_FACTORY
    _DETECTION_LOSS_FACTORY = None


def get_detection_loss_factory():
    return _DETECTION_LOSS_FACTORY
