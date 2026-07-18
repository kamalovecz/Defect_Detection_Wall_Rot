"""Install HARP-Net modules into the one-way Ultralytics extension API."""

from __future__ import annotations

from defect_modules.blocks import CSPStage, RepHFE
from ultralytics.nn.extensions import register_model_module, registered_model_modules


def install() -> dict:
    register_model_module("CSPStage", CSPStage, inject_channels=True, internal_repeat=True)
    register_model_module("RepHFE", RepHFE, inject_channels=True, internal_repeat=False)
    specs = registered_model_modules()
    return {
        "status": "ok",
        "modules": {
            name: {
                "class": f"{spec.cls.__module__}.{spec.cls.__name__}",
                "inject_channels": spec.inject_channels,
                "internal_repeat": spec.internal_repeat,
            }
            for name, spec in specs.items()
            if name in {"CSPStage", "RepHFE"}
        },
    }
