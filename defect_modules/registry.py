"""Explicit HARP-Net custom object registry.

The fixed B4_A-GFPN_RepHFE_target.yaml path intentionally exposes only two
YAML blocks: CSPStage and RepHFE. Pickle compatibility is tracked separately so
training/YAML construction does not pull in legacy extra_modules imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
if str(ULTRALYTICS_MAIN) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_MAIN))

IMPORT_ERRORS: dict[str, str] = {}


def _placeholder(name: str):
    class _MissingCustomObject:
        __name__ = name

        def __init__(self, *args, **kwargs):
            detail = IMPORT_ERRORS.get(
                name,
                "implementation is not available in the current import environment",
            )
            raise ImportError(f"{name} is registered as a HARP-Net placeholder: {detail}")

    _MissingCustomObject.__name__ = name
    _MissingCustomObject.__qualname__ = name
    return _MissingCustomObject


try:
    from defect_modules.blocks import CSPStage, RepDWConv, RepHFE
except Exception as exc:
    IMPORT_ERRORS["CSPStage"] = repr(exc)
    IMPORT_ERRORS["RepDWConv"] = repr(exc)
    IMPORT_ERRORS["RepHFE"] = repr(exc)
    CSPStage = _placeholder("CSPStage")
    RepDWConv = _placeholder("RepDWConv")
    RepHFE = _placeholder("RepHFE")

YAML_BLOCKS = {
    "CSPStage": CSPStage,
    "RepHFE": RepHFE,
}

LOSS_OBJECTS = {
    "RuleLoss": None,
}


def get_loss_objects(strict: bool = False):
    """Load training-only loss objects on demand."""
    try:
        from defect_modules.loss import RuleLoss
    except Exception as exc:
        IMPORT_ERRORS["RuleLoss"] = repr(exc)
        if strict:
            raise
        RuleLoss = _placeholder("RuleLoss")
    else:
        IMPORT_ERRORS.pop("RuleLoss", None)
    LOSS_OBJECTS["RuleLoss"] = RuleLoss
    return dict(LOSS_OBJECTS)

def active_registry(*, include_loss: bool = True):
    registry = dict(YAML_BLOCKS)
    if include_loss:
        registry.update(get_loss_objects(strict=False))
    return registry


def register_yaml_block(name: str, cls) -> None:
    """Register or override a YAML-visible custom block at runtime."""
    YAML_BLOCKS[name] = cls


def register_loss_object(name: str, cls) -> None:
    """Register or override a training/loss object at runtime."""
    LOSS_OBJECTS[name] = cls
