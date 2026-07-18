"""Inject HARP-Net custom modules into Ultralytics lookup namespaces."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types

from defect_modules.registry import (
    IMPORT_ERRORS,
    LEGACY_ALIASES,
    LEGACY_PRUNE_MODULE,
    PICKLE_COMPAT_TYPES,
    YAML_BLOCKS,
    get_loss_objects,
)
from defect_modules.loss import patch_ultralytics_loss
from defect_modules.integration import install

EXTRA_MODULES_PACKAGE = "ultralytics.nn.extra_modules"
LEGACY_REPHFE_MODULE = f"{EXTRA_MODULES_PACKAGE}.rephfe"
LEGACY_PRUNE_MODULE_NAME = f"{EXTRA_MODULES_PACKAGE}.prune_module"
LEGACY_PRUNE_BLOCK_NAME = f"{EXTRA_MODULES_PACKAGE}.block"
LEGACY_PICKLE_FILES = {
    LEGACY_PRUNE_MODULE_NAME: LEGACY_PRUNE_MODULE,
    LEGACY_PRUNE_BLOCK_NAME: LEGACY_PRUNE_MODULE.with_name("block.py"),
}


class _LegacyExtraModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        source = LEGACY_PICKLE_FILES.get(fullname)
        if source is None or not source.exists():
            return None
        return importlib.util.spec_from_file_location(fullname, source)


def _try_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception:
        return None


def _inject(module, registry) -> None:
    for name, cls in registry.items():
        setattr(module, name, cls)


def _registry(legacy_aliases: bool, strict: bool = False):
    registry = dict(YAML_BLOCKS)
    registry.update(get_loss_objects(strict=strict))
    if legacy_aliases:
        registry.update(LEGACY_ALIASES)
    return registry


def _ensure_extra_modules_parent() -> None:
    parent = sys.modules.get(EXTRA_MODULES_PACKAGE)
    if parent is None:
        parent = types.ModuleType(EXTRA_MODULES_PACKAGE)
        parent.__path__ = []
        parent.__package__ = EXTRA_MODULES_PACKAGE
        sys.modules[EXTRA_MODULES_PACKAGE] = parent
    nn_pkg = sys.modules.get("ultralytics.nn")
    if nn_pkg is not None:
        setattr(nn_pkg, "extra_modules", parent)


def _setup_legacy_rephfe_alias() -> str | None:
    compat = PICKLE_COMPAT_TYPES[LEGACY_REPHFE_MODULE]
    legacy_module = types.ModuleType(LEGACY_REPHFE_MODULE)
    legacy_module.RepDWConv = compat["RepDWConv"]
    legacy_module.RepHFE = compat["RepHFE"]
    legacy_module.__all__ = ["RepDWConv", "RepHFE"]
    legacy_module.__file__ = getattr(sys.modules.get("defect_modules.blocks"), "__file__", None)
    sys.modules[LEGACY_REPHFE_MODULE] = legacy_module
    return getattr(legacy_module, "__file__", None)


def _install_legacy_prune_finder() -> str | None:
    if not LEGACY_PRUNE_MODULE.exists():
        return None
    for finder in sys.meta_path:
        if isinstance(finder, _LegacyExtraModuleFinder):
            return f"lazy:{LEGACY_PRUNE_MODULE}"
    sys.meta_path.insert(0, _LegacyExtraModuleFinder())
    return f"lazy:{LEGACY_PRUNE_MODULE}"


def _setup_pickle_compat() -> tuple[str | None, str | None]:
    _ensure_extra_modules_parent()
    rephfe_file = _setup_legacy_rephfe_alias()
    prune_state = _install_legacy_prune_finder()
    return rephfe_file, prune_state


def apply(
    verbose: bool = True,
    pickle_compat: bool = True,
    legacy_aliases: bool = True,
    strict: bool = False,
):
    integration_result = install()
    if strict and IMPORT_ERRORS:
        raise ImportError(f"defect_modules registry has import errors: {IMPORT_ERRORS}")

    registry = _registry(legacy_aliases, strict=strict)
    targets = []

    modules_pkg = _try_import("ultralytics.nn.modules")
    if modules_pkg is not None:
        _inject(modules_pkg, registry)
        targets.append("ultralytics.nn.modules")
    elif strict:
        raise ImportError("Unable to import ultralytics.nn.modules for HARP-Net patching")

    legacy_rephfe_module = None
    legacy_prune_module = None
    if pickle_compat:
        legacy_rephfe_module, legacy_prune_module = _setup_pickle_compat()

    tasks_pkg = _try_import("ultralytics.nn.tasks")
    if tasks_pkg is not None:
        _inject(tasks_pkg, registry)
        targets.append("ultralytics.nn.tasks")
    elif strict:
        raise ImportError("Unable to import ultralytics.nn.tasks for HARP-Net patching")

    loss_targets = patch_ultralytics_loss(verbose=False)

    result = {
        "registered": sorted(registry.keys()),
        "yaml_blocks": sorted(YAML_BLOCKS.keys()),
        "loss_objects": sorted(get_loss_objects(strict=False).keys()),
        "legacy_aliases_enabled": legacy_aliases,
        "pickle_compat_enabled": pickle_compat,
        "targets": targets,
        "import_errors": dict(IMPORT_ERRORS),
        "loss_targets": loss_targets,
        "legacy_rephfe_module": legacy_rephfe_module,
        "legacy_prune_module": legacy_prune_module,
        "integration": integration_result,
    }
    if verbose:
        print(
            f"[defect_modules.patch] registered={len(result['registered'])} "
            f"targets={','.join(result['targets']) or 'none'} "
            f"pickle_compat={pickle_compat} legacy_aliases={legacy_aliases}"
        )
        if IMPORT_ERRORS:
            print(f"[defect_modules.patch] import warnings: {IMPORT_ERRORS}")
        if result["legacy_rephfe_module"]:
            print(f"[defect_modules.patch] legacy rephfe -> {result['legacy_rephfe_module']}")
        if result["legacy_prune_module"]:
            print(f"[defect_modules.patch] legacy prune_module -> {result['legacy_prune_module']}")
    return result
