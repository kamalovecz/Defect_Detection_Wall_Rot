"""Verify the HARP-Net registry boundary for the fixed B4 YAML path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def main() -> None:
    from defect_modules import registry

    yaml_blocks = registry.YAML_BLOCKS
    if set(yaml_blocks) != {"CSPStage", "RepHFE"}:
        raise RuntimeError(f"Unexpected YAML_BLOCKS: {sorted(yaml_blocks)}")
    if yaml_blocks["CSPStage"].__module__ != "defect_modules.blocks":
        raise RuntimeError("CSPStage is not sourced from defect_modules.blocks")
    if yaml_blocks["RepHFE"].__module__ != "defect_modules.blocks":
        raise RuntimeError("RepHFE is not sourced from defect_modules.blocks")
    if "RuleLoss" not in registry.LOSS_OBJECTS:
        raise RuntimeError("RuleLoss missing from LOSS_OBJECTS")
    if "ultralytics.nn.extra_modules.rephfe" not in registry.PICKLE_COMPAT_TYPES:
        raise RuntimeError("rephfe pickle compatibility path missing")
    if "ultralytics.nn.extra_modules.prune_module" not in registry.PICKLE_COMPAT_TYPES:
        raise RuntimeError("prune_module pickle compatibility path missing")
    if any("prune" in name.lower() for name in yaml_blocks):
        raise RuntimeError("prune_module leaked into YAML_BLOCKS")

    print(json.dumps({
        "status": "ok",
        "yaml_blocks": sorted(yaml_blocks),
        "loss_objects": sorted(registry.LOSS_OBJECTS),
        "pickle_compat_paths": sorted(registry.PICKLE_COMPAT_TYPES),
        "legacy_aliases": sorted(registry.LEGACY_ALIASES),
        "prune_module_in_yaml": False,
    }, indent=2))


if __name__ == "__main__":
    main()
