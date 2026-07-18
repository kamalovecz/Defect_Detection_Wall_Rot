"""Verify the active HARP-Net registry contains no legacy compatibility surface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    from defect_modules import registry

    if set(registry.YAML_BLOCKS) != {"CSPStage", "RepHFE"}:
        raise RuntimeError(f"Unexpected YAML blocks: {sorted(registry.YAML_BLOCKS)}")
    if any(hasattr(registry, name) for name in ("PICKLE_COMPAT_TYPES", "LEGACY_ALIASES", "LEGACY_PRUNE_MODULE")):
        raise RuntimeError("Legacy compatibility leaked into the active registry")
    if set(registry.LOSS_OBJECTS) != {"RuleLoss"}:
        raise RuntimeError(f"Unexpected loss objects: {sorted(registry.LOSS_OBJECTS)}")
    print(json.dumps({
        "status": "ok",
        "yaml_blocks": sorted(registry.YAML_BLOCKS),
        "loss_objects": sorted(registry.LOSS_OBJECTS),
        "legacy_surface": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
