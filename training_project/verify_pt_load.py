"""Verify fixed PT loading with explicit pickle compatibility enabled."""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

TARGET_PT = ROOT / "training_project" / "weights" / "DAD030_best_target.pt"
OLD_PREFIX = "ultralytics.nn.extra_modules"
OLD_REPHFE = f"{OLD_PREFIX}.rephfe"
OLD_PRUNE = f"{OLD_PREFIX}.prune_module"


def loaded_extra_modules():
    return sorted(name for name in sys.modules if name.startswith(OLD_PREFIX))


def main() -> None:
    for name in list(sys.modules):
        if name.startswith(OLD_PREFIX):
            sys.modules.pop(name, None)

    from defect_modules.patch import apply

    patch_result = apply(verbose=True, pickle_compat=True, legacy_aliases=False, strict=True)
    before_pt = loaded_extra_modules()
    if OLD_PRUNE in sys.modules:
        raise RuntimeError(f"prune_module loaded before PT deserialization: {before_pt}")

    from ultralytics import YOLO

    model = YOLO(str(TARGET_PT))
    after_pt = loaded_extra_modules()
    prune_module = sys.modules.get(OLD_PRUNE)
    if prune_module is None:
        raise RuntimeError("PT load did not trigger prune_module compatibility import")
    prune_file = getattr(prune_module, "__file__", "")
    if "prune_module.py" not in str(prune_file):
        raise RuntimeError(f"Unexpected prune_module source: {prune_file}")

    counts = Counter()
    custom_sources = set()
    for module in model.model.modules():
        cls = module.__class__
        if cls.__name__ in {"CSPStage", "RepHFE"}:
            counts[cls.__name__] += 1
            custom_sources.add(cls.__module__)
    if custom_sources and custom_sources != {"defect_modules.blocks"}:
        raise RuntimeError(f"Unexpected custom block sources in PT: {sorted(custom_sources)}")

    print(json.dumps({
        "status": "ok",
        "target_pt": str(TARGET_PT),
        "patch_result": patch_result,
        "extra_modules_before_pt": before_pt,
        "extra_modules_after_pt": after_pt,
        "prune_module_file": prune_file,
        "custom_block_counts": dict(counts),
        "custom_block_sources": sorted(custom_sources),
    }, indent=2))


if __name__ == "__main__":
    main()
