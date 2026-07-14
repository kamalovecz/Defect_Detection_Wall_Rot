"""Verify the fixed HARP-Net YAML builds with only external CSPStage/RepHFE blocks."""

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

TARGET_YAML = ROOT / "training_project" / "models" / "B4_A-GFPN_RepHFE_target.yaml"
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

    patch_result = apply(verbose=True, pickle_compat=False, legacy_aliases=False, strict=True)

    from ultralytics import YOLO

    model = YOLO(str(TARGET_YAML))
    custom_modules = []
    counts = Counter()
    for module in model.model.modules():
        cls = module.__class__
        if cls.__name__ in {"CSPStage", "RepHFE"}:
            counts[cls.__name__] += 1
            custom_modules.append({
                "name": cls.__name__,
                "module": cls.__module__,
                "file": sys.modules[cls.__module__].__file__,
            })

    expected_counts = {"CSPStage": 4, "RepHFE": 2}
    if dict(counts) != expected_counts:
        raise RuntimeError(f"Unexpected custom block counts: {dict(counts)} != {expected_counts}")
    bad = [item for item in custom_modules if item["module"] != "defect_modules.blocks"]
    if bad:
        raise RuntimeError(f"Non-external custom modules found: {bad}")
    loaded_legacy = loaded_extra_modules()
    if OLD_REPHFE in sys.modules or OLD_PRUNE in sys.modules:
        raise RuntimeError(f"YAML build loaded legacy modules: {loaded_legacy}")
    if any(name.startswith(OLD_PREFIX) for name in loaded_legacy):
        raise RuntimeError(f"YAML build loaded extra_modules unexpectedly: {loaded_legacy}")

    print(json.dumps({
        "status": "ok",
        "target_yaml": str(TARGET_YAML),
        "patch_result": patch_result,
        "counts": dict(counts),
        "extra_modules_loaded": loaded_legacy,
        "custom_modules": custom_modules,
    }, indent=2))


if __name__ == "__main__":
    main()
