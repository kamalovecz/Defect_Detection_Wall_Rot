"""Verify ultralytics.nn.tasks no longer imports broad legacy extra_modules."""

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

OLD_PREFIX = "ultralytics.nn.extra_modules"
OLD_REPHFE = f"{OLD_PREFIX}.rephfe"
OLD_PRUNE = f"{OLD_PREFIX}.prune_module"


def loaded_extra_modules():
    return sorted(name for name in sys.modules if name.startswith(OLD_PREFIX))


def main() -> None:
    for name in list(sys.modules):
        if name.startswith(OLD_PREFIX) or name.startswith("ultralytics") or name.startswith("defect_modules"):
            sys.modules.pop(name, None)

    import ultralytics.nn.tasks as tasks

    before_patch = loaded_extra_modules()
    if OLD_REPHFE in sys.modules or OLD_PRUNE in sys.modules:
        raise RuntimeError(f"tasks import loaded legacy modules: {before_patch}")
    if any(name.startswith(OLD_PREFIX) for name in before_patch):
        raise RuntimeError(f"tasks import loaded extra_modules unexpectedly: {before_patch}")

    from defect_modules.patch import apply

    patch_result = apply(verbose=True, pickle_compat=False, legacy_aliases=False, strict=True)
    csp = getattr(tasks, "CSPStage", None)
    rep = getattr(tasks, "RepHFE", None)
    if csp is None or csp.__module__ != "defect_modules.blocks":
        raise RuntimeError(f"Patched CSPStage source is wrong: {csp}")
    if rep is None or rep.__module__ != "defect_modules.blocks":
        raise RuntimeError(f"Patched RepHFE source is wrong: {rep}")
    after_patch = loaded_extra_modules()
    if OLD_REPHFE in sys.modules or OLD_PRUNE in sys.modules:
        raise RuntimeError(f"non-pickle patch loaded legacy modules: {after_patch}")

    print(json.dumps({
        "status": "ok",
        "tasks_file": tasks.__file__,
        "extra_modules_after_tasks_import": before_patch,
        "extra_modules_after_non_pickle_patch": after_patch,
        "patch_result": patch_result,
        "CSPStage_module": csp.__module__,
        "RepHFE_module": rep.__module__,
    }, indent=2))


if __name__ == "__main__":
    main()
