"""Verify vendor imports and active installation stay free of legacy modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def loaded_legacy():
    return sorted(name for name in sys.modules if name.startswith("ultralytics.nn.extra_modules"))


def main() -> int:
    import ultralytics.nn.tasks as tasks

    if loaded_legacy():
        raise RuntimeError(f"Vendor task import loaded legacy modules: {loaded_legacy()}")
    if "defect_modules" in Path(tasks.__file__).read_text(encoding="utf-8"):
        raise RuntimeError("Vendor tasks.py imports project code")
    from defect_modules.integration import install
    from ultralytics.nn.extensions import registered_model_modules

    result = install({"enabled": False})
    specs = registered_model_modules()
    if specs["CSPStage"].cls.__module__ != "defect_modules.blocks":
        raise RuntimeError("CSPStage registration source is wrong")
    if specs["RepHFE"].cls.__module__ != "defect_modules.blocks":
        raise RuntimeError("RepHFE registration source is wrong")
    if specs["Detect_LSCSBD"].cls.__module__ != "defect_modules.sadh":
        raise RuntimeError("Detect_LSCSBD registration source is wrong")
    if not specs["Detect_LSCSBD"].multi_input_channels or not specs["Detect_LSCSBD"].detection_head:
        raise RuntimeError("Detect_LSCSBD extension metadata is incomplete")
    if loaded_legacy():
        raise RuntimeError(f"Project installation loaded legacy modules: {loaded_legacy()}")
    print(json.dumps({
        "status": "ok",
        "tasks_file": tasks.__file__,
        "registered": sorted(result["modules"]),
        "legacy_modules": loaded_legacy(),
        "vendor_project_import": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
