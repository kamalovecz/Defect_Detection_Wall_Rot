from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TARGET_YAML = ROOT / "training_project" / "models" / "B4_A-GFPN_RepHFE_target.yaml"


def main() -> int:
    from ultralytics import YOLO

    failure = None
    try:
        YOLO(str(TARGET_YAML))
    except ValueError as exc:
        failure = str(exc)
    if failure is None or "CSPStage" not in failure or "layer index" not in failure:
        raise RuntimeError(f"Unregistered custom token did not fail clearly: {failure!r}")

    from defect_modules.integration import install

    first = install()
    second = install()
    model = YOLO(str(TARGET_YAML)).model
    counts = {"CSPStage": 0, "RepHFE": 0}
    for module in model.modules():
        if module.__class__.__name__ in counts:
            counts[module.__class__.__name__] += 1
    if counts != {"CSPStage": 4, "RepHFE": 2}:
        raise RuntimeError(f"Unexpected custom module counts: {counts}")
    legacy = sorted(name for name in sys.modules if name.startswith("ultralytics.nn.extra_modules"))
    if legacy:
        raise RuntimeError(f"Registration loaded legacy modules: {legacy}")
    print(json.dumps({
        "status": "ok",
        "unregistered_error": failure,
        "registered": sorted(first["modules"]),
        "idempotent": first["modules"] == second["modules"],
        "counts": counts,
        "legacy_modules": legacy,
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
