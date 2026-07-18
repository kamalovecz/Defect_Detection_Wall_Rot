from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TARGET = ROOT / "training_project" / "models" / "B4_A-GFPN_RepHFE_target.yaml"
EXPECTED_PARAMETERS = 2_308_655
EXPECTED_LAYERS = 25


def main() -> int:
    from defect_modules.integration import install
    from ultralytics import YOLO

    install({"enabled": False})
    model = YOLO(str(TARGET)).model
    parameters = sum(item.numel() for item in model.parameters())
    layers = len(model.model)
    if parameters != EXPECTED_PARAMETERS or layers != EXPECTED_LAYERS:
        raise RuntimeError(
            f"Target signature changed: parameters={parameters}/{EXPECTED_PARAMETERS}, layers={layers}/{EXPECTED_LAYERS}"
        )
    print(json.dumps({"status": "ok", "parameters": parameters, "layers": layers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
