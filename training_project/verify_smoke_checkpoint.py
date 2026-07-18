from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expect-ruleloss", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = (run_dir / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    values = [float(value.strip()) for value in rows[-1].split(",")]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Smoke metrics contain NaN or Inf")
    if manifest["status"] != "completed":
        raise RuntimeError(f"Run manifest is not complete: {manifest['status']}")
    if bool(manifest["rule_loss"]["enabled"]) != args.expect_ruleloss:
        raise RuntimeError("Run manifest RuleLoss state does not match the expected mode")
    if args.expect_ruleloss:
        runtime = manifest.get("criterion_runtime") or {}
        if not str(runtime.get("class", "")).startswith("defect_modules.loss."):
            raise RuntimeError(f"RuleLoss runtime criterion is not external: {runtime}")
        if int(runtime.get("rule_updates", 0)) <= 0 or float(runtime.get("lambda_rule", 0.0)) <= 0.0:
            raise RuntimeError(f"RuleLoss was enabled but did not become active: {runtime}")

    from defect_modules.integration import install
    from ultralytics import YOLO

    install({"enabled": False})
    model = YOLO(str(checkpoint)).model.float().eval().cpu()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 128, 128))
    if output is None:
        raise RuntimeError("Reloaded checkpoint produced no output")
    legacy = sorted(name for name in sys.modules if name.startswith("ultralytics.nn.extra_modules"))
    if legacy:
        raise RuntimeError(f"Checkpoint reload used legacy modules: {legacy}")
    print(json.dumps({
        "status": "ok",
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "metrics_finite": True,
        "rule_loss_enabled": manifest["rule_loss"]["enabled"],
        "criterion_runtime": manifest.get("criterion_runtime"),
        "legacy_modules": legacy,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
