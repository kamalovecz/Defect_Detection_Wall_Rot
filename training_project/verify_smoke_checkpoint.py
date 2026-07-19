from __future__ import annotations

import argparse
import hashlib
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
        if float(runtime.get("lambda_rule", 0.0)) <= 0.0:
            raise RuntimeError(f"RuleLoss was enabled but did not become active: {runtime}")
        if manifest["rule_loss"].get("version") == "v2" and int(runtime.get("rule_updates", 0)) <= 0:
            raise RuntimeError(f"RuleLoss v2 update schedule did not advance: {runtime}")

    absolute_values = []

    def find_absolute(value):
        if isinstance(value, dict):
            for child in value.values():
                find_absolute(child)
        elif isinstance(value, list):
            for child in value:
                find_absolute(child)
        elif isinstance(value, str) and Path(value).is_absolute():
            absolute_values.append(value)

    find_absolute(manifest)
    if absolute_values:
        raise RuntimeError(f"Run manifest contains absolute paths: {absolute_values}")

    from defect_modules.integration import install
    from ultralytics import YOLO

    install(manifest["rule_loss"])
    model = YOLO(str(checkpoint)).model.float().eval().cpu()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 128, 128))
    if output is None:
        raise RuntimeError("Reloaded checkpoint produced no output")

    model_yaml = ROOT / manifest["model_yaml"]
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    checkpoint_model = checkpoint_payload.get("ema") or checkpoint_payload.get("model")
    if checkpoint_model is None:
        raise RuntimeError("Checkpoint does not contain a model or EMA state")
    fresh_model = YOLO(str(model_yaml), task="detect").model.float().cpu()
    fresh_model.load_state_dict(checkpoint_model.float().state_dict(), strict=True)
    fresh_model.eval()
    fixed_sample = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        strict_output = fresh_model(fixed_sample)
    strict_prediction = strict_output[0] if isinstance(strict_output, tuple) else strict_output
    if list(strict_prediction.shape) != [1, 9, 336] or not torch.isfinite(strict_prediction).all():
        raise RuntimeError(f"Strictly loaded YAML model inference failed: {list(strict_prediction.shape)}")

    train_config = manifest["config"]["train"]
    if isinstance(fresh_model.args, dict):
        from types import SimpleNamespace

        fresh_args = dict(fresh_model.args)
        for key in ("box", "cls", "dfl"):
            fresh_args[key] = train_config[key]
        fresh_model.args = SimpleNamespace(**fresh_args)
    empty_criterion = fresh_model.init_criterion()
    if hasattr(empty_criterion, "set_rule_epoch"):
        empty_criterion.set_rule_epoch(max(0, manifest["rule_loss"]["total_epochs"] - 1), manifest["rule_loss"]["total_epochs"])
    fresh_model.train()
    empty_image = torch.rand(1, 3, 64, 64)
    empty_batch = {
        "img": empty_image,
        "batch_idx": torch.zeros(0),
        "cls": torch.zeros((0, 1)),
        "bboxes": torch.zeros((0, 4)),
    }
    fresh_model.zero_grad(set_to_none=True)
    empty_loss, empty_items = empty_criterion(fresh_model(empty_image), empty_batch)
    if not torch.isfinite(empty_loss) or not torch.isfinite(empty_items).all():
        raise RuntimeError("Empty-label batch produced NaN or Inf")
    empty_loss.backward()
    if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in fresh_model.parameters()):
        raise RuntimeError("Empty-label batch did not produce finite gradients")

    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        reported_checkpoint = checkpoint.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        reported_checkpoint = str(checkpoint.resolve())
    legacy = sorted(name for name in sys.modules if name.startswith("ultralytics.nn.extra_modules"))
    if legacy:
        raise RuntimeError(f"Checkpoint reload used legacy modules: {legacy}")
    print(json.dumps({
        "status": "ok",
        "checkpoint": reported_checkpoint,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest.hexdigest(),
        "metrics_finite": True,
        "strict_yaml_load": True,
        "strict_prediction_shape": list(strict_prediction.shape),
        "empty_batch_loss": float(empty_loss.detach()),
        "rule_loss_enabled": manifest["rule_loss"]["enabled"],
        "criterion_runtime": manifest.get("criterion_runtime"),
        "legacy_modules": legacy,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
