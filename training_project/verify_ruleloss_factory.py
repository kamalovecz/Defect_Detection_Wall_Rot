from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TARGET_YAML = ROOT / "training_project" / "models" / "B4_A-GFPN_RepHFE_target.yaml"


def main() -> int:
    from defect_modules.integration import install
    from ultralytics import YOLO

    install({"enabled": False})
    model = YOLO(str(TARGET_YAML)).model.cpu()
    if isinstance(model.args, dict):
        args = dict(model.args)
        args.setdefault("box", 7.5)
        args.setdefault("cls", 0.5)
        args.setdefault("dfl", 1.5)
        model.args = SimpleNamespace(**args)
    baseline = model.init_criterion()
    if baseline.__class__.__module__ != "ultralytics.utils.loss":
        raise RuntimeError(f"Baseline criterion is not native: {baseline.__class__}")

    rule_config = {
        "enabled": True,
        "version": "paper",
        "small_area": 1000000000.0,
        "gamma_small": 2.0,
        "gamma_contrast": 0.0,
        "lambda_max": 1.0,
        "total_epochs": 2,
        "t1_epoch": 0,
        "t2_epoch": 1,
    }
    install(rule_config)
    rule = model.init_criterion()
    if rule.__class__.__module__ != "defect_modules.loss":
        raise RuntimeError(f"RuleLoss criterion has the wrong source: {rule.__class__}")
    rule.set_rule_epoch(1, 2)
    if rule._lambda_rule_t() != 1.0:
        raise RuntimeError("RuleLoss paper schedule did not reach lambda_max at t2")
    rule.set_rule_epoch(0, 2)
    if rule._lambda_rule_t() != 0.0:
        raise RuntimeError("RuleLoss paper schedule must be zero at t1")
    rule.set_rule_epoch(1, 2)

    model.train()
    records = []
    original_builder = rule._build_rule_anchor_weights

    def record_builder(*args, **kwargs):
        weights = original_builder(*args, **kwargs)
        fg_mask = kwargs["fg_mask"]
        records.append({
            "foreground": int(fg_mask.sum()),
            "weight_max": float(weights.max()) if weights is not None else None,
        })
        return weights

    rule._build_rule_anchor_weights = record_builder
    image = torch.rand(1, 3, 128, 128)
    batch = {
        "img": image,
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros((1, 1)),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
    }
    with torch.no_grad():
        preds = model(image)
        base_loss, _ = baseline(preds, batch)
        rule_loss, _ = rule(preds, batch)
    if not torch.isfinite(base_loss) or not torch.isfinite(rule_loss):
        raise RuntimeError("Criterion produced a non-finite loss")
    if not records or records[0]["foreground"] <= 0 or records[0]["weight_max"] <= 1.0:
        raise RuntimeError(f"RuleLoss did not weight a foreground anchor: {records}")
    if abs(float(base_loss - rule_loss)) <= 1e-7:
        raise RuntimeError(
            f"Enabled RuleLoss did not change the loss: base={float(base_loss)} rule={float(rule_loss)} records={records}"
        )

    invalid_rejected = False
    try:
        install({"enabled": True, "version": "invalid"})
    except ValueError:
        invalid_rejected = True
    if not invalid_rejected:
        raise RuntimeError("Invalid RuleLoss version was accepted")
    print(json.dumps({
        "status": "ok",
        "baseline": f"{baseline.__class__.__module__}.{baseline.__class__.__name__}",
        "rule": f"{rule.__class__.__module__}.{rule.__class__.__name__}",
        "base_loss": float(base_loss),
        "rule_loss": float(rule_loss),
        "weight_records": records,
        "invalid_config_rejected": invalid_rejected,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
