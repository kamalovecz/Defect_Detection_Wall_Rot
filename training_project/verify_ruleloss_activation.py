from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

TARGET_YAML = ROOT / "training_project" / "models" / "B4_A-GFPN_RepHFE_target.yaml"
DATA_YAML = ROOT / "ultralytics-main" / "dataset" / "processed" / "processed_dataset" / "dataset.yaml"
OUTPUT = ROOT / "training_project" / "weights" / "canonical" / "ruleloss_activation_report.json"


def read_dataset_yaml(path: Path):
    data = {"names": {}}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "names:":
            current = "names"
            continue
        if current == "names" and ":" in line and raw.startswith("  "):
            k, v = line.split(":", 1)
            data["names"][int(k.strip())] = v.strip()
            continue
        current = None
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def find_real_training_sample():
    data = read_dataset_yaml(DATA_YAML)
    base = Path(data["path"])
    labels_dir = base / data["train"].replace("images", "labels")
    images_dir = base / data["train"]
    for label_path in sorted(labels_dir.glob("*.txt")):
        rows = [line.strip().split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
            image_path = images_dir / f"{label_path.stem}{suffix}"
            if image_path.exists():
                return image_path, label_path, rows, data
    raise RuntimeError(f"No labelled training sample found under {labels_dir}")


def load_image_tensor(image_path: Path, imgsz: int = 640):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for verify_ruleloss_activation.py") from exc
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0)


def make_batch():
    image_path, label_path, rows, data = find_real_training_sample()
    cls, boxes = [], []
    for row in rows:
        if len(row) >= 5:
            cls.append([float(row[0])])
            boxes.append([float(x) for x in row[1:5]])
    if not boxes:
        raise RuntimeError(f"Label file has no YOLO boxes: {label_path}")
    return {
        "img": load_image_tensor(image_path),
        "batch_idx": torch.zeros((len(boxes),), dtype=torch.float32),
        "cls": torch.tensor(cls, dtype=torch.float32),
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
    }, image_path, label_path, data


def tensor_stats(t):
    if t is None:
        return None
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min().item()) if t.numel() else None,
        "max": float(t.max().item()) if t.numel() else None,
        "mean": float(t.mean().item()) if t.numel() else None,
        "numel": int(t.numel()),
    }


def main() -> int:
    from defect_modules.patch import apply

    patch_result = apply(verbose=False, pickle_compat=False, legacy_aliases=False, strict=True)
    from ultralytics import YOLO

    batch, image_path, label_path, data = make_batch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(TARGET_YAML)).model.to(device)
    if isinstance(getattr(model, "args", None), dict):
        args = dict(model.args)
        args.setdefault("box", 7.5)
        args.setdefault("cls", 0.5)
        args.setdefault("dfl", 1.5)
        model.args = SimpleNamespace(**args)
    model.train()
    criterion = model.init_criterion()

    records = []
    original_builder = criterion._build_rule_anchor_weights

    def wrapped_builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        batch_arg = kwargs.get("batch")
        target_bboxes = kwargs.get("target_bboxes")
        fg_mask = kwargs.get("fg_mask")
        lambda_rule = kwargs.get("lambda_rule")
        records.append({
            "lambda_rule": float(lambda_rule) if lambda_rule is not None else None,
            "batch_img_shape": list(batch_arg["img"].shape) if isinstance(batch_arg, dict) and "img" in batch_arg else None,
            "target_bboxes_shape": list(target_bboxes.shape) if target_bboxes is not None else None,
            "fg_mask_shape": list(fg_mask.shape) if fg_mask is not None else None,
            "fg_mask_positive": int(fg_mask.sum().item()) if fg_mask is not None else None,
            "weights": tensor_stats(result),
        })
        return result

    criterion._build_rule_anchor_weights = wrapped_builder
    batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        preds = model(batch_device["img"])
        active_loss, active_items = criterion(preds, batch_device)
        active_loss_value = float(active_loss.detach().cpu().item())
        active_items_value = [float(x) for x in active_items.detach().cpu().view(-1)]
        previous = criterion.rule_loss_enable
        criterion.rule_loss_enable = False
        base_loss, base_items = criterion(preds, batch_device)
        criterion.rule_loss_enable = previous
        base_loss_value = float(base_loss.detach().cpu().item())
        base_items_value = [float(x) for x in base_items.detach().cpu().view(-1)]

    active_records = [r for r in records if r["weights"] is not None]
    report = {
        "status": "RULELOSS_ACTIVE" if active_records else "RULELOSS_NOT_ACTIVE",
        "missing_connection": None if active_records else "RULE_LOSS_ENABLE is false by default and training_project/train.py does not enable it or call set_rule_epoch().",
        "criterion_class": f"{criterion.__class__.__module__}.{criterion.__class__.__name__}",
        "criterion_file": sys.modules[criterion.__class__.__module__].__file__,
        "ruleloss_instance_type": f"{criterion.__class__.__module__}.{criterion.__class__.__name__}",
        "rule_loss_enable": bool(getattr(criterion, "rule_loss_enable", False)),
        "rule_loss_version": getattr(criterion, "rule_loss_version", None),
        "rule_updates_after_forward": int(getattr(criterion, "rule_updates", 0)),
        "rule_weight_builder_call_count": len(records),
        "rule_active_call_count": len(active_records),
        "ruleloss_input_records": records,
        "active_loss": active_loss_value,
        "active_loss_items": active_items_value,
        "base_loss_with_rule_disabled": base_loss_value,
        "base_loss_items_with_rule_disabled": base_items_value,
        "ruleloss_changed_base_loss": abs(active_loss_value - base_loss_value) > 1e-9,
        "sample_image": str(image_path),
        "sample_label": str(label_path),
        "class_names": data.get("names", {}),
        "patch_result": patch_result,
        "answers": {
            "is_subclass_or_copy_of_ultralytics_v8DetectionLoss": "copy/bridge: defect_modules.loss.v8DetectionLoss is an externalized copy-style implementation, not a subclass of ultralytics.utils.loss.v8DetectionLoss.",
            "where_ruleloss_called": "_build_rule_anchor_weights() is called from compute_loss() and compute_loss_aux(); __call__ routes to those functions.",
            "what_loss_is_modified": "It builds per-anchor weights from target area and image contrast. Those weights multiply classification raw loss and target_scores passed into BboxLoss, so classification is directly weighted and box/DFL are indirectly affected through target_scores_rule.",
            "extra_inputs_needed": "Uses batch['img'], target_bboxes, fg_mask, and schedule state from env/update or epoch settings.",
            "trainer_passes_required_info": "The trainer passes normal batch tensors including img/cls/bboxes, but current training_project/train.py does not enable RULE_LOSS_ENABLE or set rule epochs.",
        },
    }
    if active_records:
        weights = [r["weights"] for r in active_records if r["weights"]]
        report["hard_sample_weight_min"] = min(w["min"] for w in weights if w["min"] is not None)
        report["hard_sample_weight_max"] = max(w["max"] for w in weights if w["max"] is not None)
        report["hard_sample_weight_mean"] = sum(w["mean"] for w in weights if w["mean"] is not None) / len(weights)
    else:
        report["hard_sample_weight_min"] = None
        report["hard_sample_weight_max"] = None
        report["hard_sample_weight_mean"] = None

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] == "RULELOSS_NOT_ACTIVE":
        print("RULELOSS_NOT_ACTIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
