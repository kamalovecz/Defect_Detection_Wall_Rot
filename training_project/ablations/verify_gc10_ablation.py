from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defect_modules.integration import install
from training_project.ablations.run_gc10_300ep import DEFAULT_PATIENCE, RESULT_COLUMNS, validate_results_csv
from training_project.config import load_config
from training_project.train import dataset_name
from ultralytics.nn.tasks import DetectionModel

CONTRACT_PATH = ROOT / "training_project/ablations/gc10_dataset_contract.json"
MATRIX_PATH = ROOT / "training_project/ablations/training_matrix.yaml"
EXPERIMENTS = [
    ("A0", "training_project/configs/ablations/A0.yaml", "training_project/models/ablations/A0_yolov8n.yaml"),
    ("B1", "training_project/configs/ablations/B1.yaml", "training_project/models/ablations/B1_SADH.yaml"),
    ("B2", "training_project/configs/ablations/B2.yaml", "training_project/models/ablations/B2_RepHFE.yaml"),
    ("B3", "training_project/configs/ablations/B3.yaml", "training_project/models/ablations/B3_A-GFPN.yaml"),
    ("B4", "training_project/configs/ablations/B4.yaml", "training_project/models/B4_A-GFPN_RepHFE_target.yaml"),
    ("B5", "training_project/configs/ablations/B5.yaml", "training_project/models/ablations/B5_Full.yaml"),
    ("B6", "training_project/configs/ablations/B6.yaml", "training_project/models/ablations/B6_A-GFPN_SADH.yaml"),
    ("B7", "training_project/configs/ablations/B7.yaml", "training_project/models/ablations/B7_RepHFE_SADH.yaml"),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_validator(contract: dict) -> None:
    validator = contract["validator"]
    script = Path(validator["script"])
    tree = Path(validator["python_tree_root"])
    if not script.is_file() or sha256(script) != validator["script_sha256"]:
        raise RuntimeError("External val.py contract changed")
    if not tree.is_dir() or python_tree_sha256(tree) != validator["python_tree_sha256"]:
        raise RuntimeError("External Ultralytics Python tree contract changed")


def validate_csv_gate() -> None:
    columns = sorted(RESULT_COLUMNS)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for epoch in range(1, 301):
                row = {column: "0.1" for column in columns}
                row["epoch"] = str(epoch)
                if epoch == 300:
                    row["metrics/mAP50(B)"] = ""
                writer.writerow(row)
        try:
            validate_results_csv(path)
        except RuntimeError:
            return
        raise RuntimeError("Empty results.csv metric passed the completion gate")


def validate_dataset(contract: dict) -> dict:
    descriptor = Path(contract["descriptor"])
    if not descriptor.is_file() or sha256(descriptor) != contract["descriptor_sha256"]:
        raise RuntimeError("GC10 descriptor is missing or changed")
    data = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    if data.get("nc") != contract["nc"] or data.get("names") != contract["names"]:
        raise RuntimeError("GC10 class contract changed")
    dataset_root = Path(data.get("path", descriptor.parent))
    if not dataset_root.is_absolute():
        dataset_root = descriptor.parent / dataset_root
    combined = hashlib.sha256()
    seen = {}
    duplicate_count = 0
    observed = {}
    for split in ("train", "val", "test"):
        images_root = Path(data[split])
        if not images_root.is_absolute():
            images_root = dataset_root / images_root
        images_root = images_root.resolve()
        labels_root = images_root.parent / "labels"
        images = sorted(
            (path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.relative_to(images_root).as_posix(),
        )
        labels = sorted(labels_root.rglob("*.txt"), key=lambda path: path.relative_to(labels_root).as_posix())
        split_digest = hashlib.sha256()
        instances = 0
        classes = Counter()
        for image in images:
            relative = image.relative_to(images_root)
            label = labels_root / relative.with_suffix(".txt")
            if not label.is_file():
                raise RuntimeError(f"Missing GC10 label: {split}/{relative.as_posix()}")
            image_hash = sha256(image)
            if image_hash in seen and seen[image_hash] != split:
                duplicate_count += 1
            else:
                seen[image_hash] = split
            for line_number, row in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = row.split()
                if len(parts) != 5:
                    raise RuntimeError(f"Invalid GC10 row width: {label}:{line_number}")
                cls = int(parts[0])
                coords = [float(value) for value in parts[1:]]
                if not 0 <= cls < contract["nc"] or not all(0.0 <= value <= 1.0 for value in coords):
                    raise RuntimeError(f"Invalid GC10 row range: {label}:{line_number}")
                instances += 1
                classes[cls] += 1
            for path, root in ((image, images_root), (label, labels_root)):
                rel = path.relative_to(root).as_posix()
                payload_hash = sha256(path)
                split_digest.update(rel.encode("utf-8"))
                split_digest.update(b"\0")
                split_digest.update(payload_hash.encode("ascii"))
                split_digest.update(b"\0")
                combined.update(split.encode("ascii"))
                combined.update(b"\0")
                combined.update(rel.encode("utf-8"))
                combined.update(b"\0")
                combined.update(payload_hash.encode("ascii"))
                combined.update(b"\0")
        summary = {
            "images": len(images),
            "labels": len(labels),
            "instances": instances,
            "content_fingerprint_sha256": split_digest.hexdigest(),
        }
        if summary != contract["splits"][split]:
            raise RuntimeError(f"GC10 {split} contract changed: {summary}")
        observed[split] = {**summary, "class_instances": dict(sorted(classes.items()))}
    if combined.hexdigest() != contract["combined_content_fingerprint_sha256"]:
        raise RuntimeError("GC10 combined content fingerprint changed")
    if duplicate_count != contract["cross_split_duplicate_images"] or duplicate_count != 0:
        raise RuntimeError(f"GC10 cross-split duplicates changed: {duplicate_count}")
    return observed


def normalized_fairness(config: dict) -> dict:
    value = deepcopy(config)
    value.pop("config_path", None)
    value.pop("model", None)
    value["train"].pop("name", None)
    value["data"] = "<GC10 override>"
    return value


def prediction_tensor(output):
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Unexpected model output: {type(value)}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if DEFAULT_PATIENCE != 0:
        raise RuntimeError(f"GC10 runner must disable early stopping by default: {DEFAULT_PATIENCE}")
    dataset = validate_dataset(contract)
    validate_validator(contract)
    if dataset_name(contract["descriptor"]) != contract["dataset"]:
        raise RuntimeError("GC10 manifest dataset name derivation changed")
    port_descriptor = ROOT / "datasets/Port_Defect/data.yaml"
    if dataset_name(port_descriptor) != "Port_Defect":
        raise RuntimeError("Port_Defect manifest dataset name derivation changed")
    validate_csv_gate()
    if args.dataset_only:
        print(json.dumps({"status": "ok", "dataset": dataset}, indent=2))
        return 0

    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    matrix_items = matrix["structure_experiments"]
    configs = {}
    model_reports = {}
    install({"enabled": False})
    for experiment_id, config_relative, model_relative in EXPERIMENTS:
        if matrix_items[experiment_id]["config"] != config_relative:
            raise RuntimeError(f"Config contract changed for {experiment_id}")
        model_path = ROOT / model_relative
        if sha256(model_path) != matrix_items[experiment_id]["model_sha256"]:
            raise RuntimeError(f"Model hash changed for {experiment_id}")
        config = load_config(ROOT / config_relative)
        if config["model"] != model_relative:
            raise RuntimeError(f"Canonical model mismatch for {experiment_id}: {config['model']}")
        if config["loss"]["rule"]["enabled"] is not False:
            raise RuntimeError(f"RuleLoss must be disabled for {experiment_id}")
        configs[experiment_id] = config
        model = DetectionModel(str(model_path), nc=contract["nc"], verbose=False).eval()
        with torch.no_grad():
            prediction = prediction_tensor(model(torch.zeros(1, 3, 640, 640)))
        if list(prediction.shape) != [1, 14, 8400]:
            raise RuntimeError(f"GC10 output shape mismatch for {experiment_id}: {list(prediction.shape)}")
        model_reports[experiment_id] = {
            "model": model_relative,
            "model_sha256": sha256(model_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "prediction_shape": list(prediction.shape),
        }
    baseline = normalized_fairness(configs["A0"])
    for experiment_id, config in configs.items():
        if normalized_fairness(config) != baseline:
            raise RuntimeError(f"Training fairness mismatch for {experiment_id}")
    mutated = deepcopy(configs["B3"])
    mutated["train"]["lr0"] *= 2
    if normalized_fairness(mutated) == baseline:
        raise RuntimeError("Learning-rate mutation passed the fairness gate")
    print(json.dumps({
        "status": "ok",
        "queue": [item[0] for item in EXPERIMENTS],
        "dataset": dataset,
        "models": model_reports,
        "training_contract": {
            "epochs": 300,
            "batch": 8,
            "imgsz": 640,
            "seed": 42,
            "plots": True,
            "pretrained": False,
            "amp": False,
            "rule_loss": False,
            "patience": DEFAULT_PATIENCE,
        },
        "negative_lr_mutation_rejected": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
