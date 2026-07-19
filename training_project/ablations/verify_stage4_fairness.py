from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defect_modules.integration import normalize_rule_config
from training_project.config import load_config, resolve_repo_path
from ultralytics.cfg import get_cfg

MATRIX_PATH = ROOT / "training_project/ablations/training_matrix.yaml"
MANIFEST_PATH = ROOT / "training_project/ablations/manifest.yaml"
EXPECTED_IDS = ["A0", "B1", "B2", "B3", "B4", "B5"]
ALLOWED_STRUCTURE_OVERLAY = {"extends", "model", "train"}
EXPECTED_NAMES = {0: "Rust", 1: "Cracks", 2: "Paint Wear", 3: "Scratches", 4: "Pitting"}
EXPECTED_TRAIN_RECIPE = {
    "epochs": 300,
    "batch": 4,
    "imgsz": 640,
    "device": "0",
    "workers": 0,
    "seed": 42,
    "patience": 50,
    "optimizer": "SGD",
    "lr0": 0.001,
    "lrf": 0.00001,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "nbs": 64,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cache": True,
    "amp": False,
    "exist_ok": False,
    "pretrained": False,
    "resume": False,
    "deterministic": True,
    "cos_lr": True,
    "close_mosaic": 0,
    "val": True,
    "iou": 0.7,
    "max_det": 300,
    "plots": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_matrix_contract(matrix: dict) -> None:
    if matrix.get("screening_seed") != 42 or matrix.get("formal_seeds") != [42, 43, 44]:
        raise RuntimeError(
            f"Training seed contract changed: screening={matrix.get('screening_seed')}, "
            f"formal={matrix.get('formal_seeds')}"
        )
    if matrix.get("initialization") != "scratch" or matrix.get("structure_rule_loss") is not False:
        raise RuntimeError("Training matrix initialization/loss isolation changed")
    if matrix.get("config_hash_algorithm") != "sha256(canonical JSON with sorted keys)":
        raise RuntimeError("Training config hash algorithm changed")


def validate_common_contract(common: dict, matrix: dict) -> None:
    observed_hash = semantic_sha256(common)
    if observed_hash != matrix.get("common_config_sha256"):
        raise RuntimeError(f"Common training config hash changed: {observed_hash}")
    train = common.get("train", {})
    observed_recipe = {key: train.get(key) for key in EXPECTED_TRAIN_RECIPE}
    if observed_recipe != EXPECTED_TRAIN_RECIPE:
        raise RuntimeError(f"Common training recipe changed: {observed_recipe}")
    if train.get("seed") != matrix.get("screening_seed"):
        raise RuntimeError("Common training seed does not match the screening seed")


def validate_data_contract(matrix: dict) -> dict:
    data_path = ROOT / matrix.get("data_config", "")
    if matrix.get("data_config") != "datasets/Port_Defect/data.yaml" or not data_path.is_file():
        raise RuntimeError("Repository-relative Port_Defect data descriptor is missing")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    expected_keys = {"train", "val", "test", "nc", "names"}
    if set(data) != expected_keys or data.get("nc") != 5 or data.get("names") != EXPECTED_NAMES:
        raise RuntimeError(f"Port_Defect class/data descriptor contract changed: {data}")
    for key in ("train", "val", "test"):
        if not isinstance(data[key], str) or Path(data[key]).is_absolute():
            raise RuntimeError(f"Port_Defect descriptor {key} must be relative: {data[key]}")
    expected_root = (ROOT / "datasets/Port_Defect").resolve()
    descriptor_root = data_path.parent.resolve() if "path" not in data else None
    if descriptor_root != expected_root:
        raise RuntimeError(f"Port_Defect descriptor would resolve to the wrong root: {descriptor_root}")
    observed_hash = semantic_sha256(data)
    if observed_hash != matrix.get("data_config_sha256"):
        raise RuntimeError(f"Port_Defect data descriptor hash changed: {observed_hash}")
    snapshot = matrix.get("dataset_snapshot", {})
    if snapshot.get("algorithm") != "sha256(sorted relative image path, expected label path, label-present flag)":
        raise RuntimeError("Port_Defect split fingerprint algorithm changed")
    for split in ("train", "val", "test"):
        report = snapshot.get("splits", {}).get(split, {})
        if not all(isinstance(report.get(key), int) and report[key] >= 0 for key in ("images", "labels", "missing_labels")):
            raise RuntimeError(f"Port_Defect {split} counts are invalid: {report}")
        if len(report.get("fingerprint_sha256", "")) != 64:
            raise RuntimeError(f"Port_Defect {split} fingerprint is invalid: {report}")
    if len(snapshot.get("combined_fingerprint_sha256", "")) != 64:
        raise RuntimeError("Port_Defect combined split fingerprint is invalid")
    return data


def fairness_view(config: dict) -> dict:
    result = deepcopy(config)
    result.pop("config_path", None)
    result.pop("model", None)
    result["train"].pop("name", None)
    return result


def validate_structure_fairness(configs: dict[str, dict]) -> None:
    baseline = fairness_view(configs[EXPECTED_IDS[0]])
    for experiment_id in EXPECTED_IDS:
        config = configs[experiment_id]
        if fairness_view(config) != baseline:
            raise RuntimeError(f"Structure experiment {experiment_id} changes a protected fairness field")
        if config["loss"]["rule"]["enabled"] is not False:
            raise RuntimeError(f"Structure experiment {experiment_id} must disable RuleLoss")
        if config["train"].get("pretrained") is not False:
            raise RuntimeError(f"Structure experiment {experiment_id} must initialize from scratch")
        if config["train"].get("resume") is not False:
            raise RuntimeError(f"Structure experiment {experiment_id} must not resume a prior run")
        if Path(config["model"]).suffix.lower() != ".yaml":
            raise RuntimeError(f"Structure experiment {experiment_id} must build from a YAML, not weights")
        if config["loss"]["rule"]["total_epochs"] != config["train"]["epochs"]:
            raise RuntimeError(f"Structure experiment {experiment_id} has inconsistent epoch schedules")
        normalize_rule_config(config["loss"]["rule"])


def main() -> int:
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_items = {item["id"]: item for item in manifest["experiments"]}
    dataset_contract = manifest.get("dataset_contract", {})
    if dataset_contract.get("name") != "Port_Defect" or dataset_contract.get("nc") != 5:
        raise RuntimeError(f"Manifest dataset contract is invalid: {dataset_contract}")
    structure_entries = matrix.get("structure_experiments", {})
    if list(structure_entries) != EXPECTED_IDS:
        raise RuntimeError(f"Training matrix experiment order changed: {list(structure_entries)}")
    if matrix.get("dataset") != "Port_Defect" or matrix.get("data_config") != "datasets/Port_Defect/data.yaml":
        raise RuntimeError("Training matrix dataset contract changed")
    data_contract = validate_data_contract(matrix)
    common_path = ROOT / matrix.get("common_config", "")
    if matrix.get("common_config") != "training_project/configs/ablations/common.yaml" or not common_path.is_file():
        raise RuntimeError("Training matrix common config is missing or non-portable")
    validate_matrix_contract(matrix)
    formal_seeds = matrix["formal_seeds"]
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    validate_common_contract(common, matrix)

    configs = {}
    hashes = {}
    names = set()
    for experiment_id in EXPECTED_IDS:
        entry = structure_entries[experiment_id]
        config_path = ROOT / entry["config"]
        if Path(entry["config"]).is_absolute() or not config_path.is_file():
            raise RuntimeError(f"Structure config path is not portable for {experiment_id}: {entry['config']}")
        overlay = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if set(overlay) != ALLOWED_STRUCTURE_OVERLAY or set(overlay.get("train", {})) != {"name"}:
            raise RuntimeError(f"Structure overlay {experiment_id} contains hidden training changes: {overlay}")
        config = load_config(config_path)
        if Path(config["model"]).is_absolute() or Path(config["data"]).is_absolute():
            raise RuntimeError(f"Structure config {experiment_id} contains an absolute runtime path")
        expected_model = manifest_items[experiment_id]["canonical_path"]
        if config["model"] != expected_model:
            raise RuntimeError(f"Structure config {experiment_id} points to {config['model']} instead of {expected_model}")
        model_path = resolve_repo_path(config["model"])
        observed_hash = sha256(model_path)
        if observed_hash != entry["model_sha256"]:
            raise RuntimeError(f"Structure model hash changed for {experiment_id}: {observed_hash}")
        name = config["train"]["name"]
        if name in names or name == "must_be_overridden":
            raise RuntimeError(f"Structure run name is invalid or duplicated: {name}")
        names.add(name)
        configs[experiment_id] = config
        hashes[experiment_id] = observed_hash

    validate_structure_fairness(configs)
    accepted_recipe = get_cfg(overrides=configs["A0"]["train"])
    if accepted_recipe.resume is not False or accepted_recipe.pretrained is not False:
        raise RuntimeError("Ultralytics did not preserve the declared scratch recipe")
    mutated = deepcopy(configs)
    mutated["B3"]["train"]["lr0"] *= 2
    try:
        validate_structure_fairness(mutated)
    except RuntimeError as exc:
        fairness_negative_error = str(exc)
    else:
        raise RuntimeError("A structure-specific learning-rate mutation passed the fairness contract")

    mutated_common = deepcopy(common)
    mutated_common["train"]["lr0"] *= 2
    try:
        validate_common_contract(mutated_common, matrix)
    except RuntimeError as exc:
        common_negative_error = str(exc)
    else:
        raise RuntimeError("A global common learning-rate mutation passed the fixed recipe contract")

    mutated_matrix = deepcopy(matrix)
    mutated_matrix["screening_seed"] = 41
    try:
        validate_matrix_contract(mutated_matrix)
    except RuntimeError as exc:
        seed_negative_error = str(exc)
    else:
        raise RuntimeError("A training seed mutation passed the fixed seed contract")

    mutated_resume = deepcopy(configs)
    mutated_resume["A0"]["train"]["resume"] = True
    try:
        validate_structure_fairness(mutated_resume)
    except RuntimeError as exc:
        resume_negative_error = str(exc)
    else:
        raise RuntimeError("A resumed structure experiment passed the scratch contract")

    mutated_weights = deepcopy(configs)
    mutated_weights["A0"]["model"] = "weights.pt"
    try:
        validate_structure_fairness(mutated_weights)
    except RuntimeError as exc:
        weights_negative_error = str(exc)
    else:
        raise RuntimeError("A PT-backed structure experiment passed the scratch contract")

    with tempfile.TemporaryDirectory(prefix="harpnet_config_cycle_") as temp_dir:
        temp_root = Path(temp_dir)
        (temp_root / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
        (temp_root / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
        try:
            load_config(temp_root / "a.yaml")
        except ValueError as exc:
            inheritance_negative_error = str(exc)
            if "inheritance cycle" not in inheritance_negative_error:
                raise RuntimeError(f"Config cycle error is unclear: {inheritance_negative_error}") from exc
        else:
            raise RuntimeError("Config inheritance cycle unexpectedly passed")

    loss_entry = matrix.get("loss_experiment", {})
    if loss_entry != {
        "id": "L1",
        "config": "training_project/configs/ablations/L1_B5_RuleLoss.yaml",
        "reference": "B5",
        "isolated_factor": "rule_loss",
    }:
        raise RuntimeError(f"Loss experiment contract changed: {loss_entry}")
    loss_path = ROOT / loss_entry["config"]
    loss_overlay = yaml.safe_load(loss_path.read_text(encoding="utf-8"))
    if loss_overlay != {
        "extends": "B5.yaml",
        "loss": {"rule": {"enabled": True}},
        "train": {"name": "L1_B5_RuleLoss"},
    }:
        raise RuntimeError(f"L1 overlay changes more than RuleLoss and run name: {loss_overlay}")
    loss_config = load_config(loss_path)
    if loss_config["loss"]["rule"]["enabled"] is not True:
        raise RuntimeError("L1 did not enable RuleLoss")
    normalize_rule_config(loss_config["loss"]["rule"])
    normalized_loss = fairness_view(loss_config)
    normalized_loss["loss"]["rule"]["enabled"] = False
    if normalized_loss != fairness_view(configs["B5"]):
        raise RuntimeError("L1 changes fields other than RuleLoss relative to B5")

    forbidden = [
        value
        for config in [*configs.values(), loss_config]
        for value in (config["model"], config["data"], config["train"]["project"])
        if "D:\\defect_detection" in str(value) or "DAD030" in str(value)
    ]
    if forbidden:
        raise RuntimeError(f"Ablation configs contain non-portable legacy paths: {forbidden}")

    train_entrypoint = ROOT / "training_project/train.py"
    subprocess.run([sys.executable, str(train_entrypoint), "--help"], cwd=ROOT, check=True, capture_output=True)
    cli_checks = []
    for experiment_id in EXPECTED_IDS:
        process = subprocess.run(
            [sys.executable, str(train_entrypoint), "--config", structure_entries[experiment_id]["config"], "--check-config"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(process.stdout)
        if report.get("status") != "ok" or not report["resolved"]["model_exists"] or not report["resolved"]["data_exists"]:
            raise RuntimeError(f"Training CLI config check failed for {experiment_id}: {report}")
        cli_checks.append(experiment_id)

    train_source = train_entrypoint.read_text(encoding="utf-8")
    if 'model = YOLO(config["model"])' not in train_source:
        raise RuntimeError("Training entrypoint no longer constructs the model from the declared YAML")

    print(
        json.dumps(
            {
                "status": "ok",
                "structure_experiments": EXPECTED_IDS,
                "model_hashes": hashes,
                "data_descriptor_sha256": matrix["data_config_sha256"],
                "dataset_snapshot": matrix["dataset_snapshot"],
                "common_config_sha256": matrix["common_config_sha256"],
                "protected_train": fairness_view(configs["A0"])["train"],
                "ultralytics_recipe_accepted": True,
                "formal_seeds": formal_seeds,
                "loss_experiment": "L1-B5 isolates rule_loss",
                "fairness_negative_error": fairness_negative_error,
                "common_negative_error": common_negative_error,
                "seed_negative_error": seed_negative_error,
                "resume_negative_error": resume_negative_error,
                "weights_negative_error": weights_negative_error,
                "inheritance_negative_error": inheritance_negative_error,
                "cli_checks": cli_checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
