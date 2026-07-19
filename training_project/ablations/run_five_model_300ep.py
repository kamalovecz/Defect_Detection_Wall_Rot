"""Run the five requested 300-epoch ablations sequentially and validate each best checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = [
    ("A0", "training_project/configs/ablations/A0.yaml", "training_project/models/ablations/A0_yolov8n.yaml"),
    ("B1", "training_project/configs/ablations/B1.yaml", "training_project/models/ablations/B1_SADH.yaml"),
    ("B2", "training_project/configs/ablations/B2.yaml", "training_project/models/ablations/B2_RepHFE.yaml"),
    ("B3", "training_project/configs/ablations/B3.yaml", "training_project/models/ablations/B3_A-GFPN.yaml"),
    ("B5", "training_project/configs/ablations/B5.yaml", "training_project/models/ablations/B5_Full.yaml"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise RuntimeError(f"External Python source tree is empty: {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def external_validator_identity(val_script: Path) -> dict:
    package_root = val_script.parent / "ultralytics"
    return {
        "val_script_sha256": sha256(val_script),
        "ultralytics_python_tree_sha256": python_tree_sha256(package_root),
    }


def validate_repository_contract(dataset_source: Path) -> dict:
    from training_project.ablations.dataset_contract import fingerprint_dataset, fingerprint_dataset_content

    matrix = yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))
    manifest = yaml.safe_load((ROOT / "training_project/ablations/manifest.yaml").read_text(encoding="utf-8"))
    if manifest.get("status") != "verified":
        raise RuntimeError(f"Ablation manifest is not verified: {manifest.get('status')}")
    manifest_items = {item["id"]: item for item in manifest.get("experiments", [])}
    matrix_items = matrix.get("structure_experiments", {})
    common_path = ROOT / matrix.get("common_config", "")
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    if semantic_sha256(common) != matrix.get("common_config_sha256"):
        raise RuntimeError("Common fairness configuration hash changed")
    for experiment_id, config, model in EXPERIMENTS:
        item = manifest_items.get(experiment_id, {})
        if (
            item.get("runtime_status") != "verified"
            or item.get("blockers") != []
            or item.get("canonical_path") != model
            or matrix_items.get(experiment_id, {}).get("config") != config
            or matrix_items.get(experiment_id, {}).get("model_sha256") != sha256(ROOT / model)
        ):
            raise RuntimeError(f"Ablation manifest/model contract failed for {experiment_id}")

    logical_root = ROOT / "datasets/Port_Defect"
    for name in ("images", "labels"):
        if not (logical_root / name).samefile(dataset_source / name):
            raise RuntimeError(f"Logical dataset mapping does not point to canonical_dataset/{name}")
    observed_path = fingerprint_dataset(logical_root)
    observed_content = fingerprint_dataset_content(logical_root)
    expected = matrix["dataset_snapshot"]
    for key in ("algorithm", "splits", "combined_fingerprint_sha256"):
        if observed_path[key] != expected[key]:
            raise RuntimeError(f"Dataset path contract changed at {key}")
    for key in (
        "algorithm",
        "splits",
        "combined_content_fingerprint_sha256",
        "cross_split_duplicate_images",
        "cross_split_duplicate_fingerprint_sha256",
    ):
        if observed_content[key] != expected["content"][key]:
            raise RuntimeError(f"Dataset content contract changed at {key}")
    if expected["content"].get("formal_training_eligible") is not True:
        raise RuntimeError("The reviewed dataset is not eligible for formal training")
    if observed_content["cross_split_duplicate_images"] != 0:
        raise RuntimeError("Formal training requires zero cross-split duplicate images")
    data = yaml.safe_load((logical_root / "data.yaml").read_text(encoding="utf-8"))
    if semantic_sha256(data) != matrix["data_config_sha256"]:
        raise RuntimeError("Data descriptor semantic hash changed")
    return {"path_snapshot": observed_path, "content_snapshot": observed_content}


def validate_paper_data(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100:
        raise RuntimeError(f"Validation paper data is missing or empty: {path}")
    text = path.read_text(encoding="utf-8")
    if "all(mean)" not in text:
        raise RuntimeError(f"Validation paper data has no all(mean) summary: {path}")
    if re.search(r"(?i)(?<![A-Za-z])(nan|inf)(?![A-Za-z])", text):
        raise RuntimeError(f"Validation paper data contains NaN/Inf: {path}")


def assert_git_identity(expected_commit: str) -> None:
    observed_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if observed_commit != expected_commit:
        raise RuntimeError(f"Experiment worktree HEAD changed: {expected_commit} -> {observed_commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError(f"Experiment worktree became dirty:\n{status}")


def assert_runtime_contracts(
    dataset_source: Path,
    expected_dataset: dict,
    val_script: Path,
    expected_validator: dict,
    expected_commit: str,
) -> None:
    assert_git_identity(expected_commit)
    if validate_repository_contract(dataset_source) != expected_dataset:
        raise RuntimeError("Dataset contract changed while the experiment queue was running")
    if external_validator_identity(val_script) != expected_validator:
        raise RuntimeError("External validation implementation changed while the queue was running")


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if process.returncode:
        raise RuntimeError(f"Command failed with exit code {process.returncode}; see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="training_project/runs/ablation_300ep_b8_20260719")
    parser.add_argument("--validation-project", default="training_project/runs/ablation_300ep_b8_20260719_val")
    parser.add_argument("--val-script", default=r"D:\defect_detection\ultralytics-main\val.py")
    parser.add_argument(
        "--dataset-source",
        default=r"D:\defect_detection\repo_staging\canonical_dataset",
    )
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    project_relative = Path(args.project)
    validation_relative = Path(args.validation_project)
    if project_relative.is_absolute() or validation_relative.is_absolute():
        raise ValueError("Training and validation projects must be repository-relative")
    runs_root = (ROOT / "training_project/runs").resolve()
    project = (ROOT / project_relative).resolve()
    validation_project = (ROOT / validation_relative).resolve()
    if not project.is_relative_to(runs_root) or not validation_project.is_relative_to(runs_root):
        raise ValueError("Experiment outputs must stay under training_project/runs")
    val_script = Path(args.val_script).resolve()
    if not val_script.is_file():
        raise FileNotFoundError(f"Validation script not found: {val_script}")
    dataset_source = Path(args.dataset_source).resolve()
    if not dataset_source.is_dir():
        raise FileNotFoundError(f"Canonical dataset source not found: {dataset_source}")

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError(f"A clean experiment worktree is required:\n{status}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for _, config, model in EXPERIMENTS:
        if not (ROOT / config).is_file() or not (ROOT / model).is_file():
            raise FileNotFoundError(f"Missing experiment input: {config} or {model}")
    dataset_contract = validate_repository_contract(dataset_source)
    validator_identity = external_validator_identity(val_script)

    project.mkdir(parents=True, exist_ok=False)
    validation_project.mkdir(parents=True, exist_ok=False)
    logs = project / "logs"
    logs.mkdir()
    state_path = project / "queue_state.json"
    state = {
        "status": "running",
        "started_at": utc_now(),
        "git_commit": commit,
        "evidence_scope": "single_seed_screening",
        "contract": {
            "epochs": 300,
            "batch": 8,
            "plots": True,
            "device": args.device,
            "validation_script": str(val_script),
            "validation_split": "val",
            "validation_batch": 8,
            "dataset_source": str(dataset_source),
            "dataset": dataset_contract,
            "external_validator": validator_identity,
        },
        "order": [item[0] for item in EXPERIMENTS],
        "experiments": {},
    }
    write_state(state_path, state)

    validation_env = os.environ.copy()
    python_paths = [str(ROOT), str(ROOT / "ultralytics-main")]
    if validation_env.get("PYTHONPATH"):
        python_paths.append(validation_env["PYTHONPATH"])
    validation_env["PYTHONPATH"] = os.pathsep.join(python_paths)
    data_yaml = ROOT / "datasets/Port_Defect/data.yaml"

    try:
        for experiment_id, config, model in EXPERIMENTS:
            assert_runtime_contracts(dataset_source, dataset_contract, val_script, validator_identity, commit)
            run_name = f"{experiment_id}_seed42_e300_b8"
            run_dir = project / run_name
            validation_name = f"{run_name}_val"
            item = {
                "status": "training",
                "started_at": utc_now(),
                "config": config,
                "model": model,
                "run_dir": run_dir.relative_to(ROOT).as_posix(),
                "train_log": (logs / f"{experiment_id}_train.log").relative_to(ROOT).as_posix(),
                "validation_log": (logs / f"{experiment_id}_val.log").relative_to(ROOT).as_posix(),
            }
            state["current"] = experiment_id
            state["experiments"][experiment_id] = item
            write_state(state_path, state)

            train_command = [
                sys.executable,
                "training_project/train.py",
                "--config",
                config,
                "--model",
                model,
                "--epochs",
                "300",
                "--batch",
                "8",
                "--plots",
                "--device",
                args.device,
                "--project",
                project_relative.as_posix(),
                "--name",
                run_name,
                "--no-exist-ok",
            ]
            run_logged(train_command, logs / f"{experiment_id}_train.log")
            assert_runtime_contracts(dataset_source, dataset_contract, val_script, validator_identity, commit)
            manifest_path = run_dir / "run_manifest.json"
            best_checkpoint = run_dir / "weights/best.pt"
            if not manifest_path.is_file() or not best_checkpoint.is_file():
                raise RuntimeError(f"Training artifacts are incomplete for {experiment_id}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_config = manifest.get("config", {}).get("train", {})
            expected_train = {
                "epochs": 300,
                "batch": 8,
                "imgsz": 640,
                "device": args.device,
                "workers": 0,
                "seed": 42,
                "optimizer": "SGD",
                "amp": False,
                "plots": True,
                "pretrained": False,
                "resume": False,
            }
            if (
                manifest.get("status") != "completed"
                or manifest.get("git_commit") != commit
                or manifest.get("git_dirty") is not False
                or {key: train_config.get(key) for key in expected_train} != expected_train
                or manifest.get("rule_loss", {}).get("enabled") is not False
                or manifest.get("model_yaml") != model
                or manifest.get("model_yaml_sha256") != sha256(ROOT / model)
                or manifest.get("data_yaml") != "datasets/Port_Defect/data.yaml"
                or manifest.get("data_yaml_sha256")
                != yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))[
                    "data_config_sha256"
                ]
            ):
                raise RuntimeError(f"Training manifest contract failed for {experiment_id}")

            item["status"] = "validating"
            item["checkpoint"] = best_checkpoint.relative_to(ROOT).as_posix()
            item["checkpoint_sha256"] = sha256(best_checkpoint)
            write_state(state_path, state)
            validation_command = [
                sys.executable,
                str(val_script),
                "--model-path",
                str(best_checkpoint),
                "--data",
                str(data_yaml),
                "--split",
                "val",
                "--imgsz",
                "640",
                "--batch",
                "8",
                "--project",
                str(validation_project),
                "--name",
                validation_name,
                "--exist-ok",
            ]
            item["validation_command"] = validation_command
            assert_runtime_contracts(dataset_source, dataset_contract, val_script, validator_identity, commit)
            run_logged(validation_command, logs / f"{experiment_id}_val.log", env=validation_env)
            assert_runtime_contracts(dataset_source, dataset_contract, val_script, validator_identity, commit)
            if sha256(best_checkpoint) != item["checkpoint_sha256"]:
                raise RuntimeError(f"Validation changed the checkpoint for {experiment_id}")
            validation_dir = validation_project / validation_name
            paper_data = validation_dir / "paper_data.txt"
            validate_paper_data(paper_data)
            item["status"] = "passed"
            item["completed_at"] = utc_now()
            item["validation_dir"] = validation_dir.relative_to(ROOT).as_posix()
            item["paper_data"] = paper_data.relative_to(ROOT).as_posix()
            write_state(state_path, state)
        assert_runtime_contracts(dataset_source, dataset_contract, val_script, validator_identity, commit)
        state.pop("current", None)
        state["status"] = "passed"
        state["completed_at"] = utc_now()
        write_state(state_path, state)
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = repr(exc)
        state["failed_at"] = utc_now()
        write_state(state_path, state)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
