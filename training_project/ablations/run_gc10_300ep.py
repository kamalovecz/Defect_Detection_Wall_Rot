"""Run the canonical ablation models sequentially on GC10 and validate each best checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    ("B4", "training_project/configs/ablations/B4.yaml", "training_project/models/B4_A-GFPN_RepHFE_target.yaml"),
    ("B5", "training_project/configs/ablations/B5.yaml", "training_project/models/ablations/B5_Full.yaml"),
    ("B6", "training_project/configs/ablations/B6.yaml", "training_project/models/ablations/B6_A-GFPN_SADH.yaml"),
    ("B7", "training_project/configs/ablations/B7.yaml", "training_project/models/ablations/B7_RepHFE_SADH.yaml"),
]
RESULT_COLUMNS = {
    "epoch",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
}
DEFAULT_PATIENCE = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def assert_validator_identity(contract: dict, val_script: Path) -> None:
    expected = contract["validator"]
    tree_root = Path(expected["python_tree_root"]).resolve()
    if val_script != Path(expected["script"]).resolve():
        raise RuntimeError(f"External validator path differs from the contract: {val_script}")
    if not val_script.is_file() or sha256(val_script) != expected["script_sha256"]:
        raise RuntimeError("External val.py is missing or changed")
    if not tree_root.is_dir() or python_tree_sha256(tree_root) != expected["python_tree_sha256"]:
        raise RuntimeError("External Ultralytics Python tree is missing or changed")


def validator_import_root(val_script: Path, env: dict[str, str]) -> Path:
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import ultralytics; print(Path(ultralytics.__file__).resolve().parent)",
    ]
    return Path(subprocess.check_output(command, cwd=val_script.parent, env=env, text=True).strip()).resolve()


def assert_git_identity(commit: str) -> None:
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if observed != commit:
        raise RuntimeError(f"Experiment commit changed: {commit} -> {observed}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError(f"A clean experiment worktree is required:\n{status}")


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if process.returncode:
        raise RuntimeError(f"Command failed with exit code {process.returncode}; see {log_path}")


def validate_dataset_contract() -> None:
    subprocess.run(
        [sys.executable, "training_project/ablations/verify_gc10_ablation.py", "--dataset-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )


def validate_results_csv(path: Path) -> None:
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = {column.strip() for column in (reader.fieldnames or [])}
        if columns != RESULT_COLUMNS:
            raise RuntimeError(f"Unexpected results.csv columns in {path}: {sorted(columns)}")
        rows = [{key.strip(): (value or "").strip() for key, value in row.items()} for row in reader]
    if len(rows) != 300 or [int(float(row["epoch"])) for row in rows] != list(range(1, 301)):
        raise RuntimeError(f"Training did not complete exactly 300 epochs: {path}")
    for row in rows:
        for key in RESULT_COLUMNS - {"epoch"}:
            value = row[key]
            if not value:
                raise RuntimeError(f"Empty training value in {path}: {key}")
            try:
                finite = math.isfinite(float(value))
            except ValueError as exc:
                raise RuntimeError(f"Non-numeric training value in {path}: {key}={value}") from exc
            if not finite:
                raise RuntimeError(f"Non-finite training value in {path}: {key}={value}")


def validate_paper_data(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100:
        raise RuntimeError(f"Validation paper data is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if "all(mean)" not in text or re.search(r"(?i)(?<![A-Za-z])(nan|inf)(?![A-Za-z])", text):
        raise RuntimeError(f"Validation paper data is invalid: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="training_project/runs/gc10_ablation_300ep_b8_20260722_p0")
    parser.add_argument("--validation-project", default="training_project/runs/gc10_ablation_300ep_b8_20260722_p0_val")
    parser.add_argument("--data", default=r"D:\defect_detection\ultralytics-main\dataset\processed\GC10_clean_final\data_yolo_clean.yaml")
    parser.add_argument("--val-script", default=r"D:\defect_detection\ultralytics-main\val.py")
    parser.add_argument("--device", default="0")
    parser.add_argument("--start-at", choices=[item[0] for item in EXPERIMENTS], default="A0")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    args = parser.parse_args()
    if args.patience < 0:
        raise ValueError("patience must be non-negative")
    start_index = [item[0] for item in EXPERIMENTS].index(args.start_at)
    selected_experiments = EXPERIMENTS[start_index:]

    project_relative = Path(args.project)
    validation_relative = Path(args.validation_project)
    if project_relative.is_absolute() or validation_relative.is_absolute():
        raise ValueError("Output projects must be repository-relative")
    runs_root = (ROOT / "training_project/runs").resolve()
    project = (ROOT / project_relative).resolve()
    validation_project = (ROOT / validation_relative).resolve()
    if not project.is_relative_to(runs_root) or not validation_project.is_relative_to(runs_root):
        raise ValueError("Outputs must stay under training_project/runs")
    contract = json.loads((ROOT / "training_project/ablations/gc10_dataset_contract.json").read_text(encoding="utf-8"))
    data_yaml = Path(args.data).resolve()
    val_script = Path(args.val_script).resolve()
    if data_yaml != Path(contract["descriptor"]).resolve():
        raise RuntimeError(f"Dataset descriptor differs from the contract: {data_yaml}")
    if not data_yaml.is_file() or sha256(data_yaml) != contract["descriptor_sha256"]:
        raise RuntimeError("GC10 descriptor is missing or changed")
    assert_validator_identity(contract, val_script)
    validation_env = os.environ.copy()
    python_paths = [str(ROOT), str(ROOT / "ultralytics-main")]
    if validation_env.get("PYTHONPATH"):
        python_paths.append(validation_env["PYTHONPATH"])
    validation_env["PYTHONPATH"] = os.pathsep.join(python_paths)
    observed_import_root = validator_import_root(val_script, validation_env)
    if observed_import_root != Path(contract["validator"]["python_tree_root"]).resolve():
        raise RuntimeError(f"External val.py would import an unexpected Ultralytics tree: {observed_import_root}")
    if project.exists() or validation_project.exists():
        raise FileExistsError("GC10 output project already exists; refusing to overwrite")

    assert_git_identity(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run([sys.executable, "training_project/ablations/verify_gc10_ablation.py"], cwd=ROOT, check=True)
    matrix = yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))

    project.mkdir(parents=True)
    validation_project.mkdir(parents=True)
    logs = project / "logs"
    logs.mkdir()
    state_path = project / "queue_state.json"
    state = {
        "status": "running",
        "started_at": utc_now(),
        "git_commit": commit,
        "restart_policy": contract["restart_policy"],
        "dataset_contract": contract,
        "contract": {
            "epochs": 300,
            "batch": 8,
            "imgsz": 640,
            "seed": 42,
            "plots": True,
            "pretrained": False,
            "amp": False,
            "rule_loss": False,
            "patience": args.patience,
            "start_at": args.start_at,
            "device": args.device,
            "validation_split": "val",
            "validation_batch": 8,
            "data": str(data_yaml),
            "val_script": str(val_script),
            "val_script_sha256": sha256(val_script),
            "external_ultralytics_root": contract["validator"]["python_tree_root"],
            "external_ultralytics_import_root_observed": str(observed_import_root),
            "external_ultralytics_python_tree_sha256": contract["validator"]["python_tree_sha256"],
            "restart_policy": contract["restart_policy"],
        },
        "order": [item[0] for item in selected_experiments],
        "experiments": {},
    }
    write_state(state_path, state)
    try:
        for experiment_id, config, model in selected_experiments:
            assert_git_identity(commit)
            validate_dataset_contract()
            run_name = f"{experiment_id}_seed42_e300_b8_gc10_p{args.patience}"
            run_dir = project / run_name
            validation_name = f"{run_name}_val"
            item = {
                "status": "training",
                "started_at": utc_now(),
                "config": config,
                "model": model,
                "model_sha256": sha256(ROOT / model),
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
                "--config", config,
                "--model", model,
                "--data", str(data_yaml),
                "--epochs", "300",
                "--batch", "8",
                "--imgsz", "640",
                "--seed", "42",
                "--patience", str(args.patience),
                "--plots",
                "--no-amp",
                "--device", args.device,
                "--project", project_relative.as_posix(),
                "--name", run_name,
                "--no-exist-ok",
            ]
            item["train_command"] = train_command
            run_logged(train_command, logs / f"{experiment_id}_train.log")
            assert_git_identity(commit)
            validate_dataset_contract()
            manifest_path = run_dir / "run_manifest.json"
            best = run_dir / "weights/best.pt"
            last = run_dir / "weights/last.pt"
            results = run_dir / "results.csv"
            if not all(path.is_file() for path in (manifest_path, best, last, results, run_dir / "results.png")):
                raise RuntimeError(f"Training artifacts are incomplete for {experiment_id}")
            validate_results_csv(results)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train = manifest.get("config", {}).get("train", {})
            expected_train = {
                "epochs": 300, "batch": 8, "imgsz": 640, "seed": 42,
                "plots": True, "amp": False, "pretrained": False, "resume": False,
                "patience": args.patience,
            }
            if (
                manifest.get("status") != "completed"
                or manifest.get("git_commit") != commit
                or manifest.get("git_dirty") is not False
                or manifest.get("rule_loss", {}).get("enabled") is not False
                or manifest.get("criterion_runtime", {}).get("class") != "ultralytics.utils.loss.v8DetectionLoss"
                or {key: train.get(key) for key in expected_train} != expected_train
                or manifest.get("model_yaml") != model
                or manifest.get("dataset") != contract["dataset"]
                or manifest.get("model_yaml_sha256") != matrix["structure_experiments"][experiment_id]["model_sha256"]
                or Path(manifest.get("data_yaml", "")).resolve() != data_yaml
                or manifest.get("data_yaml_sha256") != contract["descriptor_semantic_sha256"]
            ):
                raise RuntimeError(f"Training manifest contract failed for {experiment_id}")
            item["status"] = "validating"
            item["checkpoint"] = best.relative_to(ROOT).as_posix()
            item["checkpoint_sha256"] = sha256(best)
            write_state(state_path, state)
            validation_command = [
                sys.executable,
                str(val_script),
                "--model-path", str(best),
                "--data", str(data_yaml),
                "--split", "val",
                "--imgsz", "640",
                "--batch", "8",
                "--project", str(validation_project),
                "--name", validation_name,
                "--exist-ok",
            ]
            item["validation_command"] = validation_command
            assert_validator_identity(contract, val_script)
            run_logged(validation_command, logs / f"{experiment_id}_val.log", env=validation_env)
            assert_validator_identity(contract, val_script)
            assert_git_identity(commit)
            validate_dataset_contract()
            if sha256(best) != item["checkpoint_sha256"]:
                raise RuntimeError(f"Validation changed checkpoint for {experiment_id}")
            validation_dir = validation_project / validation_name
            paper_data = validation_dir / "paper_data.txt"
            validate_paper_data(paper_data)
            item.update({
                "status": "passed",
                "completed_at": utc_now(),
                "paper_data": paper_data.relative_to(ROOT).as_posix(),
                "paper_data_sha256": sha256(paper_data),
            })
            write_state(state_path, state)
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
