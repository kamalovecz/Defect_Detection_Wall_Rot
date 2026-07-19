"""Run and immediately verify the real-data ablation smoke matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training_project.ablations.dataset_contract import fingerprint_dataset, fingerprint_dataset_content
from training_project.config import load_config

EXPERIMENTS = [
    ("A0", "training_project/configs/ablations/A0.yaml", 1, False),
    ("B1", "training_project/configs/ablations/B1.yaml", 1, False),
    ("B2", "training_project/configs/ablations/B2.yaml", 1, False),
    ("B3", "training_project/configs/ablations/B3.yaml", 1, False),
    ("B4", "training_project/configs/ablations/B4.yaml", 1, False),
    ("B5", "training_project/configs/ablations/B5.yaml", 1, False),
    ("L1", "training_project/configs/ablations/L1_B5_RuleLoss.yaml", 2, True),
]


def parse_json_output(output: str) -> dict:
    start = output.find("{")
    if start < 0:
        raise RuntimeError(f"Verifier produced no JSON: {output}")
    return json.loads(output[start:])


def semantic_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_clean_worktree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError(f"Stage 6 requires a clean Git worktree before every run:\n{status}")


def validate_smoke_contract(observed: dict, expected: dict) -> None:
    if observed != expected:
        differing = sorted(
            key for key in set(observed) | set(expected) if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(f"Smoke artifact contract mismatch in fields: {differing}")


def expected_effective_config(config_path: str, epochs: int, project: Path, run_name: str) -> dict:
    config = load_config(config_path)
    config["config_path"] = Path(config["config_path"]).resolve().relative_to(ROOT).as_posix()
    config["train"].update(
        {
            "epochs": epochs,
            "batch": 4,
            "workers": 0,
            "project": project.as_posix(),
            "name": run_name,
            "cache": False,
        }
    )
    return config


def build_static_contract(
    experiment_id: str,
    config_path: str,
    epochs: int,
    expect_ruleloss: bool,
    project: Path,
    run_name: str,
    matrix: dict,
    dataset_report: dict,
    current_commit: str,
) -> dict:
    effective = expected_effective_config(config_path, epochs, project, run_name)
    matrix_id = "B5" if experiment_id == "L1" else experiment_id
    model_entry = matrix["structure_experiments"][matrix_id]
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "config_path": config_path,
        "model_yaml": effective["model"],
        "model_yaml_sha256": model_entry["model_sha256"],
        "data_yaml": matrix["data_config"],
        "data_yaml_sha256": matrix["data_config_sha256"],
        "dataset_combined_fingerprint_sha256": dataset_report["path_snapshot"]["combined_fingerprint_sha256"],
        "dataset_content_fingerprint_sha256": dataset_report["content_snapshot"][
            "combined_content_fingerprint_sha256"
        ],
        "dataset_cross_split_duplicate_fingerprint_sha256": dataset_report["content_snapshot"][
            "cross_split_duplicate_fingerprint_sha256"
        ],
        "formal_training_eligible": False,
        "common_config_sha256": matrix["common_config_sha256"],
        "effective_config_sha256": semantic_sha256(effective),
        "git_commit": current_commit,
        "git_clean": True,
        "run_name": run_name,
        "project": project.as_posix(),
        "smoke_overrides": {
            "epochs": epochs,
            "batch": 4,
            "workers": 0,
            "cache": False,
            "seed": 42,
            "pretrained": False,
            "resume": False,
            "rule_loss": expect_ruleloss,
        },
    }


def verify_dataset(matrix: dict) -> dict:
    dataset_root = ROOT / "datasets/Port_Defect"
    observed = fingerprint_dataset(dataset_root)
    expected = matrix["dataset_snapshot"]
    for key in ("algorithm", "splits", "combined_fingerprint_sha256"):
        if observed[key] != expected[key]:
            raise RuntimeError(f"Dataset path snapshot changed at {key}")
    content = fingerprint_dataset_content(dataset_root)
    expected_content = expected["content"]
    for key in (
        "algorithm",
        "splits",
        "combined_content_fingerprint_sha256",
        "cross_split_duplicate_images",
        "cross_split_duplicate_fingerprint_sha256",
    ):
        if content[key] != expected_content[key]:
            raise RuntimeError(f"Dataset content snapshot changed at {key}")
    if expected_content.get("formal_training_eligible") is not False or content["cross_split_duplicate_images"] <= 0:
        raise RuntimeError("Known cross-split leakage is no longer represented by the formal-training blocker")

    from ultralytics.data.utils import check_det_dataset

    resolved = check_det_dataset(str(dataset_root / "data.yaml"), autodownload=False)
    for split in ("train", "val", "test"):
        logical_split = dataset_root / "images" / split
        if not Path(resolved[split]).samefile(logical_split):
            raise RuntimeError(
                f"Ultralytics resolved {split} outside the repository dataset mapping: {resolved[split]}"
            )
    return {"path_snapshot": observed, "content_snapshot": content, "formal_training_eligible": False}


def verify_run(run_dir: Path, expect_ruleloss: bool) -> dict:
    checkpoint = run_dir / "weights/last.pt"
    command = [
        sys.executable,
        "training_project/run_verifier_safely.py",
        "training_project/verify_smoke_checkpoint.py",
        str(checkpoint),
        "--run-dir",
        str(run_dir),
    ]
    if expect_ruleloss:
        command.append("--expect-ruleloss")
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(
            f"Checkpoint verification failed for {run_dir.name}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return parse_json_output(process.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--project", default="training_project/runs/ablation_smoke")
    parser.add_argument("--run-prefix", default="smoke")
    args = parser.parse_args()

    assert_clean_worktree()
    matrix = yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))
    project_relative = Path(args.project)
    if project_relative.is_absolute():
        raise ValueError("Smoke project must be repository-relative")
    project = ROOT / project_relative
    project.mkdir(parents=True, exist_ok=True)
    state_path = project / "stage6_state.json"
    previous_state = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"experiments": {}}
    )
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dataset_report = verify_dataset(matrix)
    state = {
        "status": "running",
        "git_commit": current_commit,
        "git_clean_at_start": True,
        "dataset": dataset_report,
        "experiments": {},
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        for experiment_id, config, epochs, expect_ruleloss in EXPERIMENTS:
            assert_clean_worktree()
            run_name = f"{args.run_prefix}_{experiment_id}_seed42"
            run_dir = project / run_name
            manifest_path = run_dir / "run_manifest.json"
            contract_path = run_dir / "smoke_contract.json"
            static_contract = build_static_contract(
                experiment_id,
                config,
                epochs,
                expect_ruleloss,
                project_relative,
                run_name,
                matrix,
                dataset_report,
                current_commit,
            )
            reused = False
            if run_dir.exists():
                if not args.reuse_completed or not manifest_path.is_file() or not contract_path.is_file():
                    raise RuntimeError(f"Smoke run directory already exists: {run_dir}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "completed":
                    raise RuntimeError(f"Existing smoke run is incomplete: {run_dir}")
                if (
                    manifest.get("seed") != 42
                    or manifest.get("config", {}).get("train", {}).get("epochs") != epochs
                    or bool(manifest.get("rule_loss", {}).get("enabled")) != expect_ruleloss
                ):
                    raise RuntimeError(f"Existing smoke run does not match the requested contract: {run_dir}")
                reused = True
            else:
                console_path = project / f"{run_name}.console.log"
                command = [
                    sys.executable,
                    "training_project/train.py",
                    "--config",
                    config,
                    "--epochs",
                    str(epochs),
                    "--batch",
                    "4",
                    "--workers",
                    "0",
                    "--project",
                    project_relative.as_posix(),
                    "--name",
                    run_name,
                    "--no-cache",
                ]
                started = time.perf_counter()
                with console_path.open("w", encoding="utf-8") as console:
                    process = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, text=True)
                if process.returncode:
                    raise RuntimeError(f"Smoke training failed for {experiment_id}; see {console_path}")
                duration_seconds = time.perf_counter() - started
                state["experiments"].setdefault(experiment_id, {})["duration_seconds"] = duration_seconds
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("git_commit") != current_commit or manifest.get("git_dirty") is not False:
                raise RuntimeError(
                    f"Smoke run {experiment_id} lacks clean commit evidence: "
                    f"commit={manifest.get('git_commit')} dirty={manifest.get('git_dirty')}"
                )
            effective = expected_effective_config(config, epochs, project_relative, run_name)
            manifest_contract = {
                "model_yaml": manifest.get("model_yaml"),
                "model_yaml_sha256": manifest.get("model_yaml_sha256"),
                "data_yaml": manifest.get("data_yaml"),
                "data_yaml_sha256": manifest.get("data_yaml_sha256"),
                "effective_config_sha256": manifest.get("effective_config_sha256"),
                "config": manifest.get("config"),
            }
            expected_manifest_contract = {
                "model_yaml": static_contract["model_yaml"],
                "model_yaml_sha256": static_contract["model_yaml_sha256"],
                "data_yaml": static_contract["data_yaml"],
                "data_yaml_sha256": static_contract["data_yaml_sha256"],
                "effective_config_sha256": static_contract["effective_config_sha256"],
                "config": effective,
            }
            validate_smoke_contract(manifest_contract, expected_manifest_contract)
            full_contract = {
                **static_contract,
                "checkpoint": (run_dir / "weights/last.pt").relative_to(ROOT).as_posix(),
                "checkpoint_sha256": file_sha256(run_dir / "weights/last.pt"),
            }
            if reused:
                validate_smoke_contract(json.loads(contract_path.read_text(encoding="utf-8")), full_contract)
            else:
                contract_path.write_text(json.dumps(full_contract, indent=2), encoding="utf-8")
            verification = verify_run(run_dir, expect_ruleloss)
            if verification.get("checkpoint_sha256") != full_contract["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint verifier hash disagrees for {experiment_id}")
            previous_duration = previous_state.get("experiments", {}).get(experiment_id, {}).get("duration_seconds")
            state["experiments"][experiment_id] = {
                **state["experiments"].get(experiment_id, {}),
                **({"duration_seconds": previous_duration} if reused and previous_duration is not None else {}),
                "status": "passed",
                "epochs": epochs,
                "rule_loss": expect_ruleloss,
                "reused": reused,
                "run_dir": run_dir.relative_to(ROOT).as_posix(),
                "verification": verification,
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise

    state["status"] = "passed"
    a0_contract = build_static_contract(
        "A0", EXPERIMENTS[0][1], 1, False, project_relative, f"{args.run_prefix}_A0_seed42",
        matrix, dataset_report, current_commit,
    )
    b1_contract = build_static_contract(
        "B1", EXPERIMENTS[1][1], 1, False, project_relative, f"{args.run_prefix}_B1_seed42",
        matrix, dataset_report, current_commit,
    )
    try:
        validate_smoke_contract(a0_contract, b1_contract)
    except RuntimeError as exc:
        state["mismatch_negative_error"] = str(exc)
    else:
        raise RuntimeError("A0 artifact contract unexpectedly matched the B1 contract")
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
