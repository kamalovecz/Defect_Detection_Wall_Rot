"""Run the final repository-wide acceptance gate for the ablation integration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_IDS = ["A0", "B1", "B2", "B3", "B4", "B5", "L1"]


def run_check(name: str, arguments: list[str]) -> str:
    process = subprocess.run(
        [sys.executable, "training_project/run_verifier_safely.py", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"Final acceptance check {name} failed ({process.returncode})\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return "passed"


def validate_real_data() -> dict:
    from training_project.ablations.dataset_contract import fingerprint_dataset, fingerprint_dataset_content

    matrix = yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))
    expected = matrix["dataset_snapshot"]
    dataset_root = ROOT / "datasets/Port_Defect"
    path_snapshot = fingerprint_dataset(dataset_root)
    content_snapshot = fingerprint_dataset_content(dataset_root)
    for key in ("algorithm", "splits", "combined_fingerprint_sha256"):
        if path_snapshot[key] != expected[key]:
            raise RuntimeError(f"Final dataset path snapshot changed at {key}")
    for key in (
        "algorithm",
        "splits",
        "combined_content_fingerprint_sha256",
        "cross_split_duplicate_images",
        "cross_split_duplicate_fingerprint_sha256",
    ):
        if content_snapshot[key] != expected["content"][key]:
            raise RuntimeError(f"Final dataset content snapshot changed at {key}")
    if expected["content"].get("formal_training_eligible") is not False:
        raise RuntimeError("Formal training blocker disappeared without a reviewed data-contract update")
    return {
        "status": "passed",
        "images": {split: path_snapshot["splits"][split]["images"] for split in ("train", "val", "test")},
        "cross_split_duplicate_images": content_snapshot["cross_split_duplicate_images"],
        "formal_training_status": "blocked",
    }


def validate_smoke_state(path: Path) -> dict:
    from training_project.ablations.dataset_contract import fingerprint_dataset, fingerprint_dataset_content
    from training_project.ablations.run_stage6_smoke import (
        EXPERIMENTS,
        build_static_contract,
        expected_effective_config,
        file_sha256,
        validate_smoke_contract,
        verify_run,
    )

    runs_root = (ROOT / "training_project/runs").resolve()
    path = path.resolve()
    if not path.is_relative_to(runs_root):
        raise RuntimeError(f"Smoke state must be under {runs_root}: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    experiments = state.get("experiments", {})
    if state.get("status") != "passed" or list(experiments) != EXPECTED_IDS:
        raise RuntimeError(f"Final smoke state is incomplete: {path}")
    if state.get("git_clean_at_start") is not True:
        raise RuntimeError("Smoke state does not prove a clean worktree at orchestration start")
    evidence_commit = state.get("git_commit", "")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{evidence_commit}^{{commit}}"], cwd=ROOT, capture_output=True
    ).returncode:
        raise RuntimeError(f"Smoke evidence commit is not a Git commit: {evidence_commit}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", evidence_commit, "HEAD"], cwd=ROOT, capture_output=True
    ).returncode:
        raise RuntimeError(f"Smoke evidence commit is not an ancestor of HEAD: {evidence_commit}")

    matrix = yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))
    dataset_root = ROOT / "datasets/Port_Defect"
    dataset_report = {
        "path_snapshot": fingerprint_dataset(dataset_root),
        "content_snapshot": fingerprint_dataset_content(dataset_root),
        "formal_training_eligible": False,
    }
    if state.get("dataset") != dataset_report:
        raise RuntimeError("Smoke state dataset evidence does not match the current reviewed snapshot")

    verified = []
    for experiment_id, config_path, epochs, expect_ruleloss in EXPERIMENTS:
        item = experiments[experiment_id]
        if item.get("status") != "passed" or item.get("verification", {}).get("status") != "ok":
            raise RuntimeError(f"Final smoke evidence failed for {experiment_id}")
        declared_run_dir = Path(item.get("run_dir", ""))
        run_dir = (ROOT / declared_run_dir).resolve()
        if declared_run_dir.is_absolute() or not run_dir.is_relative_to(runs_root):
            raise RuntimeError(f"Smoke run directory escapes the runs root for {experiment_id}: {declared_run_dir}")
        manifest_path = run_dir / "run_manifest.json"
        contract_path = run_dir / "smoke_contract.json"
        checkpoint = run_dir / "weights/last.pt"
        if not manifest_path.is_file() or not contract_path.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"Final smoke contract files are missing for {experiment_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        project_relative = run_dir.parent.relative_to(ROOT)
        static_contract = build_static_contract(
            experiment_id,
            config_path,
            epochs,
            expect_ruleloss,
            project_relative,
            run_dir.name,
            matrix,
            dataset_report,
            evidence_commit,
        )
        expected_contract = {
            **static_contract,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": file_sha256(checkpoint),
        }
        validate_smoke_contract(contract, expected_contract)
        effective = expected_effective_config(config_path, epochs, project_relative, run_dir.name)
        expected_manifest = {
            "model_yaml": static_contract["model_yaml"],
            "model_yaml_sha256": static_contract["model_yaml_sha256"],
            "data_yaml": static_contract["data_yaml"],
            "data_yaml_sha256": static_contract["data_yaml_sha256"],
            "effective_config_sha256": static_contract["effective_config_sha256"],
            "config": effective,
        }
        observed_manifest = {key: manifest.get(key) for key in expected_manifest}
        validate_smoke_contract(observed_manifest, expected_manifest)
        if (
            manifest.get("status") != "completed"
            or manifest.get("git_commit") != evidence_commit
            or manifest.get("git_dirty") is not False
        ):
            raise RuntimeError(f"Smoke manifest Git/status evidence failed for {experiment_id}")
        state_verification = item["verification"]
        if (
            state_verification.get("checkpoint_sha256") != expected_contract["checkpoint_sha256"]
            or state_verification.get("strict_yaml_load") is not True
            or state_verification.get("strict_prediction_shape") != [1, 9, 336]
            or state_verification.get("legacy_modules") != []
            or bool(state_verification.get("rule_loss_enabled")) != expect_ruleloss
        ):
            raise RuntimeError(f"Smoke state verification fields failed for {experiment_id}")
        runtime = state_verification.get("criterion_runtime", {})
        if expect_ruleloss and (
            runtime.get("class") != "defect_modules.loss.v8DetectionLoss"
            or float(runtime.get("lambda_rule", 0.0)) <= 0.0
        ):
            raise RuntimeError("L1 smoke state does not prove active RuleLoss")
        actual_verification = verify_run(run_dir, expect_ruleloss)
        if actual_verification.get("checkpoint_sha256") != expected_contract["checkpoint_sha256"]:
            raise RuntimeError(f"Reverified checkpoint hash failed for {experiment_id}")
        verified.append(experiment_id)
    return {"status": "passed", "experiments": verified, "evidence_commit": evidence_commit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--smoke-state", type=Path)
    parser.add_argument("--onnx-manifest", type=Path)
    args = parser.parse_args()

    missing = []
    if not args.require_real_data:
        missing.append("--require-real-data")
    if args.smoke_state is None:
        missing.append("--smoke-state")
    if args.onnx_manifest is None:
        missing.append("--onnx-manifest")
    if missing:
        parser.error(f"complete final acceptance requires: {', '.join(missing)}")

    entry_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if entry_status:
        raise RuntimeError(f"Final acceptance requires a clean worktree at entry:\n{entry_status}")
    entry_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    checks = {
        "core_verify_all": run_check("core_verify_all", ["training_project/verify_all.py"]),
        "six_model_gate": run_check(
            "six_model_gate", ["training_project/ablations/verify_ablation_models.py", "--require-cuda"]
        ),
    }
    dataset = validate_real_data()
    smoke = validate_smoke_state((ROOT / args.smoke_state).resolve())
    if args.onnx_manifest:
        declared_onnx_manifest = Path(args.onnx_manifest)
        if declared_onnx_manifest.is_absolute():
            raise ValueError("ONNX manifest argument must be repository-relative")
        resolved_onnx_manifest = (ROOT / declared_onnx_manifest).resolve()
        outputs_root = (ROOT / "export_pipeline/outputs").resolve()
        if not resolved_onnx_manifest.is_relative_to(outputs_root):
            raise ValueError("ONNX manifest must be under export_pipeline/outputs")
        checks["pt_onnx_consistency"] = run_check(
            "pt_onnx_consistency",
            ["export_pipeline/verify_onnx_consistency.py", "--manifest", str(declared_onnx_manifest)],
        )

    git_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if git_status:
        raise RuntimeError(f"Final acceptance requires a clean worktree:\n{git_status}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != entry_commit:
        raise RuntimeError(f"HEAD changed during final acceptance: {entry_commit} -> {commit}")
    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "engineering_acceptance",
                "formal_training_status": "blocked",
                "formal_training_blocker": "26 exact image-content duplicates across dataset splits",
                "git_commit": commit,
                "checks": checks,
                "dataset": dataset,
                "smoke": smoke,
                "worktree_clean": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
