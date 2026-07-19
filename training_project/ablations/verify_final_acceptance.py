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
    process = subprocess.run([sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True)
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
    state = json.loads(path.read_text(encoding="utf-8"))
    experiments = state.get("experiments", {})
    if state.get("status") != "passed" or list(experiments) != EXPECTED_IDS:
        raise RuntimeError(f"Final smoke state is incomplete: {path}")
    for experiment_id in EXPECTED_IDS:
        item = experiments[experiment_id]
        if item.get("status") != "passed" or item.get("verification", {}).get("status") != "ok":
            raise RuntimeError(f"Final smoke evidence failed for {experiment_id}")
        run_dir = ROOT / item["run_dir"]
        if not (run_dir / "run_manifest.json").is_file() or not (run_dir / "smoke_contract.json").is_file():
            raise RuntimeError(f"Final smoke contract files are missing for {experiment_id}")
    return {"status": "passed", "experiments": EXPECTED_IDS, "evidence_commit": state.get("git_commit")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--smoke-state", type=Path)
    parser.add_argument("--onnx-manifest", type=Path)
    args = parser.parse_args()

    checks = {
        "core_verify_all": run_check("core_verify_all", ["training_project/verify_all.py"]),
        "six_model_gate": run_check(
            "six_model_gate", ["training_project/ablations/verify_ablation_models.py", "--require-cuda"]
        ),
    }
    dataset = validate_real_data() if args.require_real_data else {"status": "not_requested"}
    smoke = (
        validate_smoke_state((ROOT / args.smoke_state).resolve())
        if args.smoke_state
        else {"status": "not_requested"}
    )
    if args.onnx_manifest:
        checks["pt_onnx_consistency"] = run_check(
            "pt_onnx_consistency",
            ["export_pipeline/verify_onnx_consistency.py", "--manifest", str(args.onnx_manifest)],
        )

    git_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if git_status:
        raise RuntimeError(f"Final acceptance requires a clean worktree:\n{git_status}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
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
