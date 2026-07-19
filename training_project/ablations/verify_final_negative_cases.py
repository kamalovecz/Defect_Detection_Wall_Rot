"""Verify that final acceptance rejects forged smoke and ONNX evidence."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_project.ablations.verify_final_acceptance import validate_smoke_state


def rejected_onnx_manifest(manifest: dict, directory: Path, expected_text: str) -> str:
    manifest_path = directory / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            "training_project/run_verifier_safely.py",
            "export_pipeline/verify_onnx_consistency.py",
            "--manifest",
            str(manifest_path.relative_to(ROOT)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = f"{process.stdout}\n{process.stderr}"
    if process.returncode == 0 or expected_text not in output:
        raise RuntimeError(f"Forged ONNX manifest was not clearly rejected:\n{output}")
    return next(line for line in output.splitlines() if expected_text in line)


def copy_artifact(source: Path, destination: Path, manifest: dict) -> None:
    for filename in manifest["files"].values():
        shutil.copy2(source / filename, destination / filename)


def rejected_final_without_evidence() -> str:
    process = subprocess.run(
        [
            sys.executable,
            "training_project/run_verifier_safely.py",
            "training_project/ablations/verify_final_acceptance.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = f"{process.stdout}\n{process.stderr}"
    expected = "complete final acceptance requires"
    if process.returncode == 0 or expected not in output:
        raise RuntimeError(f"Evidence-free final acceptance was not rejected:\n{output}")
    return next(line for line in output.splitlines() if expected in line)


def main() -> int:
    outputs_root = ROOT / "export_pipeline/outputs"
    source_artifact = outputs_root / "port_defect_smoke"
    source_manifest = json.loads((source_artifact / "artifact_manifest.json").read_text(encoding="utf-8"))
    missing_evidence_error = rejected_final_without_evidence()
    with tempfile.TemporaryDirectory(prefix="negative_hash_", dir=outputs_root) as temp_name:
        temp_dir = Path(temp_name)
        copy_artifact(source_artifact, temp_dir, source_manifest)
        zero_hash_manifest = deepcopy(source_manifest)
        zero_hash_manifest["sha256"] = {key: "0" * 64 for key in zero_hash_manifest["sha256"]}
        zero_hash_error = rejected_onnx_manifest(zero_hash_manifest, temp_dir, "Artifact hash mismatch")

    with tempfile.TemporaryDirectory(prefix="negative_path_", dir=outputs_root) as temp_name:
        temp_dir = Path(temp_name)
        traversal_manifest = deepcopy(source_manifest)
        traversal_manifest["files"] = {
            key: f"../port_defect_smoke/{filename}" for key, filename in traversal_manifest["files"].items()
        }
        traversal_error = rejected_onnx_manifest(traversal_manifest, temp_dir, "escapes its artifact directory")

    with tempfile.TemporaryDirectory(prefix="negative_metadata_", dir=outputs_root) as temp_name:
        temp_dir = Path(temp_name)
        copy_artifact(source_artifact, temp_dir, source_manifest)
        metadata_manifest = deepcopy(source_manifest)
        metadata_manifest["model"]["parameters"] = 1
        metadata_manifest["model"]["classes"] = {"0": "FAKE"}
        metadata_error = rejected_onnx_manifest(
            metadata_manifest, temp_dir, "model metadata differs from the tracked canonical identity"
        )

    with tempfile.TemporaryDirectory(prefix="negative_preprocess_", dir=outputs_root) as temp_name:
        temp_dir = Path(temp_name)
        copy_artifact(source_artifact, temp_dir, source_manifest)
        preprocess_manifest = deepcopy(source_manifest)
        preprocess_manifest["preprocess"]["color"] = "BGR"
        preprocess_error = rejected_onnx_manifest(
            preprocess_manifest, temp_dir, "identity mismatch at preprocess"
        )

    with tempfile.TemporaryDirectory(prefix="negative_runtime_", dir=outputs_root) as temp_name:
        temp_dir = Path(temp_name)
        copy_artifact(source_artifact, temp_dir, source_manifest)
        runtime_manifest = deepcopy(source_manifest)
        runtime_manifest["runtime"]["ultralytics"] = "FAKE"
        runtime_error = rejected_onnx_manifest(runtime_manifest, temp_dir, "identity mismatch at runtime")

    with tempfile.TemporaryDirectory(prefix="negative_substitute_", dir=outputs_root) as temp_name:
        temp_dir = Path(temp_name)
        copy_artifact(source_artifact, temp_dir, source_manifest)
        replacement = ROOT / "training_project/runs/ablation_smoke_final/final_B4_seed42/weights/last.pt"
        target = temp_dir / source_manifest["files"]["pt"]
        shutil.copy2(replacement, target)
        substitute_manifest = deepcopy(source_manifest)
        substitute_manifest["sha256"]["pt"] = hashlib.sha256(target.read_bytes()).hexdigest()
        substitute_error = rejected_onnx_manifest(
            substitute_manifest, temp_dir, "Artifact hash mismatch for pt"
        )

    runs_root = ROOT / "training_project/runs"
    real_state_path = runs_root / "ablation_smoke_final/stage6_state.json"
    real_state = json.loads(real_state_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="negative_state_", dir=runs_root) as temp_name:
        fake_commit_state = deepcopy(real_state)
        fake_commit_state["git_commit"] = "FAKE"
        fake_commit_path = Path(temp_name) / "fake_commit.json"
        fake_commit_path.write_text(json.dumps(fake_commit_state, indent=2), encoding="utf-8")
        try:
            validate_smoke_state(fake_commit_path)
        except RuntimeError as exc:
            fake_commit_error = str(exc)
        else:
            raise RuntimeError("Fake smoke evidence commit unexpectedly passed")

        fake_result_state = deepcopy(real_state)
        fake_result_state["experiments"]["A0"]["verification"]["strict_yaml_load"] = False
        fake_result_state["experiments"]["A0"]["verification"]["checkpoint_sha256"] = "FAKE"
        fake_result_path = Path(temp_name) / "fake_result.json"
        fake_result_path.write_text(json.dumps(fake_result_state, indent=2), encoding="utf-8")
        try:
            validate_smoke_state(fake_result_path)
        except RuntimeError as exc:
            fake_result_error = str(exc)
        else:
            raise RuntimeError("Fake smoke strict/hash result unexpectedly passed")

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError(f"Negative tests left the worktree dirty:\n{status}")
    print(
        json.dumps(
            {
                "status": "passed",
                "missing_evidence_error": missing_evidence_error,
                "zero_hash_error": zero_hash_error,
                "path_traversal_error": traversal_error,
                "metadata_error": metadata_error,
                "preprocess_error": preprocess_error,
                "runtime_error": runtime_error,
                "substitute_package_error": substitute_error,
                "fake_commit_error": fake_commit_error,
                "fake_strict_hash_error": fake_result_error,
                "worktree_clean": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
