"""Run and immediately verify the real-data ablation smoke matrix."""

from __future__ import annotations

import argparse
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

    matrix = yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))
    project_relative = Path(args.project)
    if project_relative.is_absolute():
        raise ValueError("Smoke project must be repository-relative")
    project = ROOT / project_relative
    project.mkdir(parents=True, exist_ok=True)
    state_path = project / "stage6_state.json"
    state = {
        "status": "running",
        "dataset": verify_dataset(matrix),
        "experiments": {},
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        for experiment_id, config, epochs, expect_ruleloss in EXPERIMENTS:
            run_name = f"{args.run_prefix}_{experiment_id}_seed42"
            run_dir = project / run_name
            manifest_path = run_dir / "run_manifest.json"
            reused = False
            if run_dir.exists():
                if not args.reuse_completed or not manifest_path.is_file():
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
            verification = verify_run(run_dir, expect_ruleloss)
            state["experiments"][experiment_id] = {
                **state["experiments"].get(experiment_id, {}),
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
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
