from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
LOCK_PATH = ROOT / "export_pipeline/canonical_artifacts.json"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--box-max-abs", type=float, default=5e-2)
    parser.add_argument("--box-mean-abs", type=float, default=1e-3)
    parser.add_argument("--score-max-abs", type=float, default=1e-3)
    parser.add_argument("--score-mean-abs", type=float, default=1e-4)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    outputs_root = (ROOT / "export_pipeline/outputs").resolve()
    if not manifest_path.is_relative_to(outputs_root):
        raise RuntimeError(f"Artifact manifest must be under {outputs_root}: {manifest_path}")
    artifact_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_document = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    artifact_id = manifest.get("artifact_id")
    lock = lock_document.get("artifacts", {}).get(artifact_id)
    if lock is None:
        raise RuntimeError(f"Artifact is not present in the tracked canonical lock: {artifact_id}")
    for key in ("artifact_kind", "source", "preprocess", "runtime", "legacy_modules"):
        if manifest.get(key) != lock.get(key):
            raise RuntimeError(f"Artifact manifest identity mismatch at {key}")

    canonical_model_yaml = (ROOT / lock["canonical_model_yaml"]).resolve()
    data_yaml = (ROOT / lock["data_yaml"]).resolve()
    for name, path, expected_hash in (
        ("canonical_model_yaml", canonical_model_yaml, lock["canonical_model_yaml_sha256"]),
        ("data_yaml", data_yaml, lock["data_yaml_sha256"]),
    ):
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError(f"Trusted {name} is missing or outside the repository: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"Tracked canonical lock mismatch for {name}: {digest} != {expected_hash}")
    source_commit = lock["source"]["git_commit"]
    import subprocess

    if subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT, capture_output=True
    ).returncode:
        raise RuntimeError(f"Artifact source commit is not present: {source_commit}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT, capture_output=True
    ).returncode:
        raise RuntimeError(f"Artifact source commit is not an ancestor of HEAD: {source_commit}")
    expected_file_keys = {"pt", "onnx", "model_yaml"}
    if set(manifest.get("files", {})) != expected_file_keys or set(manifest.get("sha256", {})) != expected_file_keys:
        raise RuntimeError("Artifact manifest file/hash set is incomplete")

    artifact_paths = {}
    for key in sorted(expected_file_keys):
        declared = Path(manifest["files"][key])
        candidate = (artifact_dir / declared).resolve()
        if declared.is_absolute() or len(declared.parts) != 1 or not candidate.is_relative_to(artifact_dir.resolve()):
            raise RuntimeError(f"Artifact file {key} escapes its artifact directory: {declared}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Artifact file is missing for {key}: {candidate}")
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        observed_hash = digest.hexdigest()
        if observed_hash != manifest["sha256"][key] or observed_hash != lock["sha256"][key]:
            raise RuntimeError(
                f"Artifact hash mismatch for {key}: {observed_hash} != {lock['sha256'][key]}"
            )
        artifact_paths[key] = candidate
    pt_path = artifact_paths["pt"]
    onnx_path = artifact_paths["onnx"]

    from defect_modules.integration import install
    from ultralytics import YOLO, __version__ as ultralytics_version
    import onnx
    import onnxruntime as ort

    install({"enabled": False})
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    model = YOLO(str(pt_path)).model.float().eval().cpu()
    yaml_model = YOLO(str(canonical_model_yaml), task="detect").model.float().eval().cpu()
    yaml_model.load_state_dict(model.state_dict(), strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    layers = len(model.model)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    classes = {str(key): value for key, value in data["names"].items()}
    model_names = {str(key): value for key, value in model.names.items()}
    if model_names != classes:
        raise RuntimeError(f"PT class names differ from the tracked data contract: {model_names} != {classes}")

    graph = onnx.load(str(onnx_path))
    default_opset = next((item.version for item in graph.opset_import if item.domain in ("", "ai.onnx")), None)
    observed_model = {
        "parameters": parameters,
        "layers": layers,
        "classes": classes,
        "imgsz": lock["model"]["imgsz"],
        "input_name": graph.graph.input[0].name,
        "output_names": [item.name for item in graph.graph.output],
        "opset": default_opset,
    }
    if observed_model != lock["model"] or manifest.get("model") != observed_model:
        raise RuntimeError("Artifact model metadata differs from the tracked canonical identity")
    observed_runtime = {"ultralytics": ultralytics_version}
    if observed_runtime != lock["runtime"] or manifest.get("runtime") != observed_runtime:
        raise RuntimeError("Artifact runtime metadata differs from the installed trusted runtime")
    sample = np.random.default_rng(42).random((1, 3, manifest["model"]["imgsz"], manifest["model"]["imgsz"]), dtype=np.float32)
    with torch.no_grad():
        pt_output = model(torch.from_numpy(sample))
    if isinstance(pt_output, (tuple, list)):
        pt_output = pt_output[0]
    pt_array = pt_output.detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_array = session.run(None, {session.get_inputs()[0].name: sample})[0]
    if pt_array.shape != ort_array.shape:
        raise RuntimeError(f"PT/ONNX output shape mismatch: {pt_array.shape} != {ort_array.shape}")
    delta = np.abs(pt_array - ort_array)
    box_delta = delta[:, :4]
    score_delta = delta[:, 4:]
    metrics = {
        "box_max_abs": float(box_delta.max()),
        "box_mean_abs": float(box_delta.mean()),
        "score_max_abs": float(score_delta.max()),
        "score_mean_abs": float(score_delta.mean()),
    }
    thresholds = {
        "box_max_abs": args.box_max_abs,
        "box_mean_abs": args.box_mean_abs,
        "score_max_abs": args.score_max_abs,
        "score_mean_abs": args.score_mean_abs,
    }
    failed = {name: value for name, value in metrics.items() if value > thresholds[name]}
    if failed:
        raise RuntimeError(f"PT/ONNX numeric mismatch: metrics={metrics}, thresholds={thresholds}")

    def contains_absolute(value) -> bool:
        if isinstance(value, dict):
            return any(contains_absolute(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_absolute(item) for item in value)
        return isinstance(value, str) and Path(value).is_absolute()

    if contains_absolute(manifest):
        raise RuntimeError("Artifact manifest contains an absolute path")
    validation = {
        "onnx_checker": "passed",
        "provider": "CPUExecutionProvider",
        "output_shape": list(pt_array.shape),
        **metrics,
        "thresholds": thresholds,
    }
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
