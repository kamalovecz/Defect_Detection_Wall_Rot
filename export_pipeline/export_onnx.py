from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

KNOWN_CASE_C_SHA256 = "ba5cc233eea726226b3efced7200018f799cb702db4a7f688bd8b06212b71656"
EXPECTED_PARAMETERS = 2_308_655
EXPECTED_LAYERS = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def reject_known_case_c(checkpoint_sha256: str) -> None:
    if checkpoint_sha256.lower() == KNOWN_CASE_C_SHA256:
        raise RuntimeError("CASE_C checkpoint is not a canonical export input for the target YAML")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a topology-compatible Port_Defect checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "export_pipeline" / "outputs" / "port_defect_smoke"))
    parser.add_argument("--name", default="port_defect_smoke")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    source = Path(args.checkpoint).resolve()
    run_manifest_path = Path(args.run_manifest).resolve()
    if not source.is_file() or not run_manifest_path.is_file():
        raise FileNotFoundError("Checkpoint and completed run manifest are required")
    source_sha = sha256(source)
    reject_known_case_c(source_sha)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed":
        raise RuntimeError("Only checkpoints from a completed training run may be exported")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"{args.name}.pt"
    yaml_path = output_dir / "model.yaml"
    shutil.copy2(source, pt_path)
    shutil.copy2(Path(run_manifest["model_yaml"]), yaml_path)

    from defect_modules.integration import install
    from ultralytics import YOLO, __version__ as ultralytics_version

    install({"enabled": False})
    yolo = YOLO(str(pt_path))
    model = yolo.model
    parameters = sum(item.numel() for item in model.parameters())
    layers = len(model.model)
    if parameters != EXPECTED_PARAMETERS or layers != EXPECTED_LAYERS:
        raise RuntimeError(f"Checkpoint topology mismatch: parameters={parameters}, layers={layers}")
    legacy = sorted({module.__class__.__module__ for module in model.modules() if "extra_modules" in module.__class__.__module__})
    if legacy:
        raise RuntimeError(f"Checkpoint depends on legacy modules: {legacy}")

    exported = Path(
        yolo.export(
            format="onnx", imgsz=args.imgsz, opset=12, simplify=False, dynamic=False, device=args.device
        )
    ).resolve()
    onnx_path = output_dir / f"{args.name}.onnx"
    if exported != onnx_path:
        shutil.move(str(exported), onnx_path)

    import onnx

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    manifest = {
        "schema_version": 1,
        "status": "exported",
        "artifact_kind": "engineering-smoke",
        "files": {
            "pt": pt_path.name,
            "onnx": onnx_path.name,
            "model_yaml": yaml_path.name,
        },
        "sha256": {
            "pt": sha256(pt_path),
            "onnx": sha256(onnx_path),
            "model_yaml": sha256(yaml_path),
        },
        "model": {
            "parameters": parameters,
            "layers": layers,
            "classes": run_manifest["class_names"],
            "imgsz": args.imgsz,
            "input_name": graph.graph.input[0].name,
            "output_names": [item.name for item in graph.graph.output],
            "opset": 12,
        },
        "preprocess": {"color": "RGB", "layout": "NCHW", "dtype": "float32", "scale": "divide_by_255"},
        "runtime": {"ultralytics": ultralytics_version},
        "legacy_modules": [],
        "validation": None,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
