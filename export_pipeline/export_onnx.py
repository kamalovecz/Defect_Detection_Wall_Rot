from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "export_pipeline" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUTPUT_DIR / "DAD030_best_target_canonical_640.onnx"
REPORT_PATH = OUTPUT_DIR / "DAD030_best_target_canonical_640_export_report.json"
CANONICAL_MANIFEST = ROOT / "training_project" / "weights" / "canonical" / "DAD030_best_target_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> int:
    manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8")) if CANONICAL_MANIFEST.exists() else {}
    if manifest.get("topology_case") == "CASE_C":
        report = {
            "status": "SKIPPED_CASE_C",
            "reason": "Canonical state_dict was not produced because source PT topology cannot be rebuilt from target YAML. Legacy PT is forbidden as final ONNX export input.",
            "onnx_path": None,
            "topology_case": manifest.get("topology_case"),
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    try:
        import torch
        import onnx
        from load_canonical_model import load_canonical_model
    except Exception as exc:
        report = {"status": "DEPENDENCY_MISSING", "error": repr(exc)}
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    model, _ = load_canonical_model(device="cpu")
    x = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    start = time.time()
    torch.onnx.export(model, x, ONNX_PATH, opset_version=12, input_names=["images"], output_names=["output0"], dynamic_axes=None, do_constant_folding=True)
    elapsed = time.time() - start
    onnx_model = onnx.load(str(ONNX_PATH))
    report = {
        "status": "ok",
        "onnx_path": str(ONNX_PATH),
        "onnx_sha256": sha256(ONNX_PATH),
        "opset": 12,
        "input_names": [i.name for i in onnx_model.graph.input],
        "output_names": [o.name for o in onnx_model.graph.output],
        "input_shapes": [[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim] for i in onnx_model.graph.input],
        "output_shapes": [[d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim] for o in onnx_model.graph.output],
        "model_size_bytes": ONNX_PATH.stat().st_size,
        "elapsed_seconds": elapsed,
        "pytorch_version": torch.__version__,
        "onnx_version": onnx.__version__,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
