from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "export_pipeline" / "outputs"
ONNX_PATH = OUTPUT_DIR / "DAD030_best_target_canonical_640.onnx"
REPORT_PATH = OUTPUT_DIR / "DAD030_best_target_canonical_640_consistency_report.json"
EXPORT_REPORT = OUTPUT_DIR / "DAD030_best_target_canonical_640_export_report.json"


def main() -> int:
    export_report = json.loads(EXPORT_REPORT.read_text(encoding="utf-8")) if EXPORT_REPORT.exists() else {}
    if not ONNX_PATH.exists():
        report = {
            "status": "SKIPPED_NO_ONNX",
            "reason": export_report.get("reason", "ONNX file does not exist."),
            "export_status": export_report.get("status"),
            "onnx_checker": None,
            "onnxruntime_result": None,
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    try:
        import onnx
    except Exception as exc:
        report = {"status": "DEPENDENCY_MISSING", "dependency": "onnx", "error": repr(exc)}
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    try:
        import onnxruntime as ort
    except Exception as exc:
        report = {"status": "ONNX_CHECKER_PASSED_ORT_MISSING", "onnx_checker": "passed", "onnxruntime_error": repr(exc)}
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    report = {"status": "ORT_AVAILABLE_BUT_NUMERIC_CHECK_NOT_IMPLEMENTED_IN_CASE_C_PATH", "onnx_checker": "passed", "onnxruntime_version": ort.__version__}
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
