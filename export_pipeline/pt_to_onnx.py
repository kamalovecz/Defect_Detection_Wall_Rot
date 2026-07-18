"""PT to ONNX entrypoint for HARP-Net.

Do not run during the MVP setup step. This script intentionally initializes the
bridge before YOLO loads the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from defect_modules.integration import install

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "export_pipeline" / "bundles" / "harpnet_b4_rephfe"
OUTPUTS = ROOT / "export_pipeline" / "outputs"


def main() -> None:
    install({"enabled": False})
    from ultralytics import YOLO

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BUNDLE / "best.pt"))
    model.export(format="onnx", simplify=True, imgsz=640, project=str(OUTPUTS), name="harpnet_b4_rephfe")


if __name__ == "__main__":
    main()
