"""PT to ONNX entrypoint for HARP-Net.

Do not run during the MVP setup step. This script intentionally initializes the
bridge before YOLO loads the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from defect_modules.patch import apply

BUNDLE = Path(r"D:\defect_detection\export_pipeline\bundles\harpnet_b4_rephfe")
OUTPUTS = Path(r"D:\defect_detection\export_pipeline\outputs")


def main() -> None:
    apply()
    from ultralytics import YOLO

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BUNDLE / "best.pt"))
    model.export(format="onnx", simplify=True, imgsz=640, project=str(OUTPUTS), name="harpnet_b4_rephfe")


if __name__ == "__main__":
    main()
