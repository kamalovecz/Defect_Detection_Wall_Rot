"""HARP-Net training entrypoint with the external module bridge enabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_MODEL = ROOT / "training_project" / "models" / "B4_A-GFPN_RepHFE_target.yaml"
DEFAULT_DATA = ROOT / "ultralytics-main" / "dataset" / "processed" / "processed_dataset" / "dataset.yaml"
DEFAULT_PROJECT = ROOT / "training_project" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HARP-Net with defect_modules.patch applied first.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="harpnet_b4_external_next_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=1e-5)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from defect_modules.patch import apply

    patch_result = apply(verbose=True, pickle_compat=False, legacy_aliases=False, strict=True)
    print(f"[train.py] patch targets: {patch_result['targets']}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        cache=args.cache,
        seed=args.seed,
        deterministic=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        patience=args.patience,
        close_mosaic=0,
        amp=args.amp,
        plots=False,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
