"""Configuration-driven HARP-Net training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from training_project.config import DEFAULT_CONFIG, load_config, resolve_repo_path, validate_training_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HARP-Net from a reproducible project configuration.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check-config", action="store_true", help="Validate paths and print the resolved config.")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def merged_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    for key in ("model", "data"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in (
        "epochs", "batch", "imgsz", "device", "workers", "project", "name", "seed",
        "patience", "optimizer", "lr0", "lrf", "cache", "amp", "exist_ok",
    ):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    return config


def main() -> None:
    args = parse_args()
    config = merged_config(args)

    if args.check_config:
        report = {
            "status": "ok",
            "config": config,
            "resolved": {
                "model": str(resolve_repo_path(config["model"])),
                "model_exists": resolve_repo_path(config["model"]).is_file(),
                "data": str(resolve_repo_path(config["data"])),
                "data_exists": resolve_repo_path(config["data"]).is_file(),
            },
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    model_path, data_path, project_path = validate_training_paths(config)
    config["model"] = str(model_path)
    config["data"] = str(data_path)
    config["train"]["project"] = str(project_path)

    from defect_modules.integration import install

    rule_config = dict(config["loss"]["rule"])
    rule_config["total_epochs"] = int(config["train"]["epochs"])
    integration_result = install(rule_config=rule_config)
    print(f"[train.py] registered modules: {sorted(integration_result['modules'])}")

    from ultralytics import YOLO

    train_args = dict(config["train"])
    model = YOLO(config["model"])

    def update_rule_epoch(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if hasattr(criterion, "set_rule_epoch"):
            criterion.set_rule_epoch(trainer.epoch, trainer.epochs)

    if rule_config["enabled"]:
        model.add_callback("on_train_epoch_start", update_rule_epoch)
    model.train(
        data=config["data"],
        deterministic=True,
        cos_lr=True,
        close_mosaic=0,
        plots=False,
        pretrained=True,
        **train_args,
    )


if __name__ == "__main__":
    main()
