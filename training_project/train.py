"""Configuration-driven HARP-Net training entrypoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

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
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
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
        "patience", "optimizer", "lr0", "lrf", "cache", "amp", "plots", "exist_ok",
    ):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None


def portable_repo_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def portable_config(config: dict) -> dict:
    result = deepcopy(config)
    for key in ("model", "data", "config_path"):
        if result.get(key):
            result[key] = portable_repo_path(result[key])
    if result.get("train", {}).get("project"):
        result["train"]["project"] = portable_repo_path(result["train"]["project"])
    return result


def write_manifest(
    config: dict,
    rule_config: dict,
    status: str,
    error: str | None = None,
    criterion_runtime: dict | None = None,
    run_dir: Path | None = None,
) -> Path:
    data_config = yaml.safe_load(Path(config["data"]).read_text(encoding="utf-8"))
    run_dir = run_dir or Path(config["train"]["project"]) / config["train"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    stored_config = portable_config(config)
    manifest = {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "model_yaml": stored_config["model"],
        "model_yaml_sha256": sha256(Path(config["model"])),
        "data_yaml": stored_config["data"],
        "data_yaml_sha256": semantic_sha256(data_config),
        "dataset": "Port_Defect",
        "class_names": data_config.get("names", {}),
        "seed": config["train"].get("seed"),
        "imgsz": config["train"].get("imgsz"),
        "rule_loss": rule_config,
        "criterion_runtime": criterion_runtime,
        "config": stored_config,
        "effective_config_sha256": semantic_sha256(stored_config),
        "error": error,
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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
    criterion_runtime = {}

    def capture_criterion(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is None:
            return
        criterion_runtime.update({
            "class": f"{criterion.__class__.__module__}.{criterion.__class__.__name__}",
            "rule_updates": int(getattr(criterion, "rule_updates", 0)),
            "lambda_rule": float(criterion._lambda_rule_t()) if hasattr(criterion, "_lambda_rule_t") else 0.0,
        })

    def update_rule_epoch(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if hasattr(criterion, "set_rule_epoch"):
            criterion.set_rule_epoch(trainer.epoch, trainer.epochs)

    if rule_config["enabled"]:
        model.add_callback("on_train_epoch_start", update_rule_epoch)
    model.add_callback("on_train_batch_end", capture_criterion)
    model.add_callback("on_train_end", capture_criterion)
    try:
        model.train(
            data=config["data"],
            **train_args,
        )
    except Exception as exc:
        trainer_save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        failure_dir = (
            Path(trainer_save_dir)
            if trainer_save_dir
            else Path(config["train"]["project"]) / config["train"]["name"]
        )
        write_manifest(config, rule_config, "failed", repr(exc), run_dir=failure_dir)
        raise
    write_manifest(
        config,
        rule_config,
        "completed",
        criterion_runtime=criterion_runtime,
        run_dir=Path(model.trainer.save_dir),
    )


if __name__ == "__main__":
    main()
