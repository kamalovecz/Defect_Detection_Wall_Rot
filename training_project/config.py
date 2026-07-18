"""Configuration loading for the HARP-Net training entrypoint."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "training_project" / "configs" / "port_defect_baseline.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a mapping: {config_path}")
    for key in ("model", "data", "train"):
        if key not in config:
            raise ValueError(f"Training config is missing required key {key!r}: {config_path}")
    if not isinstance(config["train"], dict):
        raise ValueError("Training config key 'train' must contain a mapping")
    result = deepcopy(config)
    result["config_path"] = str(config_path.resolve())
    return result


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_training_paths(config: dict) -> tuple[Path, Path, Path]:
    model = resolve_repo_path(config["model"])
    data = resolve_repo_path(config["data"])
    project = resolve_repo_path(config["train"].get("project", "training_project/runs"))
    if not model.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {model}")
    if not data.is_file():
        raise FileNotFoundError(
            f"Port_Defect data YAML does not exist: {data}. "
            "Extract the dataset under datasets/Port_Defect or pass --data <path>."
        )
    return model, data, project
