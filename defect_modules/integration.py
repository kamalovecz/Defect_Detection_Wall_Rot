"""Install HARP-Net modules into the one-way Ultralytics extension API."""

from __future__ import annotations

from defect_modules.blocks import CSPStage, RepHFE
from ultralytics.nn.extensions import (
    clear_detection_loss_factory,
    register_detection_loss_factory,
    register_model_module,
    registered_model_modules,
)

RULE_DEFAULTS = {
    "enabled": False,
    "version": "paper",
    "small_area": 1024.0,
    "gamma_small": 0.06,
    "gamma_contrast": 0.04,
    "low_contrast_std": 0.12,
    "lambda_max": 1.0,
    "schedule_iters": 12000,
    "stage0_ratio": 0.20,
    "stage1_ratio": 0.60,
    "total_epochs": 300,
    "t1_epoch": -1,
    "t2_epoch": -1,
}


def normalize_rule_config(value: dict | None) -> dict:
    value = {} if value is None else dict(value)
    unknown = sorted(set(value) - set(RULE_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown RuleLoss settings: {unknown}")
    config = {**RULE_DEFAULTS, **value}
    if not isinstance(config["enabled"], bool):
        raise ValueError("loss.rule.enabled must be true or false")
    if config["version"] not in {"v2", "paper"}:
        raise ValueError("loss.rule.version must be 'v2' or 'paper'")
    for key in ("small_area", "gamma_small", "gamma_contrast", "low_contrast_std", "lambda_max"):
        if float(config[key]) < 0:
            raise ValueError(f"loss.rule.{key} must be non-negative")
    if int(config["total_epochs"]) < 1:
        raise ValueError("loss.rule.total_epochs must be at least 1")
    return config


def install(rule_config: dict | None = None) -> dict:
    register_model_module("CSPStage", CSPStage, inject_channels=True, internal_repeat=True)
    register_model_module("RepHFE", RepHFE, inject_channels=True, internal_repeat=False)
    rule = normalize_rule_config(rule_config)
    if rule["enabled"]:
        from defect_modules.loss import RuleLoss

        register_detection_loss_factory(lambda model: RuleLoss(model, rule_config=rule))
    else:
        clear_detection_loss_factory()
    specs = registered_model_modules()
    return {
        "status": "ok",
        "modules": {
            name: {
                "class": f"{spec.cls.__module__}.{spec.cls.__name__}",
                "inject_channels": spec.inject_channels,
                "internal_repeat": spec.internal_repeat,
            }
            for name, spec in specs.items()
            if name in {"CSPStage", "RepHFE"}
        },
        "rule_loss": rule,
    }
