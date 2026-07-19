"""Run the consolidated build/forward/backward gate for all ablation models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training_project.config import load_config, resolve_repo_path

EXPECTED_IDS = ["A0", "B1", "B2", "B3", "B4", "B5"]
EXPECTED = {
    "A0": {"parameters": 3_011_823, "layers": 23, "CSPStage": 0, "RepHFE": 0, "Detect_LSCSBD": 0},
    "B1": {"parameters": 2_756_005, "layers": 25, "CSPStage": 4, "RepHFE": 0, "Detect_LSCSBD": 1},
    "B2": {"parameters": 1_969_391, "layers": 25, "CSPStage": 4, "RepHFE": 2, "Detect_LSCSBD": 0},
    "B3": {"parameters": 2_303_663, "layers": 27, "CSPStage": 4, "RepHFE": 0, "Detect_LSCSBD": 0},
    "B4": {"parameters": 2_308_655, "layers": 25, "CSPStage": 4, "RepHFE": 2, "Detect_LSCSBD": 0},
    "B5": {"parameters": 3_049_701, "layers": 25, "CSPStage": 4, "RepHFE": 2, "Detect_LSCSBD": 1},
}
EXPECTED_HEAD_SOURCES = {
    "A0": "ultralytics.nn.modules.head.Detect",
    "B1": "defect_modules.sadh.Detect_LSCSBD",
    "B2": "ultralytics.nn.modules.head.Detect",
    "B3": "ultralytics.nn.modules.head.Detect",
    "B4": "ultralytics.nn.modules.head.Detect",
    "B5": "defect_modules.sadh.Detect_LSCSBD",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_gate(name: str, arguments: list[str]) -> str:
    process = subprocess.run(
        [sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True
    )
    if process.returncode:
        raise RuntimeError(
            f"Prerequisite gate {name} failed with exit code {process.returncode}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return "passed"


def native_criterion(model, train_config: dict):
    args = dict(model.args) if isinstance(model.args, dict) else vars(model.args).copy()
    for key in ("box", "cls", "dfl"):
        args[key] = train_config[key]
    model.args = SimpleNamespace(**args)
    criterion = model.init_criterion()
    criterion_name = f"{criterion.__class__.__module__}.{criterion.__class__.__name__}"
    if criterion_name != "ultralytics.utils.loss.v8DetectionLoss":
        raise RuntimeError(f"Expected native baseline criterion, got {criterion.__class__}")
    actual_weights = {key: getattr(criterion.hyp, key) for key in ("box", "cls", "dfl")}
    expected_weights = {key: train_config[key] for key in ("box", "cls", "dfl")}
    if actual_weights != expected_weights:
        raise RuntimeError(f"Criterion loss weights disagree with config: {actual_weights} != {expected_weights}")
    return criterion


def gradient_l1(module: torch.nn.Module) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        return 0.0
    return float(sum(gradient.abs().sum() for gradient in gradients))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-reference-source", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    stage3_args = ["training_project/ablations/verify_stage3_sadh.py"]
    if args.check_reference_source:
        stage3_args.append("--check-reference-source")
    if args.require_cuda:
        stage3_args.append("--require-cuda")
    archive_args = ["training_project/ablations/verify_archive.py", "--check-git"]
    if args.check_reference_source:
        archive_args.append("--check-source")
    prerequisite_gates = {
        "archive": run_gate("archive", archive_args),
        "standard_models": run_gate("standard_models", ["training_project/ablations/verify_stage2_models.py"]),
        "sadh_models": run_gate("sadh_models", stage3_args),
        "fairness_configs": run_gate("fairness_configs", ["training_project/ablations/verify_stage4_fairness.py"]),
    }

    from defect_modules.blocks import CSPStage, RepHFE
    from defect_modules.integration import install
    from defect_modules.sadh import Detect_LSCSBD
    from ultralytics import YOLO

    install({"enabled": False})
    manifest = yaml.safe_load((ROOT / "training_project/ablations/manifest.yaml").read_text(encoding="utf-8"))
    matrix = yaml.safe_load((ROOT / "training_project/ablations/training_matrix.yaml").read_text(encoding="utf-8"))
    manifest_items = {item["id"]: item for item in manifest["experiments"]}
    entries = matrix["structure_experiments"]
    if list(entries) != EXPECTED_IDS or set(manifest_items) != set(EXPECTED_IDS):
        raise RuntimeError("Consolidated ablation experiment set changed")

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the consolidated ablation gate but is unavailable")

    results = {}
    for experiment_id in EXPECTED_IDS:
        config = load_config(entries[experiment_id]["config"])
        if config["loss"]["rule"]["enabled"] is not False:
            raise RuntimeError(f"{experiment_id} unexpectedly enables RuleLoss")
        model_path = resolve_repo_path(config["model"])
        item = manifest_items[experiment_id]
        if model_path != (ROOT / item["canonical_path"]).resolve():
            raise RuntimeError(f"{experiment_id} config/manifest model paths disagree")
        if sha256(model_path) != entries[experiment_id]["model_sha256"]:
            raise RuntimeError(f"{experiment_id} model YAML hash changed")

        torch.manual_seed(42)
        model = YOLO(str(model_path), task="detect").model.cpu()
        sample = torch.zeros(1, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            output = model(sample)
        prediction = output[0] if isinstance(output, tuple) else output
        if list(prediction.shape) != [1, 9, 84] or not torch.isfinite(prediction).all():
            raise RuntimeError(f"{experiment_id} fixed-input forward failed: {list(prediction.shape)}")

        cuda_forward = False
        if torch.cuda.is_available():
            model.cuda().eval()
            with torch.no_grad():
                cuda_output = model(sample.cuda())
            cuda_prediction = cuda_output[0] if isinstance(cuda_output, tuple) else cuda_output
            if list(cuda_prediction.shape) != [1, 9, 84] or not torch.isfinite(cuda_prediction).all():
                raise RuntimeError(f"{experiment_id} CUDA forward failed")
            model.cpu()
            cuda_forward = True

        model.train()
        criterion = native_criterion(model, config["train"])
        image = torch.rand(1, 3, 64, 64)
        batch = {
            "img": image,
            "batch_idx": torch.zeros(1),
            "cls": torch.zeros((1, 1)),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        }
        model.zero_grad(set_to_none=True)
        loss, loss_items = criterion(model(image), batch)
        if not torch.isfinite(loss) or not torch.isfinite(loss_items).all():
            raise RuntimeError(f"{experiment_id} native loss is non-finite")
        loss.backward()
        backbone_gradient = gradient_l1(model.model[0])
        head_gradient = gradient_l1(model.model[-1])
        bbox_gradient = gradient_l1(model.model[-1].cv2)
        classification_gradient = gradient_l1(model.model[-1].cv3)
        if (
            backbone_gradient <= 0.0
            or head_gradient <= 0.0
            or bbox_gradient <= 0.0
            or classification_gradient <= 0.0
        ):
            raise RuntimeError(
                f"{experiment_id} backward did not reach every required path: "
                f"backbone={backbone_gradient}, head={head_gradient}, "
                f"bbox={bbox_gradient}, classification={classification_gradient}"
            )

        head_source = f"{model.model[-1].__class__.__module__}.{model.model[-1].__class__.__name__}"
        if head_source != EXPECTED_HEAD_SOURCES[experiment_id]:
            raise RuntimeError(
                f"{experiment_id} head source changed: {head_source} != {EXPECTED_HEAD_SOURCES[experiment_id]}"
            )

        observed = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "layers": len(model.model),
            "CSPStage": sum(isinstance(module, CSPStage) for module in model.modules()),
            "RepHFE": sum(isinstance(module, RepHFE) for module in model.modules()),
            "Detect_LSCSBD": sum(isinstance(module, Detect_LSCSBD) for module in model.modules()),
            "stride": [float(value) for value in model.stride],
            "prediction_shape": list(prediction.shape),
            "criterion": f"{criterion.__class__.__module__}.{criterion.__class__.__name__}",
            "criterion_weights": {key: getattr(criterion.hyp, key) for key in ("box", "cls", "dfl")},
            "loss": float(loss.detach()),
            "loss_items": [float(value) for value in loss_items.detach()],
            "backbone_gradient_l1": backbone_gradient,
            "head_gradient_l1": head_gradient,
            "bbox_gradient_l1": bbox_gradient,
            "classification_gradient_l1": classification_gradient,
            "head_source": head_source,
            "cuda_forward": cuda_forward,
        }
        for key, expected_value in EXPECTED[experiment_id].items():
            if observed[key] != expected_value:
                raise RuntimeError(f"{experiment_id} {key} changed: {observed[key]} != {expected_value}")
        if observed["stride"] != [8.0, 16.0, 32.0]:
            raise RuntimeError(f"{experiment_id} stride changed: {observed['stride']}")
        components = item["components"]
        if components["cspstage"] != (observed["CSPStage"] > 0):
            raise RuntimeError(f"{experiment_id} CSPStage evidence disagrees with manifest")
        if components["rephfe"] != (observed["RepHFE"] > 0):
            raise RuntimeError(f"{experiment_id} RepHFE evidence disagrees with manifest")
        if components["sadh"] != (observed["Detect_LSCSBD"] > 0):
            raise RuntimeError(f"{experiment_id} SADH evidence disagrees with manifest")
        results[experiment_id] = observed
        del criterion, loss, loss_items, model, output, prediction
        if cuda_forward:
            del cuda_output, cuda_prediction
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    origins = {
        "CSPStage": f"{CSPStage.__module__}.{CSPStage.__name__}",
        "RepHFE": f"{RepHFE.__module__}.{RepHFE.__name__}",
        "Detect_LSCSBD": f"{Detect_LSCSBD.__module__}.{Detect_LSCSBD.__name__}",
    }
    if not all(origin.startswith("defect_modules.") for origin in origins.values()):
        raise RuntimeError(f"Project module origin changed: {origins}")
    legacy = sorted(name for name in sys.modules if "extra_modules" in name)
    if legacy:
        raise RuntimeError(f"Legacy modules loaded during consolidated verification: {legacy}")

    print(
        json.dumps(
            {
                "status": "ok",
                "prerequisite_gates": prerequisite_gates,
                "module_origins": origins,
                "cuda_required": args.require_cuda,
                "models": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
