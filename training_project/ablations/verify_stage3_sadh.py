from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
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

from training_project.ablations.verify_stage2_models import (
    validate_canonical_contract,
    validate_interpretation_contract,
)

MODELS = {
    "B1": ROOT / "training_project/models/ablations/B1_SADH.yaml",
    "B5": ROOT / "training_project/models/ablations/B5_Full.yaml",
}
SOURCES = {
    "B1": ROOT / "training_project/models/ablations/source/B1_SADH.yaml",
    "B5": ROOT / "training_project/models/ablations/source/B5_Full_model.yaml",
}
EXPECTED = {
    "B1": {"parameters": 2_756_005, "layers": 25, "CSPStage": 4, "RepHFE": 0, "Detect_LSCSBD": 1},
    "B5": {"parameters": 3_049_701, "layers": 25, "CSPStage": 4, "RepHFE": 2, "Detect_LSCSBD": 1},
}
HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_criterion(model):
    if isinstance(model.args, dict):
        args = dict(model.args)
        args.setdefault("box", 7.5)
        args.setdefault("cls", 0.5)
        args.setdefault("dfl", 1.5)
        model.args = SimpleNamespace(**args)
    criterion = model.init_criterion()
    if criterion.__class__.__module__ != "ultralytics.utils.loss":
        raise RuntimeError(f"SADH baseline criterion is not native: {criterion.__class__}")
    return criterion


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the decoupled SADH detection head and B1/B5 models.")
    parser.add_argument(
        "--check-reference-source",
        action="store_true",
        help="Also verify the external legacy implementation references recorded in the manifest.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require and verify a CUDA round trip for SADH runtime tensors.",
    )
    args = parser.parse_args()

    from defect_modules.blocks import CSPStage, RepHFE
    from defect_modules.integration import install
    from defect_modules.sadh import Detect_LSCSBD
    from ultralytics import YOLO
    from ultralytics.engine.exporter import configure_detection_head_for_export
    from ultralytics.nn.extensions import registered_model_modules
    from ultralytics.nn.tasks import guess_model_task, resolve_model_module

    if registered_model_modules():
        raise RuntimeError("Stage 3 verifier must start with an empty extension registry")
    try:
        resolve_model_module("Detect_LSCSBD", 24)
    except ValueError as exc:
        unregistered_error = str(exc)
        if "Detect_LSCSBD" not in unregistered_error or "layer index 24" not in unregistered_error:
            raise RuntimeError(f"Unregistered SADH error lacks context: {unregistered_error}") from exc
    else:
        raise RuntimeError("Detect_LSCSBD resolved before project integration was installed")

    first_install = install({"enabled": False})
    second_install = install({"enabled": False})
    if first_install["modules"] != second_install["modules"]:
        raise RuntimeError("Detect_LSCSBD registration is not idempotent")
    spec = registered_model_modules()["Detect_LSCSBD"]
    if spec.cls is not Detect_LSCSBD or spec.inject_channels or not spec.multi_input_channels or not spec.detection_head:
        raise RuntimeError(f"Detect_LSCSBD extension metadata is invalid: {spec}")

    manifest = yaml.safe_load((ROOT / "training_project/ablations/manifest.yaml").read_text(encoding="utf-8"))
    manifest_items = validate_interpretation_contract(manifest)
    provenance = manifest.get("implementation_provenance", {}).get("Detect_LSCSBD", {})
    if provenance.get("active_path") != "defect_modules/sadh.py" or provenance.get("migration_policy") != "minimal_detection_only_no_extra_modules":
        raise RuntimeError(f"Detect_LSCSBD implementation provenance is incomplete: {provenance}")
    references = provenance.get("references", [])
    reference_roles = {item.get("role") for item in references}
    if (
        len(references) != 2
        or reference_roles != {"authoritative_runtime_reference", "paper_cross_reference"}
        or any(not HEX64.fullmatch(item.get("sha256", "")) for item in references)
    ):
        raise RuntimeError(f"Detect_LSCSBD reference hashes are invalid: {references}")
    if args.check_reference_source:
        for reference in references:
            reference_path = Path(reference["path"])
            if not reference_path.is_file() or sha256(reference_path) != reference["sha256"]:
                raise RuntimeError(f"Detect_LSCSBD legacy reference drifted: {reference_path}")
    torch.manual_seed(42)
    sample = torch.zeros(1, 3, 64, 64)
    results = {}
    for experiment_id, model_path in MODELS.items():
        item = manifest_items[experiment_id]
        if (
            (ROOT / item["canonical_path"]).resolve() != model_path.resolve()
            or item["runtime_status"] != "verified"
            or item.get("verification_level") != "build_forward_backward"
            or item["blockers"]
        ):
            raise RuntimeError(f"Manifest SADH runtime evidence mismatch for {experiment_id}: {item}")
        canonical = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        source = yaml.safe_load(SOURCES[experiment_id].read_text(encoding="utf-8"))
        validate_canonical_contract(experiment_id, canonical, source)

        model = YOLO(str(model_path), task="detect").model.cpu()
        head = model.model[-1]
        if not isinstance(head, Detect_LSCSBD) or guess_model_task(model) != "detect":
            raise RuntimeError(f"{experiment_id} did not build a registered detection head")
        if head.nl != 3 or head.nc != 5 or head.reg_max != 16 or head.no != 69:
            raise RuntimeError(
                f"{experiment_id} SADH contract changed: nl={head.nl} nc={head.nc} reg_max={head.reg_max} no={head.no}"
            )
        if head.dw_flags != [False, False, True] or head.shared_layers != 2:
            raise RuntimeError(f"{experiment_id} SADH sharing policy changed")

        model.eval()
        with torch.no_grad():
            output = model(sample)
        prediction, raw = output
        raw_shapes = [list(tensor.shape) for tensor in raw]
        if list(prediction.shape) != [1, 9, 84] or raw_shapes != [[1, 69, 8, 8], [1, 69, 4, 4], [1, 69, 2, 2]]:
            raise RuntimeError(
                f"{experiment_id} SADH forward mismatch: prediction={list(prediction.shape)} raw={raw_shapes}"
            )
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"{experiment_id} SADH inference produced non-finite values")

        if not configure_detection_head_for_export(head, dynamic=False, export_format="onnx"):
            raise RuntimeError(f"{experiment_id} SADH head was not recognized by the exporter")
        with torch.no_grad():
            exported_prediction = model(sample)
        if not isinstance(exported_prediction, torch.Tensor) or list(exported_prediction.shape) != [1, 9, 84]:
            raise RuntimeError(f"{experiment_id} export output is not a single detection tensor")
        head.export = False

        cuda_checked = False
        if torch.cuda.is_available():
            model.cuda().eval()
            if (
                not next(head.parameters()).is_cuda
                or not head.stride.is_cuda
                or not head.anchors.is_cuda
                or not head.strides.is_cuda
            ):
                raise RuntimeError(f"{experiment_id} SADH runtime tensors did not migrate to CUDA")
            with torch.no_grad():
                cuda_output = model(sample.cuda())
            cuda_prediction = cuda_output[0] if isinstance(cuda_output, tuple) else cuda_output
            if list(cuda_prediction.shape) != [1, 9, 84] or not torch.isfinite(cuda_prediction).all():
                raise RuntimeError(f"{experiment_id} SADH CUDA forward failed")
            model.cpu()
            if (
                next(head.parameters()).is_cuda
                or head.stride.is_cuda
                or head.anchors.is_cuda
                or head.strides.is_cuda
            ):
                raise RuntimeError(f"{experiment_id} SADH runtime tensors did not return to CPU")
            cuda_checked = True
        elif args.require_cuda:
            raise RuntimeError("CUDA is required for the stage 3 acceptance gate but is unavailable")

        model.train()
        criterion = native_criterion(model)
        image = torch.rand(1, 3, 64, 64)
        batch = {
            "img": image,
            "batch_idx": torch.zeros(1),
            "cls": torch.zeros((1, 1)),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        }
        train_predictions = model(image)
        loss, loss_items = criterion(train_predictions, batch)
        if not torch.isfinite(loss) or not torch.isfinite(loss_items).all():
            raise RuntimeError(f"{experiment_id} SADH baseline loss is non-finite")
        loss.backward()
        gradient_tensors = {
            "classification": head.cv3.weight.grad,
            "regression": head.cv2.weight.grad,
            "shared_standard": head.shared_conv_std[0].weight.grad,
            "shared_depthwise": head.shared_conv_dw[0].weight.grad,
            "stem": head.stem[0].conv.weight.grad,
        }
        gradient_l1 = {}
        for name, gradient in gradient_tensors.items():
            if gradient is None or not torch.isfinite(gradient).all() or float(gradient.abs().sum()) == 0.0:
                raise RuntimeError(f"{experiment_id} SADH {name} path has no finite gradient")
            gradient_l1[name] = float(gradient.abs().sum())

        model.double()
        if head.stride.dtype != torch.float64 or head.anchors.dtype != torch.float64 or head.strides.dtype != torch.float64:
            raise RuntimeError(f"{experiment_id} SADH runtime tensors did not follow model dtype migration")
        model.float()

        observed = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "layers": len(model.model),
            "CSPStage": sum(isinstance(module, CSPStage) for module in model.modules()),
            "RepHFE": sum(isinstance(module, RepHFE) for module in model.modules()),
            "Detect_LSCSBD": sum(isinstance(module, Detect_LSCSBD) for module in model.modules()),
            "stride": [float(value) for value in model.stride],
            "prediction_shape": list(prediction.shape),
            "raw_shapes": raw_shapes,
            "loss": float(loss.detach()),
            "loss_items": [float(value) for value in loss_items.detach()],
            "gradient_l1": gradient_l1,
            "export_prediction_shape": list(exported_prediction.shape),
            "cuda_round_trip": cuda_checked,
        }
        for key, value in EXPECTED[experiment_id].items():
            if observed[key] != value:
                raise RuntimeError(f"{experiment_id} {key} changed: {observed[key]} != {value}")
        if observed["stride"] != [8.0, 16.0, 32.0]:
            raise RuntimeError(f"{experiment_id} stride mismatch: {observed['stride']}")
        components = item["components"]
        if components["cspstage"] != (observed["CSPStage"] > 0):
            raise RuntimeError(f"{experiment_id} CSPStage evidence disagrees with manifest")
        if components["rephfe"] != (observed["RepHFE"] > 0):
            raise RuntimeError(f"{experiment_id} RepHFE evidence disagrees with manifest")
        if components["sadh"] != (observed["Detect_LSCSBD"] > 0):
            raise RuntimeError(f"{experiment_id} SADH evidence disagrees with manifest")
        results[experiment_id] = observed
        del criterion, gradient_tensors, head, loss, loss_items, model, train_predictions
        if cuda_checked:
            del cuda_output, cuda_prediction
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    legacy = sorted(name for name in sys.modules if "extra_modules" in name)
    if legacy:
        raise RuntimeError(f"Legacy modules loaded during SADH verification: {legacy}")
    print(
        json.dumps(
            {
                "status": "ok",
                "unregistered_error": unregistered_error,
                "head_source": f"{Detect_LSCSBD.__module__}.{Detect_LSCSBD.__name__}",
                "reference_source_checked": args.check_reference_source,
                "cuda_required": args.require_cuda,
                "models": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
