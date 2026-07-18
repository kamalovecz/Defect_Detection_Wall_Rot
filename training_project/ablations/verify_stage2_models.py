from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODELS = {
    "A0": ROOT / "training_project/models/ablations/A0_yolov8n.yaml",
    "B2": ROOT / "training_project/models/ablations/B2_RepHFE.yaml",
    "B3": ROOT / "training_project/models/ablations/B3_A-GFPN.yaml",
    "B4": ROOT / "training_project/models/B4_A-GFPN_RepHFE_target.yaml",
}
SOURCES = {
    "A0": ROOT / "training_project/models/ablations/source/yolov8n.yaml",
    "B2": ROOT / "training_project/models/ablations/source/B2_RepHFE.yaml",
    "B3": ROOT / "training_project/models/ablations/source/B3_A-GFPN.yaml",
    "B4": ROOT / "training_project/models/ablations/source/B4_A-GFPN_RepHFE_target.yaml",
}
EXPECTED = {
    "A0": {"parameters": 3_011_823, "layers": 23, "CSPStage": 0, "RepHFE": 0},
    "B2": {"parameters": 1_969_391, "layers": 25, "CSPStage": 4, "RepHFE": 2},
    "B3": {"parameters": 2_303_663, "layers": 27, "CSPStage": 4, "RepHFE": 0},
    "B4": {"parameters": 2_308_655, "layers": 25, "CSPStage": 4, "RepHFE": 2},
}
COMPONENT_KEYS = {"cspstage", "a_gfpn", "rephfe", "sadh", "rule_loss"}
EXPECTED_ROLES = {
    "A0": "external_baseline",
    "B1": "standalone_architecture",
    "B2": "controlled_component_candidate",
    "B3": "controlled_component_candidate",
    "B4": "controlled_component_candidate",
    "B5": "controlled_component_candidate",
}
EXPECTED_COMPARISONS = {
    "B4-B2": "a_gfpn",
    "B4-B3": "rephfe",
    "B5-B4": "sadh",
}


def validate_canonical_contract(experiment_id: str, canonical: dict, source: dict) -> None:
    expected = deepcopy(source)
    expected["nc"] = 5
    if experiment_id == "A0":
        expected["scales"] = {"n": expected["scales"]["n"]}
    expected["scale"] = "n"

    observed = deepcopy(canonical)
    if list(observed.get("scales", {})) == ["n"]:
        observed.setdefault("scale", "n")
    if observed != expected:
        differing_keys = sorted(
            key for key in set(observed) | set(expected) if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(
            f"{experiment_id} canonical semantics drifted outside declared nc/scale transforms: {differing_keys}"
        )


def validate_interpretation_contract(manifest: dict) -> dict[str, dict]:
    items = {item["id"]: item for item in manifest["experiments"]}
    if set(items) != set(EXPECTED_ROLES):
        raise RuntimeError(f"Manifest experiment set changed: {sorted(items)}")
    for experiment_id, expected_role in EXPECTED_ROLES.items():
        item = items[experiment_id]
        if item.get("reference_role") != expected_role:
            raise RuntimeError(f"Manifest reference_role mismatch for {experiment_id}")
        components = item.get("components", {})
        if set(components) != COMPONENT_KEYS or not all(isinstance(value, bool) for value in components.values()):
            raise RuntimeError(f"Manifest components are incomplete for {experiment_id}: {components}")

    comparisons = manifest.get("interpretation_contract", {}).get("controlled_comparisons", [])
    observed_pairs = {item.get("comparison"): item for item in comparisons}
    if len(observed_pairs) != len(comparisons) or set(observed_pairs) != set(EXPECTED_COMPARISONS):
        raise RuntimeError(f"Controlled comparison set changed: {sorted(observed_pairs)}")
    for pair, expected_factor in EXPECTED_COMPARISONS.items():
        comparison = observed_pairs[pair]
        left_id, right_id = pair.split("-", 1)
        isolated_factor = comparison.get("isolated_factor")
        if isolated_factor != expected_factor:
            raise RuntimeError(f"Controlled comparison {pair} has wrong isolated factor: {isolated_factor}")
        left = items[left_id]["components"]
        right = items[right_id]["components"]
        differences = {key for key in COMPONENT_KEYS if left[key] != right[key]}
        if differences != {isolated_factor}:
            raise RuntimeError(f"Controlled comparison {pair} differs in {sorted(differences)}")
        controls = comparison.get("controls", [])
        expected_controls = COMPONENT_KEYS - {isolated_factor, "rule_loss"}
        if set(controls) != expected_controls or len(controls) != len(expected_controls):
            raise RuntimeError(f"Controlled comparison {pair} has incomplete controls: {controls}")
        if any(left[key] != right[key] for key in controls):
            raise RuntimeError(f"Controlled comparison {pair} has invalid controls: {controls}")
    return items


def main() -> int:
    from defect_modules.blocks import CSPStage, RepHFE
    from defect_modules.integration import install
    from ultralytics import YOLO
    from ultralytics.nn.extensions import registered_model_modules
    from ultralytics.nn.tasks import DetectionModel

    manifest = yaml.safe_load((ROOT / "training_project/ablations/manifest.yaml").read_text(encoding="utf-8"))
    manifest_items = validate_interpretation_contract(manifest)
    for experiment_id, model_path in MODELS.items():
        item = manifest_items[experiment_id]
        declared_path = (ROOT / item["canonical_path"]).resolve()
        if (
            declared_path != model_path.resolve()
            or item["runtime_status"] != "verified"
            or item.get("verification_level") != "build_forward"
            or item["blockers"]
        ):
            raise RuntimeError(f"Manifest runtime evidence mismatch for {experiment_id}: {item}")

    mutated_manifest = deepcopy(manifest)
    next(item for item in mutated_manifest["experiments"] if item["id"] == "B2")["components"]["rephfe"] = False
    try:
        validate_interpretation_contract(mutated_manifest)
    except RuntimeError as exc:
        manifest_negative_error = str(exc)
    else:
        raise RuntimeError("Mutated manifest component unexpectedly passed interpretation validation")

    if registered_model_modules():
        raise RuntimeError("Stage 2 verifier must start with an empty extension registry")
    torch.manual_seed(42)
    sample = torch.zeros(1, 3, 64, 64)

    baseline_config = yaml.safe_load(MODELS["A0"].read_text(encoding="utf-8"))
    baseline_source = yaml.safe_load(SOURCES["A0"].read_text(encoding="utf-8"))
    mutated_canonical = deepcopy(baseline_config)
    mutated_canonical["activation"] = "nn.ReLU()"
    try:
        validate_canonical_contract("A0", mutated_canonical, baseline_source)
    except RuntimeError as exc:
        canonical_negative_error = str(exc)
    else:
        raise RuntimeError("Mutated canonical activation unexpectedly passed semantic validation")

    identity_config = deepcopy(baseline_config)
    identity_config["backbone"][-1] = [-1, 1, "nn.Identity", []]
    identity_model = DetectionModel(identity_config, verbose=False).eval()
    with torch.no_grad():
        identity_output = identity_model(sample)
    identity_prediction = identity_output[0] if isinstance(identity_output, tuple) else identity_output
    if list(identity_prediction.shape) != [1, 9, 84]:
        raise RuntimeError(f"Standard nn.Identity parser fallback failed: {list(identity_prediction.shape)}")

    unknown_config = deepcopy(baseline_config)
    unknown_config["backbone"][0][2] = "DefinitelyMissingBlock"
    try:
        DetectionModel(unknown_config, verbose=False)
    except ValueError as exc:
        unknown_error = str(exc)
        if "DefinitelyMissingBlock" not in unknown_error or "layer index 0" not in unknown_error:
            raise RuntimeError(f"Unknown-token error lacks context: {unknown_error}") from exc
    else:
        raise RuntimeError("Unknown YAML token unexpectedly passed parser validation")

    removed_legacy_config = deepcopy(baseline_config)
    removed_legacy_config["Warehouse_Manager"] = True
    try:
        DetectionModel(removed_legacy_config, verbose=False)
    except ValueError as exc:
        removed_legacy_error = str(exc)
        if "removed legacy feature" not in removed_legacy_error:
            raise RuntimeError(f"Removed legacy feature error is unclear: {removed_legacy_error}") from exc
    else:
        raise RuntimeError("Removed Warehouse_Manager feature unexpectedly passed validation")

    results = {}
    for experiment_id, model_path in MODELS.items():
        if experiment_id == "B2":
            install({"enabled": False})
        config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        source = yaml.safe_load(SOURCES[experiment_id].read_text(encoding="utf-8"))
        validate_canonical_contract(experiment_id, config, source)
        model = YOLO(str(model_path), task="detect").model.eval()
        with torch.no_grad():
            output = model(sample)
        prediction = output[0] if isinstance(output, tuple) else output
        if list(prediction.shape) != [1, 9, 84] or not torch.isfinite(prediction).all():
            raise RuntimeError(f"{experiment_id} forward mismatch: shape={list(prediction.shape)}")
        observed = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "layers": len(model.model),
            "CSPStage": sum(isinstance(module, CSPStage) for module in model.modules()),
            "RepHFE": sum(isinstance(module, RepHFE) for module in model.modules()),
            "stride": [float(value) for value in model.stride],
            "prediction_shape": list(prediction.shape),
        }
        expected = EXPECTED[experiment_id]
        for key, value in expected.items():
            if observed[key] != value:
                raise RuntimeError(f"{experiment_id} {key} changed: {observed[key]} != {value}")
        if observed["stride"] != [8.0, 16.0, 32.0]:
            raise RuntimeError(f"{experiment_id} stride mismatch: {observed['stride']}")
        components = manifest_items[experiment_id]["components"]
        if components["cspstage"] != (observed["CSPStage"] > 0):
            raise RuntimeError(f"{experiment_id} CSPStage evidence disagrees with manifest")
        if components["rephfe"] != (observed["RepHFE"] > 0):
            raise RuntimeError(f"{experiment_id} RepHFE evidence disagrees with manifest")
        results[experiment_id] = observed
    legacy = sorted(name for name in sys.modules if "extra_modules" in name)
    if legacy:
        raise RuntimeError(f"Legacy modules loaded during stage 2 verification: {legacy}")
    print(
        json.dumps(
            {
                "status": "ok",
                "unknown_token_error": unknown_error,
                "removed_legacy_error": removed_legacy_error,
                "canonical_negative_error": canonical_negative_error,
                "manifest_negative_error": manifest_negative_error,
                "identity_prediction_shape": list(identity_prediction.shape),
                "models": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
