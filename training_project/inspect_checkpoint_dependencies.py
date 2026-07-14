from __future__ import annotations

from collections import Counter
import datetime
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

SOURCE_PT = ROOT / "training_project" / "weights" / "DAD030_best_target.pt"
TARGET_YAML = ROOT / "training_project" / "models" / "B4_A-GFPN_RepHFE_target.yaml"
CANONICAL_DIR = ROOT / "training_project" / "weights" / "canonical"
STATE_DICT_PATH = CANONICAL_DIR / "DAD030_best_target_state_dict.pt"
MANIFEST_PATH = CANONICAL_DIR / "DAD030_best_target_manifest.json"
SHA_PATH = CANONICAL_DIR / "SHA256SUMS.json"
REPORT_PATH = CANONICAL_DIR / "pruned_architecture_report.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def class_path(obj) -> str:
    cls = obj if isinstance(obj, type) else obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def class_file(obj):
    try:
        return inspect.getfile(obj if isinstance(obj, type) else obj.__class__)
    except Exception:
        mod = sys.modules.get((obj if isinstance(obj, type) else obj.__class__).__module__)
        return getattr(mod, "__file__", None)


def first_weight_shape(module):
    for _, p in module.named_parameters(recurse=True):
        return list(p.shape)
    for _, b in module.named_buffers(recurse=True):
        return list(b.shape)
    return None


def layer_summary(model):
    rows = []
    for i, m in enumerate(getattr(model, "model", [])):
        rows.append({
            "index": i,
            "type": class_path(m),
            "from": getattr(m, "f", None),
            "params": int(sum(p.numel() for p in m.parameters())),
            "first_weight_shape": first_weight_shape(m),
        })
    return rows


def search_rebuild_candidates(limit=80):
    roots = [ROOT / "training_project", ROOT / "export_pipeline", ROOT / "prune_distill_exp", ROOT / "harpnet_decouple_workspace", ROOT / "harpnet_selected_artifacts", ROOT / "ultralytics-main"]
    needles = ["C2f_v2", "prune_module", "model_c2f_v2", "DAD030", "sparse_prune"]
    hits = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if len(hits) >= limit:
                return hits
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json", ".py", ".md", ".txt"}:
                continue
            if "backups" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            matched = [n for n in needles if n in text or n.lower() in path.name.lower()]
            if matched:
                hits.append({"path": str(path), "matched": matched})
    return hits


def compare_models(source_model, target_model):
    source_layers = layer_summary(source_model)
    target_layers = layer_summary(target_model)
    layer_diffs = []
    for i in range(max(len(source_layers), len(target_layers))):
        s = source_layers[i] if i < len(source_layers) else None
        t = target_layers[i] if i < len(target_layers) else None
        if s != t:
            layer_diffs.append({"index": i, "source": s, "target": t})
    source_sd = source_model.state_dict()
    target_sd = target_model.state_dict()
    source_keys = set(source_sd)
    target_keys = set(target_sd)
    shape_mismatch = []
    for k in sorted(source_keys & target_keys):
        if tuple(source_sd[k].shape) != tuple(target_sd[k].shape):
            shape_mismatch.append({"key": k, "source_shape": list(source_sd[k].shape), "target_shape": list(target_sd[k].shape)})
    if len(source_layers) == len(target_layers) and not layer_diffs and not (source_keys - target_keys) and not (target_keys - source_keys) and not shape_mismatch:
        case = "CASE_A"
    elif len(source_layers) == len(target_layers) and not shape_mismatch:
        case = "CASE_B"
    else:
        case = "CASE_C"
    return {
        "topology_case": case,
        "source_layer_count": len(source_layers),
        "target_layer_count": len(target_layers),
        "source_parameter_count": int(sum(p.numel() for p in source_model.parameters())),
        "target_parameter_count": int(sum(p.numel() for p in target_model.parameters())),
        "source_state_dict_key_count": len(source_sd),
        "target_state_dict_key_count": len(target_sd),
        "missing_keys": sorted(target_keys - source_keys),
        "unexpected_keys": sorted(source_keys - target_keys),
        "shape_mismatch_keys": shape_mismatch,
        "layer_diffs": layer_diffs,
        "source_layers": source_layers,
        "target_layers": target_layers,
    }


def main() -> int:
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    from defect_modules.patch import apply
    apply(verbose=False, pickle_compat=True, legacy_aliases=False, strict=True)
    from ultralytics import YOLO
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, _ = torch_safe_load(str(SOURCE_PT))
    source_model = YOLO(str(SOURCE_PT)).model.float().eval()
    target_model = YOLO(str(TARGET_YAML)).model.float().eval()
    comparison = compare_models(source_model, target_model)

    module_counts = Counter(class_path(m) for m in source_model.modules())
    custom_modules = []
    for module_type, count in sorted(module_counts.items()):
        if any(token in module_type for token in ("defect_modules", "extra_modules", "prune_module")):
            cls = next((m.__class__ for m in source_model.modules() if class_path(m) == module_type), None)
            custom_modules.append({
                "module_path": module_type,
                "class_name": module_type.rsplit(".", 1)[-1],
                "instance_count": int(count),
                "class_file": class_file(cls) if cls else None,
                "from_defect_modules": "defect_modules" in module_type,
                "from_extra_modules": "extra_modules" in module_type,
                "from_prune_module": "prune_module" in module_type,
                "belongs_to_model_structure": True,
                "belongs_only_to_training_state": False,
                "replaceable_by_state_dict": comparison["topology_case"] in {"CASE_A", "CASE_B"},
            })

    sd = source_model.state_dict()
    report = {
        "source_checkpoint": str(SOURCE_PT),
        "source_sha256": sha256(SOURCE_PT),
        "model_yaml": str(TARGET_YAML),
        "model_yaml_sha256": sha256(TARGET_YAML),
        "checkpoint_top_level_fields": list(ckpt.keys()) if isinstance(ckpt, dict) else None,
        "model_type": class_path(ckpt.get("model")) if isinstance(ckpt, dict) and ckpt.get("model") is not None else None,
        "ema_type": class_path(ckpt.get("ema")) if isinstance(ckpt, dict) and ckpt.get("ema") is not None else None,
        "optimizer_exists": bool(isinstance(ckpt, dict) and ckpt.get("optimizer") is not None),
        "train_args_exists": bool(isinstance(ckpt, dict) and ckpt.get("train_args") is not None),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_fitness": str(ckpt.get("best_fitness")) if isinstance(ckpt, dict) else None,
        "model_parameter_count": int(sum(p.numel() for p in source_model.parameters())),
        "state_dict_key_count": len(sd),
        "custom_module_types": custom_modules,
        "loaded_extra_modules": sorted(k for k in sys.modules if k.startswith("ultralytics.nn.extra_modules")),
        "comparison": comparison,
        "rebuild_candidate_files": search_rebuild_candidates(),
    }

    manifest = {
        "source_checkpoint": str(SOURCE_PT),
        "source_sha256": report["source_sha256"],
        "model_yaml": str(TARGET_YAML),
        "model_yaml_sha256": report["model_yaml_sha256"],
        "topology_case": comparison["topology_case"],
        "model_class": class_path(source_model),
        "parameter_count": report["model_parameter_count"],
        "state_dict_key_count": report["state_dict_key_count"],
        "custom_module_types": custom_modules,
        "required_registry_entries": ["CSPStage", "RepHFE"] if comparison["topology_case"] in {"CASE_A", "CASE_B"} else [],
        "input_size": [1, 3, 640, 640],
        "number_of_classes": getattr(source_model.model[-1], "nc", None),
        "class_names": getattr(source_model, "names", None),
        "pruning_metadata": {"uses_prune_module": any(m["from_prune_module"] for m in custom_modules), "source_layers": comparison["source_layer_count"], "target_layers": comparison["target_layer_count"]},
        "creation_time": datetime.datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "ultralytics_path": str(ULTRALYTICS_MAIN),
        "defect_modules_version": getattr(__import__("defect_modules"), "__version__", None),
        "canonical_state_dict": str(STATE_DICT_PATH) if comparison["topology_case"] in {"CASE_A", "CASE_B"} else None,
    }
    if comparison["topology_case"] in {"CASE_A", "CASE_B"}:
        target_model.load_state_dict(sd, strict=True)
        torch.save(target_model.state_dict(), STATE_DICT_PATH)
    else:
        if STATE_DICT_PATH.exists():
            STATE_DICT_PATH.unlink()
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    sums = {"source_checkpoint": report["source_sha256"], "model_yaml": report["model_yaml_sha256"], "manifest": sha256(MANIFEST_PATH)}
    if STATE_DICT_PATH.exists():
        sums["canonical_state_dict"] = sha256(STATE_DICT_PATH)
    if REPORT_PATH.exists():
        sums["pruned_architecture_report"] = sha256(REPORT_PATH)
    SHA_PATH.write_text(json.dumps(sums, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "status": "ok",
        "topology_case": comparison["topology_case"],
        "canonical_state_dict": str(STATE_DICT_PATH) if STATE_DICT_PATH.exists() else None,
        "manifest": str(MANIFEST_PATH),
        "sha256sums": str(SHA_PATH),
        "pruned_architecture_report": str(REPORT_PATH) if REPORT_PATH.exists() else None,
        "custom_module_types": custom_modules,
        "source_parameter_count": comparison["source_parameter_count"],
        "target_parameter_count": comparison["target_parameter_count"],
        "shape_mismatch_count": len(comparison["shape_mismatch_keys"]),
        "layer_diff_count": len(comparison["layer_diffs"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if comparison["topology_case"] == "CASE_C":
        print("CASE_C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
