from __future__ import annotations

import importlib.abc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

CANONICAL_DIR = ROOT / "training_project" / "weights" / "canonical"
MANIFEST_PATH = CANONICAL_DIR / "DAD030_best_target_manifest.json"
STATE_DICT_PATH = CANONICAL_DIR / "DAD030_best_target_state_dict.pt"
BLOCKED = {"ultralytics.nn.extra_modules.rephfe", "ultralytics.nn.extra_modules.prune_module"}


class BlockLegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in BLOCKED:
            raise ImportError(f"Blocked legacy module during canonical load: {fullname}")
        return None


def load_canonical_model(device="cpu"):
    sys.meta_path.insert(0, BlockLegacyFinder())
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing canonical manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("topology_case") == "CASE_C" or not STATE_DICT_PATH.exists():
        raise RuntimeError("Canonical state_dict is unavailable because source checkpoint is CASE_C against target YAML.")
    import torch
    from ultralytics import YOLO
    model = YOLO(str(Path(manifest["model_yaml"]))).model.float().eval().to(device)
    state_dict = torch.load(STATE_DICT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model, manifest


if __name__ == "__main__":
    model, manifest = load_canonical_model()
    print(json.dumps({"status": "ok", "topology_case": manifest.get("topology_case"), "model": model.__class__.__name__}, indent=2))
