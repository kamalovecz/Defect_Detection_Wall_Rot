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
REPORT_PATH = CANONICAL_DIR / "canonical_checkpoint_verification.json"
BLOCKED = {"ultralytics.nn.extra_modules.rephfe", "ultralytics.nn.extra_modules.prune_module"}


class BlockLegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in BLOCKED:
            raise ImportError(f"Blocked legacy module during canonical verification: {fullname}")
        return None


def main() -> int:
    sys.meta_path.insert(0, BlockLegacyFinder())
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    if manifest.get("topology_case") == "CASE_C" or not STATE_DICT_PATH.exists():
        report = {
            "status": "CANONICAL_REBUILD_NOT_AVAILABLE",
            "topology_case": manifest.get("topology_case"),
            "reason": "Source PT topology uses prune_module/C2f_v2 and does not match target YAML, so no strict canonical state_dict was produced.",
            "blocked_modules": sorted(BLOCKED),
            "legacy_modules_loaded": sorted(k for k in sys.modules if k in BLOCKED),
            "defect_modules_loss_loaded": "defect_modules.loss" in sys.modules,
            "training_project_harp_trainer_loaded": "training_project.harp_trainer" in sys.modules,
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "not_reached_in_case_c"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
