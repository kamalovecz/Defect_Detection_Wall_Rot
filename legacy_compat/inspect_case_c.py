from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "legacy_compat" / "records" / "case_c" / "DAD030_best_target_manifest.json"


def main() -> int:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    modules = record.get("custom_module_types", [])
    c2f = [item for item in modules if item.get("class_name") == "C2f_v2"]
    count = sum(int(item.get("instance_count", 0)) for item in c2f)
    if record.get("topology_case") != "CASE_C" or count != 8:
        raise RuntimeError(f"Unexpected legacy record: topology={record.get('topology_case')} C2f_v2={count}")
    print(json.dumps({
        "status": "ok",
        "topology_case": "CASE_C",
        "C2f_v2_instances": count,
        "canonical_state_dict": record.get("canonical_state_dict"),
        "active_training_dependency": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
