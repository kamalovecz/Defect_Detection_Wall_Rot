from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    ROOT / "training_project" / "verify_environment.py",
    ROOT / "training_project" / "verify_config.py",
    ROOT / "training_project" / "verify_registry.py",
    ROOT / "training_project" / "verify_tasks_import_boundary.py",
    ROOT / "training_project" / "verify_registration_boundary.py",
    ROOT / "training_project" / "verify_external_blocks.py",
    ROOT / "training_project" / "verify_blocks.py",
    ROOT / "training_project" / "verify_model_signature.py",
    ROOT / "training_project" / "verify_ruleloss_factory.py",
    ROOT / "legacy_compat" / "inspect_case_c.py",
    ROOT / "export_pipeline" / "verify_export_rejections.py",
]


def main() -> int:
    for check in CHECKS:
        print(f"[verify_all] {check.relative_to(ROOT)}", flush=True)
        subprocess.run(
            [sys.executable, "training_project/run_verifier_safely.py", str(check)],
            cwd=ROOT,
            check=True,
        )
    print(f"[verify_all] PASSED checks={len(CHECKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
