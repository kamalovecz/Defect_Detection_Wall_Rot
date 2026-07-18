from __future__ import annotations

import json
from pathlib import Path

from config import DEFAULT_CONFIG, ROOT, load_config, resolve_repo_path


def main() -> int:
    config = load_config(DEFAULT_CONFIG)
    if Path(config["model"]).is_absolute() or Path(config["data"]).is_absolute():
        raise RuntimeError("Active model and data configuration must use repository-relative paths")
    if "DAD030" in DEFAULT_CONFIG.read_text(encoding="utf-8"):
        raise RuntimeError("Active baseline config must use the Port_Defect identity")
    report = {
        "status": "ok",
        "config": str(DEFAULT_CONFIG.relative_to(ROOT)),
        "model": str(resolve_repo_path(config["model"])),
        "data": str(resolve_repo_path(config["data"])),
        "data_exists": resolve_repo_path(config["data"]).is_file(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
