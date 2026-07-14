"""Create or refresh the HARP-Net bundle from fixed selected artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(r"D:\defect_detection")
SELECTED = ROOT / "harpnet_selected_artifacts"
BUNDLE = ROOT / "export_pipeline" / "bundles" / "harpnet_b4_rephfe"


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SELECTED / "B4_A-GFPN_RepHFE_target.yaml", BUNDLE / "model.yaml")
    shutil.copy2(SELECTED / "DAD030_best_target.pt", BUNDLE / "best.pt")
    print(f"Bundle refreshed: {BUNDLE}")


if __name__ == "__main__":
    main()
