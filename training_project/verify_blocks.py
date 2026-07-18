from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_MAIN = ROOT / "ultralytics-main"
for path in (ROOT, ULTRALYTICS_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    from defect_modules.blocks import CSPStage, RepDWConv, RepHFE

    torch.manual_seed(42)
    csp = CSPStage(64, 128, 2).eval()
    csp_shape = list(csp(torch.randn(1, 64, 32, 32)).shape)
    if csp_shape != [1, 128, 32, 32]:
        raise RuntimeError(f"CSPStage shape mismatch: {csp_shape}")
    hfe = RepHFE(64, 128).eval()
    hfe_shape = list(hfe(torch.randn(1, 64, 16, 16)).shape)
    if hfe_shape != [1, 128, 32, 32]:
        raise RuntimeError(f"RepHFE shape mismatch: {hfe_shape}")
    rep = RepDWConv(16).eval()
    sample = torch.randn(1, 16, 16, 16)
    with torch.no_grad():
        before = rep(sample)
        rep.switch_to_deploy()
        after = rep(sample)
    max_abs = float((before - after).abs().max())
    if max_abs > 1e-5:
        raise RuntimeError(f"RepDWConv deploy conversion mismatch: {max_abs}")
    print(json.dumps({
        "status": "ok",
        "CSPStage_shape": csp_shape,
        "RepHFE_shape": hfe_shape,
        "RepDWConv_deploy_max_abs": max_abs,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
