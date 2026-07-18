# Stage 0 Baseline

- Status: PASSED
- Captured: 2026-07-18 (Asia/Shanghai)
- Branch: `codex/decouple-training-v1`
- Baseline commit: `937a757da8f55a55e5b31e1fa287ba721de421bd`
- Python: `3.10.19`
- PyTorch: `2.1.0+cu121`
- CUDA runtime: `12.1`

## Verification

- `verify_registry.py`: passed
- `verify_tasks_import_boundary.py`: passed
- `verify_external_blocks.py`: passed
- Target model: `CSPStage=4`, `RepHFE=2`
- Legacy modules loaded by the main YAML build: none

## Baseline hashes

- `training_project/models/B4_A-GFPN_RepHFE_target.yaml`: `32c400d0a2ee4d6619eaeeabf638a1bb5ba2bd6682affe5cca1577af6c56b4e2`
- `defect_modules/blocks.py`: `3a5ffd278f3af8c82f38fd8128fe1482eb73dab2e5406411e33b215a7fc6c4f9`
- `ultralytics-main/ultralytics/nn/tasks.py`: `00d32bc819f4b42b013dbea810486a0537ffe856d01dca21fe1c7323a6d08f8f`

## Known baseline limitations

- RuleLoss exists but is disabled by default and is not active in the standard training path.
- The historical checkpoint is topology case `CASE_C`; it is not a canonical checkpoint for the target YAML.
- Canonical ONNX export is skipped because no topology-compatible canonical state dict exists.
- The staged runtime still contains the historical `extra_modules` tree.

## Gate

Stage 0 is traceable and all existing boundary checks pass. Stage 1 may proceed.
