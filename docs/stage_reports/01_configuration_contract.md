# Stage 1 Configuration Contract

- Status: PASSED
- Dataset identity: `Port_Defect`
- Default config: `training_project/configs/port_defect_baseline.yaml`
- Default data contract: `datasets/Port_Defect/data.yaml`

## Verification

- Training help command: passed.
- Configuration parsing (`verify_config.py`): passed.
- `train.py --check-config`: passed and reported the resolved missing dataset without starting training.
- Normal training with the absent dataset: failed before model construction with the expected actionable `FileNotFoundError`.
- Active configs contain no `DAD030` references or external absolute paths.
- Registry, task import boundary, and external block construction checks: passed.
- Git whitespace check: passed.

## Changes

- Added a configuration-driven training entrypoint with CLI overrides.
- Added repository-relative Port_Defect baseline and dataset example configurations.
- Removed the two active DAD030 dataset configurations.
- Added explicit PyTorch and NVIDIA conda channels and pinned PyTorch 2.1.
- Updated README training commands to use the baseline config.

## Residual risk

The real dataset is intentionally not tracked. The one-epoch smoke train remains gated until `datasets/Port_Defect/data.yaml` and its image/label directories are available.

## Gate

All Stage 1 requirements that do not require the external dataset pass. Stage 2 may proceed.
