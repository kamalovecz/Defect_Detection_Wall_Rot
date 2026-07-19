# HARP-Net Ablation Guide

## Sources and canonical models

- `training_project/models/ablations/source/` is the byte-exact historical archive. Do not edit it.
- `training_project/models/ablations/` contains runnable canonical models A0, B1, B2, B3 and B5.
- B4 uses `training_project/models/B4_A-GFPN_RepHFE_target.yaml` as its single canonical YAML.
- `training_project/ablations/manifest.yaml` records provenance, components and valid controlled comparisons.

Only B4-B2 (A-GFPN), B4-B3 (RepHFE), B5-B4 (SADH) and L1-B5 (RuleLoss) are single-factor controlled comparisons. A0 and B1 are useful reference points but are not single-factor evidence against one another.

## Dataset materialization

Keep `datasets/Port_Defect/data.yaml` tracked and place or map the real data below it:

```text
datasets/Port_Defect/
  images/train  images/val  images/test
  labels/train  labels/val  labels/test
```

Then verify both paths and contents:

```powershell
python training_project/ablations/dataset_contract.py datasets/Port_Defect --content
```

The committed snapshot currently detects 26 exact images across different splits, so formal experiments are blocked. Do not silently change files or split membership: update the matrix hashes through a reviewed data-migration change.

## Build and training gates

```powershell
python training_project/ablations/verify_ablation_models.py --require-cuda
python training_project/ablations/run_stage6_smoke.py `
  --project training_project/runs/ablation_smoke `
  --run-prefix smoke
```

The smoke runner requires a clean Git worktree. It stops after the first failed experiment, validates each checkpoint immediately, and writes `run_manifest.json` plus `smoke_contract.json`. `--reuse-completed` is safe only when the complete experiment/config/model/data/checkpoint contract matches.

## Copying an experiment

Do not copy a completed run directory to represent another experiment. To create a new reviewed structure experiment:

1. Add a canonical model YAML without modifying the source archive.
2. Add its component/provenance entry to `manifest.yaml`.
3. Add a minimal overlay that extends `common.yaml` and changes only model/name.
4. Add the ID and model hash to `training_matrix.yaml`.
5. Extend the closed-set expectations and signatures in the verification scripts.
6. Run all earlier gates and obtain adversarial review before real training.

RuleLoss remains a separate loss dimension. Do not enable it inside A0-B5 structure configs.
