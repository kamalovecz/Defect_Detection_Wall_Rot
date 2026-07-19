# HARP-Net Port_Defect Training

This repository is the training-side source of truth for the HARP-Net defect detector. It owns model modules, reproducible training, checkpoints, and validated ONNX artifacts. RKNN conversion and board validation belong to `331_PC_RKNN`.

## Architecture

```text
training_project -> defect_modules -> pinned ultralytics-main
        |                                  |
        +---------- PT / ONNX -------------+
                           -> 331_PC_RKNN
```

- `defect_modules`: CSPStage, RepHFE, RuleLoss, and the one-way runtime installer.
- `training_project`: Port_Defect configs, model YAML, training entrypoint, and verification scripts.
- `ultralytics-main`: pruned Ultralytics 8.2.50 runtime with a narrow extension registry.
- `legacy_compat`: isolated diagnostics for the historical CASE_C checkpoint.
- `export_pipeline`: topology validation, ONNX export, and PT/ONNX consistency checks.

Vendor code never imports project code. Project modules are registered explicitly by `defect_modules.integration.install()` before model construction.

## Environment

```powershell
conda env create -f environment.yml
conda activate harpnet_acceptance
```

The environment pins Python 3.10, PyTorch 2.1, CUDA 12.1, ONNX, and ONNX Runtime. The bundled runtime is installed editable from `ultralytics-main`.

## Dataset

Extract the external dataset to:

```text
datasets/Port_Defect/
  data.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

The tracked descriptor is `datasets/Port_Defect/data.yaml`. Images, labels, checkpoints, runs, and exports are intentionally ignored by Git.

The currently verified dataset snapshot contains 26 exact image-content duplicates across train/val/test. It is accepted only for engineering smoke tests. Formal ablation training and accuracy claims are blocked until the splits are rebuilt and the content fingerprint is reviewed.

## Verification

```powershell
python training_project/verify_all.py
```

This checks configuration, one-way imports, module registration, shapes, model signature, RuleLoss activation, CASE_C isolation, and export rejection behavior.

Run the six-model ablation gate on a CUDA host:

```powershell
python training_project/ablations/verify_ablation_models.py --require-cuda
```

For the current host's complete engineering acceptance, including real-data and existing smoke/ONNX evidence:

```powershell
python training_project/ablations/verify_final_acceptance.py `
  --require-real-data `
  --smoke-state training_project/runs/ablation_smoke_final/stage6_state.json `
  --onnx-manifest export_pipeline/outputs/port_defect_smoke/artifact_manifest.json
```

## Training

Baseline, with RuleLoss disabled:

```powershell
python -m training_project.train `
  --config training_project/configs/port_defect_baseline.yaml
```

RuleLoss engineering smoke configuration:

```powershell
python -m training_project.train `
  --config training_project/configs/port_defect_ruleloss_smoke.yaml
```

CLI arguments such as `--data`, `--epochs`, `--batch`, `--device`, and `--name` override the selected YAML config. Every run writes `run_manifest.json` with the Git commit, model hash, class mapping, seed, RuleLoss state, and observed criterion runtime.

RuleLoss is disabled by default. It is selected through the criterion factory only when `loss.rule.enabled` is true; the training callback synchronizes epoch state.

## Ablation matrix

| ID | Structure | RuleLoss | Purpose |
| --- | --- | --- | --- |
| A0 | YOLOv8n | off | external baseline |
| B1 | SADH | off | standalone SADH point |
| B2 | RepHFE | off | RepHFE candidate |
| B3 | A-GFPN | off | A-GFPN candidate |
| B4 | A-GFPN + RepHFE | off | controlled B4-B2/B4-B3 comparisons |
| B5 | A-GFPN + RepHFE + SADH | off | full structure |
| L1 | B5 | on | RuleLoss-only comparison against B5 |

Canonical model YAMLs are under `training_project/models/ablations/`, training overlays under `training_project/configs/ablations/`, and the immutable source archive plus provenance is described by `training_project/ablations/manifest.yaml`. B4 intentionally reuses `training_project/models/B4_A-GFPN_RepHFE_target.yaml` as its single canonical source.

To run the engineering smoke matrix after materializing the reviewed dataset:

```powershell
python training_project/ablations/run_stage6_smoke.py `
  --project training_project/runs/ablation_smoke `
  --run-prefix smoke
```

The runner requires a clean Git worktree and binds every checkpoint to its experiment, effective config, model/data/content hashes, run name, overrides, and commit. See `docs/ABLATION_GUIDE.md` before copying or launching experiments.

## Export

Only a completed, topology-compatible checkpoint may be exported:

```powershell
python export_pipeline/export_onnx.py `
  --checkpoint training_project/runs/<run>/weights/best.pt `
  --run-manifest training_project/runs/<run>/run_manifest.json `
  --output-dir export_pipeline/outputs/<artifact> `
  --name <artifact>

python export_pipeline/verify_onnx_consistency.py `
  --manifest export_pipeline/outputs/<artifact>/artifact_manifest.json
```

The artifact contains PT, ONNX, model YAML, and a relative-path manifest with hashes, classes, preprocessing, topology, and numeric validation. A one-epoch smoke artifact proves the engineering path but is not a converged deployment candidate.

## Legacy checkpoint

The historical DAD030-named pruned checkpoint is CASE_C against the target YAML and cannot be used as a canonical export input. Its compatibility source and diagnostic record are isolated under `legacy_compat`; the active registry and runtime do not import them.
