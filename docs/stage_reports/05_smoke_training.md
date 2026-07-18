# Stage 5 Real-Data Smoke Training

- Status: PASSED
- Hardware: NVIDIA GeForce RTX 4090
- Dataset: external read-only Port_Defect source, 5 classes
- Dataset cache: disabled

## Baseline run

- Run: `training_project/runs/stage5_baseline`
- Epochs: 1
- Checkpoint reload and random-input inference: passed
- Metrics and losses: finite
- RuleLoss enabled: false
- `best.pt` SHA256: `2bfecdff4022b16ed514d4d6ee4529122742fbc4062c480b842c26eaf1925f4a`

## RuleLoss run

- Run: `training_project/runs/stage5_ruleloss`
- Epochs: 1
- Checkpoint reload and random-input inference: passed
- Metrics and losses: finite
- Criterion: `defect_modules.loss.v8DetectionLoss`
- Rule updates: 607
- Final lambda: 1.0
- `best.pt` SHA256: `06a0d8cb5d6aa887a90c63e326c0ae2546a02fcc77252765e8dc52d3cb2efc5e`

## Verification

- Forward, backward, optimizer updates, validation, and checkpoint writes completed for both modes.
- Reloaded checkpoints require no legacy modules.
- Run outputs are ignored by Git and the external dataset was not modified.
- Run manifests record config, model hash, dataset identity, classes, seed, RuleLoss state, and runtime criterion evidence.

## Limitation

These are one-epoch engineering smoke checkpoints, not converged accuracy results or deployment candidates.

## Gate

The real-data training loop passes in baseline and active RuleLoss modes. Stage 6 may use the baseline smoke checkpoint to validate the canonical PT-to-ONNX contract.
