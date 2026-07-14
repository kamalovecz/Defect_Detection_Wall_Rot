# Repository Content Policy

## Track in Git

- Source code under `defect_modules`, `training_project`, `export_pipeline`, and the minimal `ultralytics-main` source package.
- Model YAML files.
- Verification scripts.
- Small JSON reports that document current validation status.
- Documentation under `docs`.
- Minimal legacy pickle compatibility source files required for loading the fixed PT.

## Keep out of Git

- Datasets and image/label folders.
- Training runs and logs.
- Weights: `.pt`, `.pth`, `.ckpt`.
- Exports: `.onnx`, `.rknn`, `.engine`, `.torchscript`.
- Python caches and local environment folders.
- The protected sibling directory `D:\defect_detection\ultralytics`.
- Massive historical trees from `open_mmlab`, `runs`, or full `prune_distill_exp`.

## Artifact handling

Use Git LFS or a release/artifact store for weights and exported models. The current fixed PT is a compatibility artifact, not a canonical deployable checkpoint for the target YAML.
