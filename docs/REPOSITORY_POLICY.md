# Repository Content Policy

## Track in Git

- Source code under `defect_modules`, `training_project`, `export_pipeline`, and the minimal `ultralytics-main` source package.
- Model YAML files.
- Verification scripts.
- Small JSON reports that document current validation status.
- Documentation under `docs`.
- Legacy compatibility source and diagnostic records only under `legacy_compat`.

## Keep out of Git

- Datasets and image/label folders.
- Training runs and logs.
- Weights: `.pt`, `.pth`, `.ckpt`.
- Exports: `.onnx`, `.rknn`, `.engine`, `.torchscript`.
- Python caches and local environment folders.
- Any sibling checkout or machine-specific absolute path.
- Massive historical trees from `open_mmlab`, `runs`, or full `prune_distill_exp`.

## Artifact handling

Use Git LFS or a release/artifact store for weights and exported models. CASE_C
records are diagnostic history only; the active training and export paths reject
that topology.
