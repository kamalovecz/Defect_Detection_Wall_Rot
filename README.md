# HARP-Net Defect Detection Research Project

This repository is a Git-ready staging snapshot of the current HARP-Net defect detection project.
It is organized for continued model research, especially new modules under `defect_modules`, while keeping generated artifacts outside the source repository.

## Included source

- `defect_modules/`: custom blocks, loss bridge, registry, and Ultralytics patching.
- `training_project/`: model YAML, unified training entrypoint, and verification scripts.
- `export_pipeline/`: canonical model loading and ONNX/RKNN export utilities.
- `ultralytics-main/`: minimal modified Ultralytics source package needed by this project.
- `prune_distill_exp/.../extra_modules/{prune_module.py,block.py}`: minimal lazy pickle compatibility path for the fixed legacy PT only.
- `docs/`: repository policy and current validation truth.

## Not included

The staging directory intentionally excludes datasets, `runs/`, `.pt/.pth` weights, ONNX/RKNN/engine exports, Python caches, and large historical experiment trees.

## Why Ultralytics source is included

Yes, this project currently needs the modified Ultralytics source. The HARP-Net boundary change lives in `ultralytics-main/ultralytics/nn/tasks.py`, where broad `extra_modules` wildcard import was removed and YAML token resolution was narrowed to official modules plus `defect_modules.registry.YAML_BLOCKS`.

Do not upload or depend on the protected sibling directory `D:\defect_detection\ultralytics`; it is not part of this staging snapshot.

## Quick checks

Run from the repository root in the `yolo_ultra` environment:

```powershell
python training_project\verify_registry.py
python training_project\verify_tasks_import_boundary.py
python training_project\verify_external_blocks.py
```

Training requires a real dataset config:

```powershell
python training_project\train.py --data C:\path\to\dataset.yaml --epochs 1 --batch 4 --imgsz 640 --workers 0
```

Fixed PT compatibility validation requires the external `DAD030_best_target.pt` artifact at `training_project\weights\DAD030_best_target.pt`.

## Current truth

See `docs\VALIDATION_STATUS.md` before claiming export or loss results. In the current snapshot RuleLoss is present but not active, the fixed PT is CASE_C against the target YAML, and ONNX export is skipped because no canonical state_dict exists.
