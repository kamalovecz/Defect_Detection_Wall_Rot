# HARP-Net Defect Detection Research Project

This repository is a Git-ready source baseline for the HARP-Net defect detection project.
It is organized for continued model research: new structure blocks, losses, detection heads, model YAMLs, training entrypoints, verification scripts, and export utilities are kept in project-level modules instead of being scattered through the Ultralytics source tree.

## Included Source

- `defect_modules/`: custom blocks, loss bridge, registry, and Ultralytics patching.
- `training_project/`: model YAML, unified training entrypoint, dataset config examples, and verification scripts.
- `export_pipeline/`: canonical model loading and ONNX/RKNN export utilities.
- `ultralytics-main/`: minimal modified Ultralytics source package required by this project.
- `prune_distill_exp/.../extra_modules/{prune_module.py,block.py}`: minimal lazy pickle compatibility path for the fixed legacy PT only.
- `docs/`: repository policy, environment notes, and current validation truth.

## Not Included

The repository intentionally excludes datasets, `runs/`, `.pt/.pth` weights, ONNX/RKNN/engine exports, Python caches, and large historical experiment trees.

Datasets will be distributed through Baidu Netdisk. Stage checkpoints and other `.pt` artifacts should be published through GitHub Releases or another artifact store, not committed into Git.

## Why Ultralytics Source Is Included

This project currently needs the modified Ultralytics source. The HARP-Net boundary change lives in `ultralytics-main/ultralytics/nn/tasks.py`, where the broad `extra_modules` wildcard import was removed and YAML token resolution was narrowed to official modules plus `defect_modules.registry.YAML_BLOCKS`.

Do not upload or depend on the protected sibling directory `D:\defect_detection\ultralytics`; it is not part of this repository snapshot.

## Environment

Use the `yolo_ultra` environment, or create an equivalent Python 3.10 environment from `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate yolo_ultra
```

Run commands from the repository root.

## Dataset Setup

The dataset is not stored in Git. After the Baidu Netdisk dataset link is provided, download and extract the dataset outside this repository, for example:

```text
D:\datasets\DAD030_processed_dataset
```

Copy or edit the example config:

```powershell
copy training_project\datasets\DAD030_dataset.example.yaml training_project\datasets\DAD030_dataset.local.yaml
```

Then update `path` in `training_project\datasets\DAD030_dataset.local.yaml`:

```yaml
path: D:\datasets\DAD030_processed_dataset
train: images/train
val: images/val
test: images/test
names:
  0: Rust
  1: Cracks
  2: Paint Wear
  3: Scratches
  4: Pitting
```

Use the local dataset YAML explicitly when training:

```powershell
python training_project\train.py `
  --data training_project\datasets\DAD030_dataset.local.yaml `
  --epochs 1 `
  --batch 4 `
  --imgsz 640 `
  --workers 0 `
  --name smoke_dataset_check
```

## Custom Module Training Flow

This project is designed as an extensible custom-module framework. New modules should follow this chain:

```text
defect_modules/blocks.py
        ->
defect_modules/registry.py
        ->
defect_modules/patch.py
        ->
training_project/models/*.yaml
        ->
training_project/train.py
        ->
Ultralytics parse_model
        ->
model training
```

### 1. Implement The Module

Place structure modules in `defect_modules/blocks.py`. A simple single-input, single-output block usually follows this form:

```python
import torch.nn as nn


class NewHFEBlock(nn.Module):
    def __init__(self, c1, c2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

For ordinary channel-aware blocks, the expected tensor contract is:

```text
input:  B x C1 x H x W
output: B x C2 x H x W
```

### 2. Register The YAML Token

Expose only real YAML-visible modules in `defect_modules/registry.py`:

```python
from defect_modules.blocks import CSPStage, RepHFE, NewHFEBlock

YAML_BLOCKS = {
    "CSPStage": CSPStage,
    "RepHFE": RepHFE,
    "NewHFEBlock": NewHFEBlock,
}
```

Do not add historical `extra_modules` names to `YAML_BLOCKS`. Legacy PT compatibility belongs in `PICKLE_COMPAT_TYPES`.

### 3. Confirm Patch Injection

The training entrypoint applies the project patch before constructing YOLO:

```python
from defect_modules.patch import apply

apply(
    pickle_compat=False,
    legacy_aliases=False,
    strict=True,
)
```

This injects the registered project modules into the Ultralytics namespaces used during model construction.

### 4. Use The Module In YAML

Create a new model YAML under `training_project/models/`:

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, NewHFEBlock, [128, 3]]
  - [-1, 2, CSPStage, [128]]
  - [-1, 1, RepHFE, [128]]
```

For `NewHFEBlock`, YAML provides `c2=128` and `kernel_size=3`. `parse_model()` must supply `c1` from the previous layer so the final constructor call becomes:

```python
NewHFEBlock(c1, 128, 3)
```

If a new module changes parameter parsing rules, update the registry/parse-model boundary deliberately instead of restoring wildcard imports.

Common cases that may need explicit parsing support:

- automatic input channel `c1`
- automatic output channel `c2`
- internal repeat count `n`
- multiple input indexes
- `Concat` or `Add` style multi-input behavior
- multiple output feature maps
- special width/depth scaling
- detection head input channel lists

### 5. Train Through The Unified Entrypoint

```powershell
python training_project\train.py `
  --model training_project\models\new_harpnet.yaml `
  --data training_project\datasets\DAD030_dataset.local.yaml `
  --epochs 100 `
  --batch 16 `
  --imgsz 640 `
  --workers 0 `
  --name new_harpnet_exp01
```

Keep training outputs under `training_project\runs`. They are ignored by Git.

## Losses And Heads

Structure modules, feature-fusion modules, backbone modules, and future custom detection heads are YAML components and should enter through `YAML_BLOCKS`.

Losses, distillation objectives, pruning constraints, and hard-sample weighting are training components. Registering a loss in `LOSS_OBJECTS` only makes the object discoverable; it does not automatically change training behavior. The loss must also be called by the trainer or criterion, such as `v8DetectionLoss` or a project-specific trainer.

Recommended ownership:

- `defect_modules/blocks.py`: convolution, CSP, attention, high-frequency enhancement, and feature-fusion blocks.
- `defect_modules/heads.py`: future custom detection heads when they become large enough to split from `blocks.py`.
- `defect_modules/loss.py`: `RuleLoss`, distillation losses, and hard-sample weighting losses.
- `defect_modules/registry.py`: model tokens, loss objects, parameter parsing groups, and compatibility mappings.
- `defect_modules/patch.py`: Ultralytics namespace injection and legacy PT compatibility.
- `training_project/models`: YAML files for different module combinations.
- `training_project/train.py`: unified training entrypoint.
- `training_project/verify_*.py`: source, shape, registry, PT loading, and training validations.

## Required Validation For A New Module

At minimum, complete these checks before treating a new module as trainable.

1. Module shape check:

```python
import torch
from defect_modules.blocks import NewHFEBlock

x = torch.randn(1, 64, 80, 80)
module = NewHFEBlock(64, 128)
y = module(x)
assert y.shape == (1, 128, 80, 80)
```

2. Registry and patch check:

```python
from defect_modules.patch import apply

apply(pickle_compat=False, legacy_aliases=False, strict=True)

import ultralytics.nn.tasks as tasks

assert tasks.NewHFEBlock.__module__ == "defect_modules.blocks"
```

3. YAML build check:

```python
from ultralytics import YOLO

model = YOLO("training_project/models/new_harpnet.yaml")
```

4. Smoke training check:

```powershell
python training_project\train.py `
  --model training_project\models\new_harpnet.yaml `
  --data training_project\datasets\DAD030_dataset.local.yaml `
  --epochs 1 `
  --batch 4 `
  --imgsz 640 `
  --workers 0 `
  --name new_module_smoke
```

Confirm that loss is finite, output is written to `training_project\runs`, and the custom module source is `defect_modules.blocks`.

## Export Constraints

If a module must eventually deploy to RK3588S/RKNN, prefer conservative operators during design:

- `Conv2d`
- depthwise `Conv2d`
- `BatchNorm2d`
- `SiLU` or `ReLU`
- `Add`
- `Concat`
- `Interpolate`
- `MaxPool`
- ordinary `reshape` and `permute`

Use caution with:

- dynamic Python control flow depending on tensor values
- custom CUDA operators
- complex `einsum`
- dynamic indexing
- dynamically generated convolution kernels
- irregular `grid_sample`
- operators unsupported by ONNX or RKNN

## Model Artifacts And Releases

Do not commit `.pt`, `.pth`, ONNX, RKNN, engine, or TorchScript artifacts to Git.

When stage checkpoints are ready, publish them as GitHub Release assets. Recommended naming:

```text
harpnet-baseline-b4-epochXXX.pt
harpnet-cspstage-rephfe-b4-epochXXX.pt
harpnet-ruleloss-b4-epochXXX.pt
harpnet-deploy-candidate-b4-epochXXX.pt
```

Each release should include:

- checkpoint `.pt` files
- matching model YAML
- dataset version or Baidu Netdisk dataset identifier
- training command
- metrics summary
- SHA256 checksums
- known limitations, especially whether RuleLoss was active and whether ONNX export passed

Suggested release tags:

```text
weights-vYYYYMMDD
smoke-vYYYYMMDD
deploy-candidate-vYYYYMMDD
```

The current repository commit does not include release assets. Add them only after the dataset and stage PT files are finalized.

## Quick Checks

Run from the repository root in the `yolo_ultra` environment:

```powershell
python training_project\verify_registry.py
python training_project\verify_tasks_import_boundary.py
python training_project\verify_external_blocks.py
```

Fixed PT compatibility validation requires the external `DAD030_best_target.pt` artifact at:

```text
training_project\weights\DAD030_best_target.pt
```

## Current Truth

See `docs\VALIDATION_STATUS.md` before claiming export or loss results. In the current snapshot:

- `RuleLoss` is present but not active in the current training path.
- The fixed PT is `CASE_C` against the target YAML.
- ONNX export is skipped because no canonical state_dict exists.
- Datasets and stage checkpoints are intentionally external to Git.
