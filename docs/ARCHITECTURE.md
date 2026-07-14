# Project Architecture

## Core idea

The project separates research modules from the modified Ultralytics runtime:

- `defect_modules.blocks`: current custom model blocks such as `CSPStage`, `RepDWConv`, and `RepHFE`.
- `defect_modules.loss`: `RuleLoss` bridge and future distillation or hard-sample losses.
- `defect_modules.registry`: explicit YAML-visible blocks, loss objects, pickle compatibility paths, and legacy aliases.
- `defect_modules.patch`: controlled injection into Ultralytics namespaces.
- `training_project.models`: YAML files for model structures and module combinations.
- `training_project.train`: one training entrypoint that applies the patch before constructing YOLO.
- `training_project.verify_*.py`: source, registry, shape, PT loading, and training-path validation scripts.
- `export_pipeline`: canonical loading and export utilities.

## Custom heads

`defect_modules/heads.py` is not implemented in the current snapshot. Keep detection heads inside `blocks.py` or Ultralytics until there is a real custom head family to split out. When custom heads are added, register YAML tokens explicitly in `defect_modules.registry` instead of using wildcard imports.

## Registry policy

`YAML_BLOCKS` is intentionally narrow. For the current HARP-Net main path it only contains:

- `CSPStage`
- `RepHFE`

Do not put historical `extra_modules` names into `YAML_BLOCKS`. Legacy PT pickle compatibility belongs in `PICKLE_COMPAT_TYPES`, not in the YAML registry.

## Ultralytics boundary

`ultralytics-main/ultralytics/nn/tasks.py` must stay free of:

```python
from ultralytics.nn.extra_modules import *
```

YAML token resolution should fail fast for unknown research modules and include the token name plus YAML layer index.
