# HARP-Net legacy dependency candidates

Updated for the tasks.py import-boundary tightening step. This document is a cleanup map only; no legacy source file is deleted in this step.

## Current fixed target

- Model YAML: `D:\defect_detection\training_project\models\B4_A-GFPN_RepHFE_target.yaml`
- Fixed PT: `D:\defect_detection\training_project\weights\DAD030_best_target.pt`
- YAML-visible custom blocks: `defect_modules.blocks.CSPStage`, `defect_modules.blocks.RepHFE`
- Training-side loss bridge: `defect_modules.loss.RuleLossDetectionLoss`

## Boundary changes in this step

- `ultralytics-main\ultralytics\nn\tasks.py` no longer imports `from ultralytics.nn.extra_modules import *`.
- `parse_model()` resolves YAML tokens through `resolve_model_module(name, layer_index)` and supports only the fixed HARP-Net main path plus existing official Ultralytics globals.
- `defect_modules.registry.YAML_BLOCKS` contains only `CSPStage` and `RepHFE`; `RuleLoss` is separated into `LOSS_OBJECTS`.
- `PICKLE_COMPAT_TYPES` records only the old `ultralytics.nn.extra_modules.rephfe` and `ultralytics.nn.extra_modules.prune_module` compatibility paths.
- `LEGACY_ALIASES` keeps `SADH` and `A_GFPN`, but `training_project\train.py` calls `apply(..., legacy_aliases=False, strict=True)` so aliases are opt-in only.

## Keep for now

| Path | Current role | Cleanup condition |
|---|---|---|
| `D:\defect_detection\ultralytics-main\ultralytics\nn\extra_modules\rephfe.py` | Rollback/source reference only. YAML construction must use `defect_modules.blocks.RepHFE`; pickle compatibility maps the old module path to the external class when explicitly enabled. | Delete only after all fixed PT artifacts are resaved without the old pickle path. |
| `D:\defect_detection\prune_distill_exp\ultralytics-prune-20240726\ultralytics-prune\ultralytics\nn\extra_modules\prune_module.py` | Controlled fixed-PT pickle compatibility dependency. It is not a YAML dependency and must not be added to `YAML_BLOCKS`. | Delete only after fixed PT files no longer deserialize objects from this path. |
| `D:\defect_detection\ultralytics-main\ultralytics\nn\extra_modules\__init__.py` | Historical experiment aggregator kept for rollback and old notebooks/scripts. It is no longer imported by the fixed HARP-Net YAML/training path. | Slim or delete only after historical YAML compatibility is formally dropped. |
| `D:\defect_detection\ultralytics-main\ultralytics\nn\extra_modules\block.py` | Historical custom block source kept for rollback and unrelated old YAMLs. | Clean only after no retained artifact references its symbols. |
| `D:\defect_detection\ultralytics-main\ultralytics\utils\loss.py` | Original bundled loss remains as rollback; runtime training uses `defect_modules.loss`. | Clean only after external loss is validated across full training/eval/export flow. |


## Additional import-boundary notes

- `defect_modules.blocks` loads `ultralytics-main\ultralytics\nn\modules\conv.py` by file path for `Conv/RepConv` to avoid the `ultralytics.nn.__init__ -> tasks -> registry` circular import.
- `ultralytics-main\ultralytics\engine\trainer.py` now keeps an inline `get_temperature()` helper instead of importing `ultralytics.nn.extra_modules.kernel_warehouse`, because that import executes the legacy `extra_modules.__init__` aggregator.
- PT pickle compatibility lazily exposes `ultralytics.nn.extra_modules.block` only as `prune_module.py`'s internal import dependency. It is not a YAML dependency and is not recorded in `YAML_BLOCKS`.

## Validation required before cleanup

- `training_project\verify_tasks_import_boundary.py` proves importing `ultralytics.nn.tasks` does not load `ultralytics.nn.extra_modules.*`.
- `training_project\verify_registry.py` proves YAML blocks are only `CSPStage` and `RepHFE`.
- `training_project\verify_external_blocks.py` proves the fixed YAML builds with `CSPStage=4`, `RepHFE=2`, and no `rephfe.py`/`prune_module.py` load.
- `training_project\verify_pt_load.py` proves fixed PT loading works only with explicit pickle compatibility, and `prune_module.py` is triggered during PT deserialization rather than YAML/training patch setup.
