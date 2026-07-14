# Current Validation Status

Generated from the remote working tree on 2026-07-14.

## Boundary checks that passed

- `verify_registry.py`: passed. `YAML_BLOCKS` only exposes `CSPStage` and `RepHFE`; `prune_module` is not a YAML dependency.
- `verify_tasks_import_boundary.py`: passed. Importing `ultralytics.nn.tasks` no longer loads `ultralytics.nn.extra_modules.rephfe` or `prune_module`.
- `verify_external_blocks.py`: passed. The target YAML builds with `CSPStage=4` and `RepHFE=2`, both from `defect_modules.blocks`.
- `verify_pt_load.py`: passed in the original working tree only when `apply(pickle_compat=True, legacy_aliases=False, strict=True)` is used and the fixed PT is present.

## RuleLoss truth

- `RuleLoss` exists in `defect_modules.loss` and is exposed through `defect_modules.registry.LOSS_OBJECTS`.
- Current `training_project/train.py` calls `apply(pickle_compat=False, legacy_aliases=False, strict=True)`.
- The latest report says `RULELOSS_NOT_ACTIVE`.
- Recorded call stats: `rule_weight_builder_call_count=2`, `rule_active_call_count=0`, `ruleloss_changed_base_loss=false`.
- Practical meaning: do not claim RuleLoss improved training yet. It is wired as a loss object/bridge, but the current training path did not activate rule weighting because `RULE_LOSS_ENABLE` was not enabled.

## CASE_C truth

- The fixed PT `DAD030_best_target.pt` is a legacy/pruned checkpoint whose model topology uses `ultralytics.nn.extra_modules.prune_module.C2f_v2`.
- The dependency inspection found 8 `C2f_v2` instances from `prune_module.py`.
- Source PT parameter count: 7,927,270.
- Target YAML parameter count: 2,308,655.
- Shape mismatch count: 119.
- Layer diff count: 25.
- Practical meaning: this is CASE_C. A strict canonical state_dict for `B4_A-GFPN_RepHFE_target.yaml` was not produced from the fixed PT.

## ONNX truth

- No canonical state_dict exists in this snapshot.
- `export_pipeline/export_onnx.py` returns `SKIPPED_CASE_C` for the current manifest.
- ONNX path: null.
- ONNX SHA256: null.
- Consistency verification status: `SKIPPED_NO_ONNX`.
- Practical meaning: do not upload or claim a new ONNX/RKNN artifact from this stage.

## Protected directory truth

- `D:\defect_detection\ultralytics` had external drift during the previous validation stage.
- This staging snapshot does not copy that protected directory and should not depend on it.

## Status wording

Use this wording for the current repository:

`Repository staging: PASSED. Functional canonical ONNX validation: FAILED/SKIPPED because the fixed PT is CASE_C and RuleLoss is not active in the current training path.`
