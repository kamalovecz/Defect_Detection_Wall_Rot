# Stage 4 Legacy Isolation and Runtime Pruning

- Status: PASSED
- Pre-prune tracked files: 1,720
- Post-prune tracked files: 992
- Post-prune runtime size: 1,043 files / 4.52 MiB
- Active `extra_modules` files: 0

## Verification

- Active registry has no pickle compatibility or legacy alias surface.
- Vendor import and project installation load no legacy modules.
- Target YAML builds with the expected custom module sources and counts.
- Target signature remains 2,308,655 parameters and 25 layers.
- Baseline and RuleLoss factory checks pass after pruning.
- Legacy diagnostic reports `CASE_C` and 8 historical `C2f_v2` instances.
- Active project and export source contains no `defect_modules.patch` import.

## Changes

- Moved the two retained pruning pickle sources under `legacy_compat/vendor`.
- Added a manifest-based CASE_C diagnostic tool.
- Removed the compatibility patch module and obsolete legacy verification scripts from the main training tree.
- Removed 729 tracked `extra_modules` files and the historical loss backup.
- Locked the runtime baseline to Ultralytics 8.2.50 and documented project patch points.

## Recovery

The complete pre-deletion snapshot is commit `4f9c409`. Removed source remains recoverable from Git history.

## Gate

Legacy code is isolated and the active model signature is unchanged. Stage 5 may proceed when the external Port_Defect dataset is available.
