# Stage 3 RuleLoss Factory

- Status: PASSED
- Baseline default: native `ultralytics.utils.loss.v8DetectionLoss`
- RuleLoss activation: explicit `loss.rule.enabled: true`

## Verification

- Baseline configuration selects the native Ultralytics criterion.
- Enabled configuration selects `defect_modules.loss.v8DetectionLoss` through the runtime criterion factory.
- A deterministic synthetic target produced 10 foreground anchors with maximum weight 3.0.
- The same predictions produced a finite baseline loss and a different finite RuleLoss value.
- Paper schedule is zero at `t1` and reaches `lambda_max` at `t2`.
- Invalid RuleLoss versions are rejected before training.
- Model registration, import boundary, external model build, registry, and whitespace checks pass.

## Changes

- Added a runtime detection-loss factory without vendor-to-project imports.
- Replaced environment-driven RuleLoss construction with validated project configuration.
- Added an epoch callback in the standard training entrypoint.
- Disabled global replacement of `ultralytics.utils.loss` and `ultralytics.nn.tasks` symbols.

## Gate

RuleLoss is explicit, observable, and isolated from the baseline path. Stage 4 may proceed.
