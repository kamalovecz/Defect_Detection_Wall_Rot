# Stage 2 One-Way Registration

- Status: PASSED
- Public installer: `defect_modules.integration.install()`
- Runtime registry: `ultralytics.nn.extensions`

## Verification

- Vendor source contains no import or textual reference to `defect_modules`.
- Building the target YAML before installation fails on `CSPStage` at YAML layer 13.
- Installation registers `CSPStage` and `RepHFE`; repeated installation is idempotent.
- The target model builds with `CSPStage=4` and `RepHFE=2` from `defect_modules.blocks`.
- No `ultralytics.nn.extra_modules` module is loaded.
- Registry, task import boundary, external block, and whitespace checks pass.

## Integration rules

- `CSPStage`: inject input/output channels and pass scaled YAML repeat count to the module constructor.
- `RepHFE`: inject input/output channels and keep YAML repeat handling external.
- Unknown project tokens fail with the token name and YAML layer index.

## Residual risk

The compatibility `patch.apply()` wrapper still performs the historical loss monkey patch. It remains only to keep existing checks operational and will be removed from the main path during Stage 3.

## Gate

The model-module dependency direction is one-way and all Stage 2 checks pass. Stage 3 may proceed.
