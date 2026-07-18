# Runtime source lock

- Ultralytics release baseline: `8.2.50`
- Project-owned extensions: `ultralytics/nn/extensions.py`
- Project patch points: model token resolution, channel/repeat extension specifications, and detection criterion factory lookup in `ultralytics/nn/tasks.py`
- Policy: vendor code must not import `defect_modules`, `training_project`, or `legacy_compat`.

The historical `ultralytics/nn/extra_modules` tree is intentionally excluded from the active runtime.
