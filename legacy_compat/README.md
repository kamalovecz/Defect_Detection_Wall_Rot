# Legacy checkpoint compatibility

This directory is outside the active training and export paths. It documents and diagnoses the historical CASE_C pruned checkpoint. The active registry and runtime must not import this package.

Run `python legacy_compat/inspect_case_c.py` to validate the retained migration record. Historical pickle source is kept under `legacy_compat/vendor` only for explicit offline inspection of an externally supplied checkpoint.
