# Upload Checklist

Before uploading this staging directory to a repository:

1. Review `docs/VALIDATION_STATUS.md` and keep the wording honest.
2. Initialize Git from `D:\defect_detection\repo_staging`.
3. Check that `git status` does not include weights, datasets, runs, ONNX, RKNN, or caches.
4. Keep the included Ultralytics AGPL-3.0 license file with the modified source.
5. Add large artifacts through Git LFS or an external artifact store only if you intentionally want to publish them.
6. Run the three source-boundary checks:

```powershell
python training_project\verify_registry.py
python training_project\verify_tasks_import_boundary.py
python training_project\verify_external_blocks.py
```

7. Do not claim ONNX/RKNN success until a non-CASE_C canonical state_dict is produced and export consistency passes.
