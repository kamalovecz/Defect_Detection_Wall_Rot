# Upload Checklist

Before uploading this staging directory to a repository:

1. Review `docs/VALIDATION_STATUS.md` and keep the wording honest.
2. Work on a reviewable feature branch and keep `main` unchanged.
3. Check that `git status` does not include weights, datasets, runs, ONNX, RKNN, or caches.
4. Keep the included Ultralytics AGPL-3.0 license file with the modified source.
5. Add large artifacts through Git LFS or an external artifact store only if you intentionally want to publish them.
6. Run the consolidated acceptance checks:

```powershell
python training_project\verify_all.py
python training_project\ablations\verify_ablation_models.py --require-cuda
python training_project\ablations\verify_final_acceptance.py `
  --require-real-data `
  --smoke-state training_project/runs/ablation_smoke_final/stage6_state.json `
  --onnx-manifest export_pipeline/outputs/port_defect_smoke/artifact_manifest.json
```

7. Validate any delivered ONNX with `verify_onnx_consistency.py` and its
   artifact manifest.
8. Do not include RKNN conversion code here; hand the validated ONNX and
   manifest to the deployment repository.
9. Do not publish formal ablation metrics while `training_matrix.yaml` records
   `formal_training_eligible: false`; rebuild the dataset splits and review the
   new content fingerprint first.
