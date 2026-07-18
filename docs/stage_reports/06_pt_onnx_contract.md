# Stage 6 Canonical PT/ONNX Contract

- Status: PASSED
- Artifact kind: engineering smoke
- Source: Stage 5 baseline checkpoint
- Output shape: `[1, 9, 8400]`

## Artifact

- PT SHA256: `2bfecdff4022b16ed514d4d6ee4529122742fbc4062c480b842c26eaf1925f4a`
- ONNX SHA256: `9a97ef15ce8ef30d45179348f243de698ae93e873a14c42bec21e197b28fbddf`
- Model YAML SHA256: `32c400d0a2ee4d6619eaeeabf638a1bb5ba2bd6682affe5cca1577af6c56b4e2`
- Parameters/layers: 2,308,655 / 25
- Opset: 12

## Verification

- The known historical CASE_C checkpoint hash is rejected before loading.
- PT loads without legacy modules and matches the target topology signature.
- ONNX checker passes.
- ONNX Runtime CPU output shape matches PyTorch.
- Box error: max 0.0244141, mean 0.000689476; limits 0.05 / 0.001.
- Score error: max 0.0000808239, mean 0.000000733124; limits 0.001 / 0.0001.
- Artifact manifest contains only relative filenames and records preprocessing, classes, hashes, runtime, and validation.

## Boundary

The training repository no longer contains RKNN conversion behavior. `331_PC_RKNN` receives the validated manifest bundle and owns ONNX-to-RKNN conversion and board validation.

## Limitation

This artifact proves the export contract using a one-epoch smoke model. It is not a converged deployment candidate.

## Gate

The topology-compatible PT-to-ONNX engineering path passes. Stage 7 may proceed to repository-wide final acceptance.
