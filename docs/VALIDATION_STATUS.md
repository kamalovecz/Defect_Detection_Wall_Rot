# Current Validation Status

Validated on 2026-07-18 in a newly created environment using Python 3.10.20,
PyTorch 2.1.2/CUDA 12.1, Ultralytics 8.2.50, and an RTX 4090.

- One-way module registration: passed.
- Target signature: 2,308,655 parameters and 25 layers.
- Custom modules: CSPStage=4 and RepHFE=2 from `defect_modules.blocks`.
- Active legacy imports: none.
- Native baseline criterion: passed.
- Active RuleLoss criterion: passed; real-data smoke recorded 607 updates and lambda 1.0.
- Baseline and RuleLoss one-epoch Port_Defect smoke training: passed.
- Clean-environment baseline one-epoch smoke training: passed.
- Checkpoint reload and inference: passed.
- Known CASE_C rejection: passed.
- ONNX checker and ONNX Runtime consistency: passed.
- RKNN conversion: intentionally outside this repository.

The validated checkpoints and ONNX are engineering smoke artifacts, not
converged deployment candidates. Formal training and accuracy claims remain
outstanding.
