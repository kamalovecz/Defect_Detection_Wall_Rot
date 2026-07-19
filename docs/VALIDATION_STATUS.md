# Current Validation Status

Validated on 2026-07-19 in `harpnet_acceptance` using Python 3.10.20, PyTorch 2.1.2/CUDA 12.1, Ultralytics 8.2.50, and an RTX 4090.

## Engineering acceptance: PASSED

- One-way module registration and vendor import boundary: passed.
- Canonical A0-B5 build/forward/backward/CUDA gate: passed.
- CSPStage, RepHFE and Detect_LSCSBD originate from `defect_modules`.
- Native baseline criterion and active RuleLoss criterion: passed.
- A0-B5 real-data 1-epoch smoke: passed.
- L1 real-data 2-epoch smoke: passed; `lambda_rule=1.0`.
- Seven checkpoint strict YAML reloads, fixed inference and empty-label backward: passed.
- Manifest/contract/checkpoint hashes and clean Git evidence: passed.
- Dirty worktree, cross-experiment reuse and artifact tampering rejection: passed.
- Known CASE_C rejection and existing PT/ONNX numerical consistency: passed.
- Active legacy imports: none.

## Formal ablation training: BLOCKED

The current dataset contains 26 exact image-content duplicates across train, validation and test splits. The dataset contract records `formal_training_eligible: false`. Engineering smoke results prove only that the pipeline runs; they must not be used for model ranking, accuracy claims or publication tables.

Formal training may begin only after rebuilding the splits, obtaining zero cross-split content duplicates, updating the reviewed dataset hashes, and rerunning the acceptance gates.

RKNN conversion and board validation remain outside this repository and belong to `331_PC_RKNN`.
