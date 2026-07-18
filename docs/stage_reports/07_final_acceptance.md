# Stage 7 Final Acceptance

阶段：7 - 全新环境最终验收

状态：PASSED

修改文件：

- `README.md` and architecture, environment, repository, upload, and validation documentation.
- `environment.yml` (added the vendor runtime dependency `timm==1.0.15`).
- `ultralytics-main/README.md` (restored the packaging input required by `setup.py`).
- `training_project/verify_all.py`, `verify_environment.py`, and `verify_blocks.py`.
- CASE_C diagnostic records moved from active training paths to `legacy_compat/records`.
- ONNX export rejects the known CASE_C hash without loading a legacy manifest.
- Removed obsolete root checksum inventories that referenced deleted files.

验证命令：

```text
conda env create -n harpnet_acceptance -f environment.yml
D:\Anaconda\envs\harpnet_acceptance\python.exe training_project\verify_all.py
D:\Anaconda\envs\harpnet_acceptance\python.exe -m training_project.train --config ... --data <external Port_Defect yaml> --name stage7_clean_env --epochs 1
D:\Anaconda\envs\harpnet_acceptance\python.exe training_project\verify_smoke_checkpoint.py ...
D:\Anaconda\envs\harpnet_acceptance\python.exe export_pipeline\verify_onnx_consistency.py --manifest ...
git diff --check
git grep defect_modules -- ultralytics-main/ultralytics
git ls-files ultralytics-main/ultralytics/nn/extra_modules
```

通过项：

- Clean environment creation and editable vendor installation.
- Python 3.10.20, PyTorch 2.1.2/CUDA 12.1, CUDA available, Ultralytics 8.2.50.
- Consolidated acceptance suite: 11/11 checks passed.
- CSPStage and RepHFE shapes; model signature 2,308,655 parameters and 25 layers.
- Native baseline and configured RuleLoss criterion factory behavior.
- Real Port_Defect one-epoch clean-environment baseline training and validation.
- Finite metrics, checkpoint save, strict reload, fixed-input inference, and no legacy imports.
- ONNX checker and PT/ONNX numerical comparison within the artifact thresholds.
- Vendor-to-project reverse dependency absent; tracked `extra_modules` absent.

失败项：无（修复后完整重跑通过）。首次环境创建暴露缺失 vendor README，首次导入暴露未声明的 `timm`；两项均已加入仓库契约并从空环境重建验证。

风险或残留：现有训练与 ONNX 均为工程 smoke 产物，不代表正式收敛精度。RKNN 转换和板端验证仍由部署仓库负责。真实数据保留在仓库外部，只读用于验收。

是否允许进入下一阶段：是；阶段 0-7 工程验收完成。正式长周期训练和指标调优应作为后续独立工作开展。
