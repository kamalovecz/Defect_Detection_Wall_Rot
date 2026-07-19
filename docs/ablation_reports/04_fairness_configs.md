# 阶段 4：公平训练配置

阶段：4（统一训练配置与公平性契约）

状态：PASSED

修改文件：

- `.gitignore`
- `datasets/Port_Defect/data.yaml`
- `training_project/config.py`
- `training_project/train.py`
- `training_project/configs/port_defect_baseline.yaml`
- `training_project/configs/port_defect_ruleloss_smoke.yaml`
- `training_project/configs/ablations/common.yaml`
- `training_project/configs/ablations/A0.yaml` ～ `B5.yaml`
- `training_project/configs/ablations/L1_B5_RuleLoss.yaml`
- `training_project/ablations/dataset_contract.py`
- `training_project/ablations/training_matrix.yaml`
- `training_project/ablations/verify_stage4_fairness.py`

验证命令：

```powershell
python training_project/ablations/dataset_contract.py D:\defect_detection\ultralytics-main\dataset\processed\processed_dataset
python training_project/ablations/verify_archive.py --check-source --check-git
python training_project/ablations/verify_stage2_models.py
python training_project/ablations/verify_stage3_sadh.py --check-reference-source --require-cuda
python training_project/ablations/verify_stage4_fairness.py
python training_project/verify_all.py
git diff --check
git ls-files --others --exclude-standard datasets\Port_Defect\data.yaml
```

通过项：

- A0～B5 只允许覆盖模型路径和运行名；有效训练配置的其余字段完全一致。
- L1 只相对 B5 开启 RuleLoss 并修改运行名。
- 公共配方显式固定优化器、学习率、warmup、损失权重、增强、验证、随机种子和 scratch 初始化参数。
- 公共配方与数据描述符均由 canonical JSON SHA-256 锁定；公共学习率篡改会被拒绝。
- 固定筛查种子为 42，正式种子为 42、43、44；种子篡改会被拒绝。
- scratch 训练要求 `pretrained: false`、`resume: false` 且模型来源必须为 `.yaml`。
- Ultralytics 8.2.50 的 `get_cfg` 实际接受完整训练参数。
- 仓库包含可跟踪的 `Port_Defect` 数据描述符；不设置 `path`，数据根按 YAML 父目录解析。
- 数据快照记录 train/val/test 图像和标签数量 2428/607/350、缺失标签为 0，以及逐划分与合并清单指纹。
- 配置继承支持相对 `extends`、深合并和循环检测。
- 六份配置均通过真实 `train.py --check-config`，`model_exists` 与 `data_exists` 都为 true。
- 阶段 1～3 门禁、全仓 11 项验证、CUDA 验证和 Git diff 检查无回归。
- 独立对抗性审查在三轮修复后最终通过。

失败项：无。

风险或残留：

- 仓库只保存数据描述符，不保存真实图片和标签；阶段 6 必须将真实数据物化到描述符对应目录并重算快照。
- 当前快照锁定相对文件清单和标签存在性，不锁定文件内容；阶段 6 增加内容哈希和跨划分重复检测。
- 正式三种子长周期训练尚未运行。

是否允许进入下一阶段：是。
