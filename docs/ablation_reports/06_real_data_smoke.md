# 阶段 6：真实数据训练闭环

阶段：6（真实 `Port_Defect` 工程 smoke）

状态：PASSED（仅工程 smoke）

正式消融状态：BLOCKED

修改文件：

- `training_project/train.py`
- `training_project/verify_smoke_checkpoint.py`
- `training_project/ablations/dataset_contract.py`
- `training_project/ablations/training_matrix.yaml`
- `training_project/ablations/run_stage6_smoke.py`

验证命令：

```powershell
python training_project/ablations/dataset_contract.py datasets/Port_Defect --content
python training_project/ablations/run_stage6_smoke.py `
  --project training_project/runs/ablation_smoke_final `
  --run-prefix final
python training_project/ablations/run_stage6_smoke.py `
  --project training_project/runs/ablation_smoke_final `
  --run-prefix final `
  --reuse-completed
python training_project/ablations/verify_ablation_models.py --check-reference-source --require-cuda
python training_project/verify_all.py
git status --porcelain --untracked-files=all
git diff --check
```

通过项：

- 数据通过仓库 `datasets/Port_Defect/` 下的受控 junction 物化，不修改或复制外部源数据。
- Ultralytics 8.2.50 的解析结果与仓库逻辑 train/val/test 目录通过 `samefile()` 验证。
- 数据规模固定为 train/val/test 图像 2428/607/350，标签数量相同；无缺失标签、空标签或非法标签行。
- 数据相对清单、图像内容、标签内容及跨划分重复项均生成 SHA-256 快照。
- A0～B5 在真实数据上各运行 1 epoch；L1 运行 2 epochs，使 paper RuleLoss 在第二轮真实激活。
- 七组训练均完成数据加载、前向、反向、优化器更新、验证和 checkpoint 保存。
- 七组 results 均为有限值，无 NaN/Inf。
- 七组 checkpoint 在新进程中按 manifest YAML 新建模型并 `strict=True` 加载成功。
- 七组固定输入输出均为 `[1, 9, 336]` 且有限，空标签 batch 的 loss/backward 通过。
- L1 criterion 为 `defect_modules.loss.v8DetectionLoss`，最终 `lambda_rule=1.0`。
- 所有 run manifest 只记录仓库相对 model/data/config/project 路径。
- manifest、`smoke_contract.json` 和 checkpoint 相互绑定实验 ID、配置、模型/数据/内容哈希、run name、override、Git commit 与 checkpoint SHA。
- 最终七份证据均由干净提交 `b72bcd996558e3b2c3368c77d16b09856deef5d3` 生成，manifest 为 `git_dirty=false`，contract 为 `git_clean=true`。
- `--reuse-completed` 对最终七份产物完整复验退出 0。
- dirty worktree、A0 冒充 B1、合同字段篡改、checkpoint hash 篡改和真实 checkpoint 副本单字节篡改均被拒绝。
- 未加载 legacy 模块；运行输出目录未污染 Git。
- 独立对抗性审查最终通过工程 smoke 门禁。

失败项：

- 正式消融训练数据门禁未通过：发现 26 组图像内容跨 train/val/test 完全重复。

风险或残留：

- 本阶段只证明训练工程可运行，不提供任何模型优劣或精度提升结论。
- 在重新划分数据并使跨 split 内容重复为 0 之前，禁止启动或引用正式三种子消融实验。
- junction 是当前机器的运行时物化方式；新机器需将同一数据内容映射或复制到 `datasets/Port_Defect/images|labels` 后重算快照。
- checkpoint 内部 Ultralytics `train_args` 仍可能保存本机绝对路径，canonical 交付阶段需要清理或避免将其作为交付元数据。
- 后续可增加 RuleLoss 在真实数据上的实际加权 batch/anchor 计数。

是否允许进入下一阶段：是，仅允许进入工程验收、文档和交付整理；不允许进入正式消融训练。
