# HARP-Net 训练工程 Release Notes

## Ablation Configuration Integration v1.1

> 分支：`codex/ablation-config-integration`
>
> 基线：`codex/decouple-training-v1@7c6a7a4`
>
> 验证环境：Python 3.10.20 / PyTorch 2.1.2 / CUDA 12.1 / Ultralytics 8.2.50 / RTX 4090
>
> 验收日期：2026-07-19

本版本将历史消融 YAML 固定为可追溯的只读归档，并建立 A0～B5 与 L1 的 canonical 模型、统一训练配置、真实数据 smoke、checkpoint 合同和对抗性门禁。

### 主要改动

- 原样归档六份历史 YAML，记录来源、字节数、SHA-256 和 Git blob；活动 B4 与归档 B4 保持逐字节一致。
- 建立 A0、B1、B2、B3、B4、B5 六份 canonical 模型；统一 `nc=5`、scale 与 parser 语义。
- 将 `Detect_LSCSBD` 最小迁移到 `defect_modules.sadh`，通过通用 extension registry 注册，不恢复 `extra_modules`。
- 清理 vendor parser 中未使用的 Fusion/legacy 分支；未知 token 与已移除特性快速失败。
- 新增相对 `extends` 深合并配置，六结构实验仅允许修改 model/name；L1 仅相对 B5 开启 RuleLoss。
- 显式固定 optimizer、学习率、warmup、loss 权重、增强、验证、seed、scratch 和 resume 契约。
- 新增 `Port_Defect` 仓库相对数据描述符、清单指纹、内容指纹和跨 split 重复检测。
- 新增六模型总门禁：CPU/CUDA 前向、原生 loss、分类/回归/骨干梯度、模型签名、模块来源和 legacy 拒绝。
- 新增真实数据 smoke 编排：A0～B5 各 1 epoch，L1 2 epochs；每个实验结束后立即验证。
- 修复 manifest 预建运行目录导致 Ultralytics 自动追加编号的问题，改为使用 trainer 最终 `save_dir`。
- run manifest 使用仓库相对路径，并记录 model/data/effective-config 哈希、Git commit 和 dirty 状态。
- 每个 smoke 产物增加 `smoke_contract.json`，绑定实验、配置、模型、数据内容、override、run 名、commit 和 checkpoint SHA。
- runner 强制 Git clean；dirty worktree、跨实验冒用、合同篡改和 checkpoint 字节篡改均会失败。
- 增加最终工程验收入口、消融复制指南、阶段 0～6 报告和本 Release Notes。

### 消融矩阵

| ID | A-GFPN | RepHFE | SADH | RuleLoss | 定位 |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 | × | × | × | × | YOLOv8n 外部基线 |
| B1 | × | × | ✓ | × | SADH 独立结构点 |
| B2 | × | ✓ | × | × | RepHFE 候选 |
| B3 | ✓ | × | × | × | A-GFPN 候选 |
| B4 | ✓ | ✓ | × | × | A-GFPN + RepHFE |
| B5 | ✓ | ✓ | ✓ | × | 完整结构 |
| L1 | ✓ | ✓ | ✓ | ✓ | B5 上的 RuleLoss 对照 |

合法单因素对比为 B4-B2、B4-B3、B5-B4 和 L1-B5。A0 与 B1 不能被解释为严格单因素对比。

### 验收结果

| 检查项 | 结果 |
| --- | --- |
| 六份历史 YAML 逐字节归档 | PASSED |
| A0～B5 构建、CPU/CUDA 前向 | PASSED |
| 六模型原生 loss 与分类/回归/骨干反向 | PASSED |
| B1/B5 SADH exporter 输出契约 | PASSED |
| 六结构公平配置与 RuleLoss 隔离 | PASSED |
| A0～B5 真实数据 1 epoch smoke | PASSED |
| L1 真实数据 2 epoch、`lambda_rule=1.0` | PASSED |
| 七份 checkpoint strict YAML 重载 | PASSED |
| 空标签 batch loss/backward | PASSED |
| Git/manifest/contract/checkpoint 证据链 | PASSED |
| CASE_C 与 legacy 隔离 | PASSED |
| 既有 PT/ONNX 一致性链 | PASSED |
| 正式消融数据门禁 | **BLOCKED** |

正式消融被阻断的原因是当前 train/val/test 中存在 26 组图像内容完全重复。现有结果仅证明工程 smoke 可运行，不得用于模型排名、精度提升结论或论文消融表。

### 关键提交

| Commit | 阶段 | 内容 |
| --- | --- | --- |
| `b77ef48` | 0 | 可回退基线 |
| `61c4920` | 1 | 历史 YAML 只读归档 |
| `4fc548c` | 2 | A0/B2/B3/B4 canonical 模型 |
| `2138983` | 3 | SADH 迁移与 B1/B5 |
| `74900d6` | 3 修复 | B4 Git blob 完整性 |
| `908c9a2` | 4 | 公平配置与数据描述符 |
| `d8f947b` | 5 | 六模型自动化总门禁 |
| `1f956c3` | 6 | 真实数据 smoke 工作流 |
| `b72bcd9` | 6 修复 | clean Git 与不可变证据合同 |
| `d276e7e` | 6 报告 | 工程通过/正式阻断分级 |

### 使用入口

```powershell
conda activate harpnet_acceptance
python training_project/verify_all.py
python training_project/ablations/verify_ablation_models.py --require-cuda
```

完整说明见 `README.md`、`docs/ABLATION_GUIDE.md` 与 `docs/ablation_reports/`。

## Decoupled Training v1

基线版本完成了训练工程与历史代码的解耦：建立单向 module/criterion registry、配置驱动训练、RuleLoss factory、CASE_C 隔离、Ultralytics 精简、真实数据 baseline/RuleLoss smoke，以及 canonical PT/ONNX 一致性验证。RKNN 转换与板端验证由 `331_PC_RKNN` 仓库负责。

### Breaking changes

- 活动数据集名称统一为 `Port_Defect`；`DAD030` 仅保留在历史诊断记录。
- 不再支持 `defect_modules.patch` monkey patch；使用配置和 criterion factory。
- 自定义模型必须先通过 `defect_modules.integration.install()` 注册。
- `ultralytics.nn.extra_modules` 不属于主运行时；依赖其 pickle 路径的旧 PT 只能诊断，不能作为 canonical 输入。
- 本仓库不提供 ONNX 到 RKNN 的转换入口。
