# B5 与 RuleLoss 单变量对照实验计划

## 1. 任务理解

本实验用于判断：在 B5 Full 模型、Port_Defect 数据集和既定训练预算完全不变时，显式启用 RuleLoss 是否改善缺陷检测性能。

对照组使用已经完成的 `B5_seed42_e300_b8`；实验组记为 `L1_B5_RuleLoss_seed42_e300_b8`。本轮是 seed 42 筛选实验，不直接产生跨随机种子的稳定性结论。

## 2. 核心假设

- 原假设 H0：启用 RuleLoss 后，主要指标 mAP50-95 没有改善。
- 研究假设 H1：启用 RuleLoss 后，主要指标 mAP50-95 高于同配置 B5。
- 次要观察：mAP50、Precision、Recall、F1、mAP75 与五类缺陷 AP。
- 效率观察：参数量和 FLOPs 理论上不应变化；训练耗时可能因 RuleLoss 计算增加，推理耗时不应因训练损失而产生结构性变化。

## 3. 冻结的公平性合同

| 项目 | B5 对照组 | L1 实验组 |
|---|---|---|
| 模型 YAML | `training_project/models/ablations/B5_Full.yaml` | 相同 |
| 初始化 | scratch，`pretrained=false` | 相同 |
| 数据 | `datasets/Port_Defect/data.yaml` | 相同 |
| 数据划分 | train/val/test 合同不变 | 相同 |
| epochs | 300 | 300 |
| batch | 8 | 8 |
| imgsz | 640 | 640 |
| seed | 42 | 42 |
| optimizer | SGD | SGD |
| lr0 / lrf | 0.001 / 0.00001 | 相同 |
| AMP | false | false |
| plots | true | true |
| RuleLoss | false | **true** |

除 RuleLoss 开关、实验名称和输出目录外，训练有效配置必须与已完成的 B5 运行清单一致。

## 4. RuleLoss 冻结参数

```yaml
version: paper
small_area: 1024.0
gamma_small: 0.06
gamma_contrast: 0.04
low_contrast_std: 0.12
lambda_max: 1.0
schedule_iters: 12000
stage0_ratio: 0.20
stage1_ratio: 0.60
total_epochs: 300
t1_epoch: -1
t2_epoch: -1
```

本轮不得依据 B5 的 val/test 结果调整上述参数，否则不再是预先固定的单变量验证。

## 5. 分阶段执行与门禁

### 阶段 A：配置与损失函数门禁

1. 解析 L1 专用配置并确认数据与模型路径存在。
2. 将其有效配置与 B5 已完成的 `run_manifest.json` 比较。
3. 差异白名单只能包含 RuleLoss 开关、配置路径、实验名称和输出目录。
4. 验证 criterion 来源为项目 RuleLoss，anchor weight 实际产生，loss 相对原生 criterion 发生有限变化。

只有上述检查全部通过，才允许启动正式训练。

### 阶段 B：300 epochs 训练

运行名称：`L1_B5_RuleLoss_seed42_e300_b8`。

训练完成条件：

- `results.csv` 恰好包含 300 个 epoch；
- loss 全部有限，无 NaN/Inf；
- `best.pt`、`last.pt`、`run_manifest.json` 和 plots 完整；
- manifest 记录 RuleLoss 已启用、criterion 来源正确、`rule_updates > 0`；
- checkpoint 可在新进程严格加载并完成固定样本推理。

### 阶段 C：独立 val 评估

使用现有外部 `D:\defect_detection\ultralytics-main\val.py`：`split=val`、`imgsz=640`、`batch=8`。首先比较 L1 与冻结的 B5 val 结果，主要差值为 `L1 − B5` 的 mAP50-95。

### 阶段 D：test 评估

在查看 L1 test 结果前，先将 val 结论、候选身份、checkpoint SHA 和评价规则写入不可变 Git 提交。随后使用相同外部脚本执行 `split=test`、`imgsz=640`、`batch=8`。test 只用于独立测量，不用于回调 RuleLoss 参数。

### 阶段 E：报告与审查

报告总体指标、逐类别 AP、复杂度、速度、训练耗时和 RuleLoss 运行状态。所有结论限定为 seed 42；若要写成稳定有效性结论，需对 B5 与 L1 成对补跑 seed 43、44，并报告 mean ± std。

## 6. 直接运行配置

配置文件：

```text
training_project/configs/ablations/L1_B5_RuleLoss_seed42_e300_b8.yaml
```

门禁通过后的训练命令：

```powershell
D:\Anaconda\envs\harpnet_acceptance\python.exe training_project\train.py `
  --config training_project\configs\ablations\L1_B5_RuleLoss_seed42_e300_b8.yaml `
  --no-exist-ok
```

本计划生成阶段不启动长周期训练；训练应在配置提交、工作区干净且 GPU 空闲后单独挂载。
