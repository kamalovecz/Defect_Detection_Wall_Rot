# HARP-Net 六结构与 RuleLoss Test 数据集检测报告

## 1. 报告用途

本报告记录六个结构配置及一个 RuleLoss 配置的已锁定 `best.pt` 在 Port_Defect **test split** 上的一次性评估结果。模型训练与 checkpoint 选择已经在 train/val 流程中完成，本次 test 评估不参与调参、早停、checkpoint 选择或模型重训。

B1（精度候选）、B3（效率候选）和 B5（F1/召回平衡候选）的身份，以及补充 B4 与多 seed 的计划，已经在查看 test 结果前由 val 报告 `ablation_300ep_b8_20260719_summary.md` 锁定。不可变证据定位为 Git commit `61214a234c31759faacbba9f3ec22873ae2f56de` 中的该文件，其内容 SHA-256 为 `cc232aee80e0a8ae29bb64124ba0e1882b8d9bd128bcaf6539044692934f2c97`；当前工作树中的同名文件是加入 B4 后的新版。B4 的模型 YAML 与实验配置分别在首次 test 前由 commit `74900d6290a1d34f2966176265cffcddc1a016fa` 和 `908c9a2e4778fd59f1c6cef36be68a7803136a68` 锁定，SHA-256 分别为 `32c400d0a2ee4d6619eaeeabf638a1bb5ba2bd6682affe5cca1577af6c56b4e2` 和 `d2e7b5aebc5eff86052064dc52efdb823f10c1926195945db86d14c2a61428cf`。B4 未根据前五组 test 结果修改结构或超参数，并在其 best checkpoint 锁定后才执行自身 test。本次 test 结果只用于验证这些预先锁定的方案，不用于改变候选集合、训练配置或后续实验范围。

L1 的 RuleLoss 参数和单变量配置由 commit `489eacd4fc25f6c81804484d18aad202c519fa9d` 锁定。L1 完成独立 val 后，结论在查看 L1 test 结果前由 commit `761143650b2d879ba28da501e5429a0153988df8` 锁定；该 commit 中 val 报告 SHA-256 为 `4da0c91855e6c018b5e7b1f51fdb54f156d0d4f2f7f559f36d42cd75e6ccfeb8`，预先结论为 L1−B5 的 val mAP50-95 等于 -0.0108。L1 test 仅用于独立测量，未用于修改 RuleLoss 参数或重新选择 checkpoint。

原六组的目录前置检查、退出码、命令、日志和 checkpoint 前后哈希由执行时测试 manifest 记录。L1 的 val/test 命令在交互式执行中返回退出码 0，产物包含完整 `paper_data.txt`；但当时未持久化 stdout/stderr 或执行时 manifest。为避免把事后证据表述为执行时证据，本次补充的 `l1_ruleloss_seed42_eval_manifest.json` 明确标注为事后现场审计：精确命令、观察到的退出码和阶段哈希来自本次编排记录，当前 checkpoint、代码、数据、输出路径和 `paper_data.txt` 哈希可由持久文件独立复核。

## 2. Test 评估合同

| 项目 | 配置 |
|---|---|
| 数据集 | Port_Defect |
| 数据集源目录 | `D:\defect_detection\repo_staging\canonical_dataset` |
| Test 图像/标签 | 301 / 301 |
| 类别 | Rust、Cracks、Paint Wear、Scratches、Pitting |
| 输入尺寸 | 640 × 640 |
| Batch size | 8 |
| 模型来源 | seed 42、300 epochs 训练所得 `best.pt` |
| RuleLoss | A0–B5 关闭；L1 开启 `paper` 版本 RuleLoss |
| Test 脚本 | `D:\defect_detection\ultralytics-main\val.py` |
| Test 参数 | `--split test --imgsz 640 --batch 8` |
| Test 脚本 SHA-256 | `c2aab95c65633147bbb74bd9dd1c7a6d89f061a2a45bd06a6a1e6c17d6539f26` |
| Git commit | A0/B1/B2/B3/B5：`e4f2003cdd56e447244186aaa8de4445369e8915`；B4：`61214a234c31759faacbba9f3ec22873ae2f56de`；L1：`489eacd4fc25f6c81804484d18aad202c519fa9d` |
| 证据范围 | `single_seed_screening` 的独立 test 评估 |

数据集路径指纹为 `9939ccfcee7c541482e4bfc1d084f423a1b36cfe68599a7c2abc5aa7a751bfa8`，内容指纹为 `4269dc8b69ccf97f4b8d9cb2742c0c578a1a7f28b1f3606807fba9f061cee0e4`。train/val/test 间完全重复图像数为 0。

## 3. Test 总体指标原始表

| 模型 | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.7249 | **0.7902** | 0.7521 | 0.7527 | 0.4974 | 0.4727 |
| B1 | **0.8234** | 0.6951 | 0.7507 | **0.8026** | 0.5612 | **0.5128** |
| B2 | 0.7634 | 0.6638 | 0.7072 | 0.7533 | 0.5094 | 0.4597 |
| B3 | 0.7893 | 0.6953 | 0.7343 | 0.7774 | **0.5698** | 0.4944 |
| B4 | 0.7789 | 0.7280 | 0.7502 | 0.7793 | 0.5435 | 0.4836 |
| B5 | 0.7621 | 0.7673 | **0.7631** | 0.7937 | 0.5514 | 0.5000 |
| L1 | 0.8017 | 0.7022 | 0.7463 | 0.7845 | 0.5529 | 0.4966 |

在本次 test 评估中：B1 的 Precision、mAP50 和 mAP50-95 最高；A0 的 Recall 最高；B3 的 mAP75 最高；B5 的 F1 最高。

## 4. 相对 A0 Test 基线的变化

括号前为绝对变化，括号内为相对变化百分比。

| 模型 | ΔPrecision | ΔRecall | ΔF1 | ΔmAP50 | ΔmAP75 | ΔmAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | +0.0985 (+13.59%) | -0.0951 (-12.03%) | -0.0014 (-0.19%) | +0.0499 (+6.63%) | +0.0638 (+12.83%) | +0.0401 (+8.48%) |
| B2 | +0.0385 (+5.31%) | -0.1264 (-16.00%) | -0.0449 (-5.97%) | +0.0006 (+0.08%) | +0.0120 (+2.41%) | -0.0130 (-2.75%) |
| B3 | +0.0644 (+8.88%) | -0.0949 (-12.01%) | -0.0178 (-2.37%) | +0.0247 (+3.28%) | +0.0724 (+14.56%) | +0.0217 (+4.59%) |
| B4 | +0.0540 (+7.45%) | -0.0622 (-7.87%) | -0.0019 (-0.25%) | +0.0266 (+3.53%) | +0.0461 (+9.27%) | +0.0109 (+2.31%) |
| B5 | +0.0372 (+5.13%) | -0.0229 (-2.90%) | +0.0110 (+1.46%) | +0.0410 (+5.45%) | +0.0540 (+10.86%) | +0.0273 (+5.78%) |
| L1 | +0.0768 (+10.59%) | -0.0880 (-11.14%) | -0.0058 (-0.77%) | +0.0318 (+4.22%) | +0.0555 (+11.16%) | +0.0239 (+5.06%) |

## 5. 逐类别 Test mAP50-95

| 模型 | Rust | Cracks | Paint Wear | Scratches | Pitting | 均值 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | **0.6873** | 0.3651 | 0.4624 | 0.3020 | 0.5469 | 0.4727 |
| B1 | 0.6618 | **0.4063** | **0.5604** | **0.3905** | 0.5452 | **0.5128** |
| B2 | 0.6442 | 0.3286 | 0.4829 | 0.3203 | 0.5223 | 0.4597 |
| B3 | 0.6673 | 0.3607 | 0.5347 | 0.3854 | 0.5238 | 0.4944 |
| B4 | 0.6628 | 0.3158 | 0.5425 | 0.3653 | 0.5318 | 0.4836 |
| B5 | 0.6481 | 0.3912 | 0.5356 | 0.3738 | **0.5515** | 0.5000 |
| L1 | 0.6830 | 0.3640 | 0.5350 | 0.3660 | 0.5350 | 0.4966 |

直接观测结果：

1. A0 在 Rust 上最高，但 B1 在 Cracks、Paint Wear 和 Scratches 上最高。
2. B5 在 Pitting 上最高，并取得七个配置中最高 F1。
3. Cracks 与 Scratches 仍是 mAP50-95 较低的类别。
4. B1 的总体领先主要对应 Paint Wear、Cracks 和 Scratches 的较高 test AP；由于模型间存在多项结构差异，不能把该差值归因于某一个模块。

## 6. 模型复杂度与本次 Test 吞吐率

| 模型 | 参数量 | GFLOPs | 模型大小 | Test 端到端 FPS | Test 纯推理 FPS | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 3.007M | 8.1 | 6.0 MB | 351.18 | 605.67 | 0.4727 |
| B1 | 2.747M | 23.4 | 5.5 MB | 310.62 | 466.66 | **0.5128** |
| B2 | **1.959M** | **6.6** | **4.0 MB** | 260.71 | 426.04 | 0.4597 |
| B3 | 2.291M | 6.9 | 4.7 MB | **404.46** | **702.60** | 0.4944 |
| B4 | 2.293M | 7.0 | 4.7 MB | 262.43 | 430.59 | 0.4836 |
| B5 | 3.035M | 23.8 | 6.1 MB | 326.33 | 489.35 | 0.5000 |
| L1 | 3.035M | 23.8 | 6.1 MB | 328.92 | 500.01 | 0.4966 |

B3 在本次串行 test 评估中取得最高吞吐率，同时 mAP50-95 比 A0 高 0.0217。该速度数据来自一次 RTX 4090 测量，适合用于当前模型间的初步筛选，但不能替代重复延迟测试或 RKNN 目标设备测试。

## 7. Val 到 Test 的指标变化

下表计算 `test − val`，正值表示 test 数值更高，负值表示 test 数值更低。val 与 test 图像组成不同，因此这里只描述跨 split 的数值变化，不进行统计显著性判断。

| 模型 | ΔPrecision | ΔRecall | ΔF1 | ΔmAP50 | ΔmAP75 | ΔmAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | -0.0471 | +0.0635 | +0.0045 | -0.0097 | -0.0551 | -0.0178 |
| B1 | -0.0020 | -0.0082 | -0.0054 | +0.0042 | -0.0323 | -0.0007 |
| B2 | -0.0304 | +0.0090 | -0.0047 | +0.0035 | -0.0103 | -0.0087 |
| B3 | +0.0080 | -0.0220 | -0.0104 | -0.0081 | -0.0180 | -0.0125 |
| B4 | +0.0232 | +0.0263 | +0.0248 | +0.0237 | +0.0208 | +0.0106 |
| B5 | -0.0407 | +0.0437 | +0.0052 | +0.0079 | -0.0088 | -0.0028 |
| L1 | +0.0069 | +0.0097 | +0.0102 | +0.0070 | +0.0232 | +0.0046 |

原六个结构中，除 B4 外的五个模型 test mAP50-95 均低于对应 val 结果；B4 高出 0.0106。L1 的 test mAP50-95 比 val 高 0.0046。跨 split 差值只能描述当前两个划分的数值变化，不能替代跨 seed 稳定性验证。

## 8. 受控比较

### 8.1 三组结构比较

下表为 test 上的配置级绝对差值，均以后项减前项。

| 受控比较 | 隔离因素 | ΔPrecision | ΔRecall | ΔF1 | ΔmAP50 | ΔmAP75 | ΔmAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| B4−B2 | A-GFPN | +0.0155 | +0.0642 | +0.0430 | +0.0260 | +0.0341 | +0.0239 |
| B4−B3 | RepHFE | -0.0104 | +0.0327 | +0.0159 | +0.0019 | -0.0263 | -0.0108 |
| B5−B4 | SADH | -0.0168 | +0.0393 | +0.0129 | +0.0144 | +0.0079 | +0.0164 |

在 seed 42 的 test split 上，A-GFPN 因素在指定上下文中提高本表全部总体指标；RepHFE 因素提高 Recall/F1，但降低 mAP75 和 mAP50-95；SADH 因素提高 Recall、F1 与三项 AP，但降低 Precision。这些方向与幅度不能外推为跨 seed 稳定效应。

### 8.2 RuleLoss 单变量比较

| 受控比较 | 隔离因素 | ΔPrecision | ΔRecall | ΔF1 | ΔmAP50 | ΔmAP75 | ΔmAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| L1−B5 | RuleLoss | +0.0396 | -0.0651 | -0.0168 | -0.0092 | +0.0015 | -0.0034 |

在 seed 42 的 test split 上，RuleLoss 相对 B5 提高 Precision 和 mAP75，但降低 Recall、F1、mAP50 和 mAP50-95。mAP50-95 相对下降 0.68%。逐类别 mAP50-95 中 Rust 增加 0.0349，Cracks、Paint Wear、Scratches 和 Pitting 分别变化 -0.0272、-0.0006、-0.0078 和 -0.0165。该结果与 test 前锁定的 val 方向一致：两者的 mAP50-95 差值均为负，但只有单 seed，不能宣称稳定效应。

## 9. 论文结果使用建议

论文中可以将本报告的 test 指标作为最终检测性能，但应满足以下规则：

1. 明确写明模型选择、超参数选择和 checkpoint 选择仅使用 train/val，test 只用于锁定后的最终评估。
2. 主结果表优先报告 test 的 Precision、Recall、F1、mAP50 和 mAP50-95；val 指标可用于消融选择过程或附录。
3. 不应根据本次 test 排名继续修改模型并再次反复查看同一 test 结果；如果继续调参，应回到 val，并将最终泛化评估迁移到新的保留测试集。
4. 当前只有 seed 42，不能报告 mean ± std，也不能使用统计显著性措辞。
5. 三组受控比较已经补齐，但只有 seed 42，只能报告当前上下文中的数值差异，不能宣称跨随机种子的稳定因果贡献。
6. 多 seed 范围不根据本次 test 排名决定。若正式论文消融表报告 mean ± std，应对 A0、B1、B2、B3、B4、B5 统一执行 seed 42、43、44，并预先固定聚合规则。
7. RuleLoss 的有效性必须使用 B5 与 L1 的成对比较；当前 val/test 均未显示 mAP50-95 正向收益。若该损失是论文贡献，应保持参数不变补跑 seed 43、44，而不是依据 test 调参。

## 10. 关键发现

### 10.1 B1：预先锁定精度候选的 Test 验证

- **观察**：B1 的 mAP50-95 为 0.5128，比 A0 高 0.0401；Precision 和 mAP50 同样最高。
- **解释边界**：B1 与 A0 存在多项结构差异，当前结果只支持 B1 完整配置的 test 表现，不能拆分 SADH 或 CSPStage 的独立贡献。
- **含义**：本次 test 结果记录了 test 前已由 val 锁定的 B1 精度候选表现，不用于重新选择候选。
- **下一步**：按照 test 前声明的统一实验矩阵补充多 seed 训练；不得根据本次 test 数值改变超参数或只追加某个模型。

### 10.2 B3：预先锁定效率候选的 Test 验证

- **观察**：B3 的 mAP50-95 为 0.4944，GFLOPs 为 6.9，本次端到端速度为 404.46 FPS。
- **解释边界**：B4−B2 的 test 结果支持 A-GFPN 因素在指定上下文中的 +0.0239 mAP50-95 数值差异，但仍需多 seed 验证。
- **含义**：B3 的部署验证候选身份来自 test 前的 val/复杂度分析；本次 test 只补充其独立划分性能，尚未证明目标设备优势。
- **下一步**：执行固定 warmup、固定重复次数的目标设备延迟和内存测试。

### 10.3 B5：预先锁定平衡候选的 Test 验证

- **观察**：B5 的 F1 为 0.7631，为七个配置中最高；mAP50-95 为 0.5000，比 A0 高 0.0273。
- **解释边界**：B5−B4 的受控结果显示 SADH 因素在指定上下文中带来 +0.0164 mAP50-95 和 +0.0129 F1，同时 Precision 下降 0.0168；仍需多 seed 验证。
- **含义**：B5 的平衡候选身份已在 test 前由 val 结果锁定；本次 test 只记录该预先方案的 F1、召回和计算量表现。
- **下一步**：补齐所有模型的 seed 43、44 后，再判断完整结构是否具有稳定收益。

### 10.4 L1：RuleLoss 单变量 Test 验证

- **观察**：L1 的 test mAP50-95 为 0.4966，比 B5 低 0.0034；Precision 高 0.0396，但 Recall 低 0.0651。
- **跨 split 一致性**：L1−B5 的 mAP50-95 在 val 和 test 分别为 -0.0108 和 -0.0034，方向一致但幅度不同。
- **实现证据**：L1 使用与 B5 相同的模型 YAML 和训练预算；清单记录 RuleLoss 已启用、criterion 来源为项目实现、最终 `lambda_rule=1.0`。
- **含义**：本次结果不支持“RuleLoss 提升总体检测精度”的论文表述；可以如实报告其提高 test Precision、但牺牲 Recall 且未提高 mAP50-95。
- **下一步**：保持预先固定的损失参数，对 B5/L1 成对补跑 seed 43、44，并报告 mean ± std。

## 11. 结果证据位置

```text
Test 输出：
D:\defect_detection\repo_staging_ablation_300ep\training_project\runs\ablation_300ep_b8_20260719_test

Test 日志：
D:\defect_detection\repo_staging_ablation_300ep\training_project\runs\ablation_300ep_b8_20260719_test_logs

Test 执行清单：
D:\defect_detection\repo_staging_ablation_300ep\training_project\runs\ablation_300ep_b8_20260719_test_manifest.json

训练队列与 checkpoint：
D:\defect_detection\repo_staging_ablation_300ep\training_project\runs\ablation_300ep_b8_20260719

L1 RuleLoss 训练：
D:\defect_detection\repo_staging_ablation_300ep\training_project\runs\ruleloss_300ep_b8_seed42\L1_B5_RuleLoss_seed42_e300_b8

L1 RuleLoss Test：
D:\defect_detection\repo_staging_ablation_300ep\training_project\runs\ruleloss_300ep_b8_seed42_test\L1_B5_RuleLoss_seed42_e300_b8_test

L1 val/test 事后审计清单：
D:\defect_detection\repo_staging_ablation_300ep\training_project\ablations\reports\l1_ruleloss_seed42_eval_manifest.json
```

| 模型 | best.pt SHA-256 |
|---|---|
| A0 | `03b3c28cbb4fae14db25289bc6375fae446acb48bfd5545792760f917c61d7f3` |
| B1 | `3473e81b03d6d688c1d098e954dec0578748c9fdd50dbb57c27751934b47e8e7` |
| B2 | `824a487f5e102ca08d74aad5e956016a77949d407caf3efe02ff5f1094f537af` |
| B3 | `374f7d4c65657159da71c28e5a7201f53016453f5261a56bac4593f70ece41f0` |
| B4 | `8b8be99dc0f20c079705509ee78390f6e9cde691189ddd8cc4550f299f8ffe13` |
| B5 | `03a5514454f2d0dce62db18b209948be11e430b636f9a16abba734191c63080a` |
| L1 | `df5c569ef20d6dc1106404e802e8d313d4f2454380ddd64730bdd855312be331` |

---

报告更新日期：2026-07-21
实验分支：`codex/ablation-300ep-rerun`
