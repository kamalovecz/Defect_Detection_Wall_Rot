# B6 在 NEU-DET 验证集与测试集上的 PT 性能报告

## 实验契约

- 模型：`B6`
- 正式训练：300 epochs，batch=8，imgsz=640，seed=42，patience=0
- 权重初始化：scratch；AMP：关闭；RuleLoss：关闭
- 数据集：`NEU-DET_YOLO`
- 数据描述：`D:\defect_detection\ultralytics-main\dataset\NEU-DET_YOLO\neu-det.yaml`
- 检查点：`training_project/runs/neu_det_ablation_b6_b7_300ep_b8_20260722_p0/B6_seed42_e300_b8_neu_p0/weights/best.pt`
- 检查点 SHA-256：`44282bec233895cedd6fdaa227f37aadb6c2d8f0ab1a03680c6f93dc9e322027`

## 总体性能

| Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.6410 | 0.6577 | 0.6222 | 0.6889 | 0.2390 | 0.3238 |
| test | 0.6042 | 0.6626 | 0.6106 | 0.6785 | 0.2222 | 0.3238 |
| test - val | -0.0368 | 0.0049 | -0.0116 | -0.0104 | -0.0168 | 0.0000 |

## 分类别性能

| Class | Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| crazing | val | 0.5324 | 0.1111 | 0.1839 | 0.3433 | 0.0355 | 0.1070 |
| crazing | test | 0.4723 | 0.1379 | 0.2135 | 0.3473 | 0.0193 | 0.1078 |
| inclusion | val | 0.6514 | 0.7334 | 0.6899 | 0.7259 | 0.1930 | 0.3137 |
| inclusion | test | 0.6695 | 0.6846 | 0.6770 | 0.7378 | 0.2139 | 0.3250 |
| patches | val | 0.8202 | 0.9162 | 0.8656 | 0.9338 | 0.6623 | 0.5751 |
| patches | test | 0.7705 | 0.9518 | 0.8516 | 0.9083 | 0.4205 | 0.5013 |
| pitted_surface | val | 0.5870 | 0.7344 | 0.6525 | 0.7389 | 0.3551 | 0.4019 |
| pitted_surface | test | 0.6384 | 0.8462 | 0.7277 | 0.8443 | 0.5028 | 0.5206 |
| rolled-in_scale | val | 0.5442 | 0.5472 | 0.5457 | 0.5344 | 0.1396 | 0.2294 |
| rolled-in_scale | test | 0.4680 | 0.5072 | 0.4868 | 0.4525 | 0.1131 | 0.1868 |
| scratches | val | 0.7105 | 0.9040 | 0.7957 | 0.8573 | 0.0484 | 0.3157 |
| scratches | test | 0.6063 | 0.8478 | 0.7070 | 0.7807 | 0.0633 | 0.3010 |

## 产物与哈希

| Artifact | Path | SHA-256 |
|---|---|---|
| results.csv | `training_project/runs/neu_det_ablation_b6_b7_300ep_b8_20260722_p0/B6_seed42_e300_b8_neu_p0/results.csv` | `79be69bbefbe290694bfa406c995f24d26163de5a8ddc0eacda40940b9aabd9a` |
| run_manifest.json | `training_project/runs/neu_det_ablation_b6_b7_300ep_b8_20260722_p0/B6_seed42_e300_b8_neu_p0/run_manifest.json` | `97c866c2766aa6e0a9729fa24327fd5218d35d6251ef754b14a9333535692e0d` |
| val paper_data | `training_project/runs/neu_det_b6_b7_pt_val_test_20260722/B6/val/paper_data.txt` | `ca9b7b2d619fd3f30fa14368c23f043a4977f03a78e8bb6d3b9965a0c3ff0699` |
| test paper_data | `training_project/runs/neu_det_b6_b7_pt_val_test_20260722/B6/test/paper_data.txt` | `35070c1d6e5c85a39dad78de781ddfa85bf4265c19c75f4872f0ab735f9d3baa` |

## 结论边界

- 本报告使用同一 `best.pt` 分别在官方 val/test split 上独立评估。
- 当前结果仅包含 seed=42，不能用于估计跨随机种子的均值和标准差。
- test 指标用于最终泛化能力陈述；val 指标用于模型选择与消融过程比较。
