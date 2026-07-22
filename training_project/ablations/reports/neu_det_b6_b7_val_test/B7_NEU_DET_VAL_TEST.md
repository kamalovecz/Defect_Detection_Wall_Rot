# B7 在 NEU-DET 验证集与测试集上的 PT 性能报告

## 实验契约

- 模型：`B7`
- 正式训练：300 epochs，batch=8，imgsz=640，seed=42，patience=0
- 权重初始化：scratch；AMP：关闭；RuleLoss：关闭
- 数据集：`NEU-DET_YOLO`
- 数据描述：`D:\defect_detection\ultralytics-main\dataset\NEU-DET_YOLO\neu-det.yaml`
- 检查点：`training_project/runs/neu_det_ablation_b6_b7_300ep_b8_20260722_p0/B7_seed42_e300_b8_neu_p0/weights/best.pt`
- 检查点 SHA-256：`9d0026f9a3b69c907980fe5517df1f33bb2aee654acdf52ff0a1be00aedc322a`

## 总体性能

| Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.5994 | 0.6122 | 0.5847 | 0.6506 | 0.2528 | 0.3171 |
| test | 0.5750 | 0.6731 | 0.6022 | 0.6494 | 0.2520 | 0.3108 |
| test - val | -0.0244 | 0.0609 | 0.0175 | -0.0012 | -0.0008 | -0.0063 |

## 分类别性能

| Class | Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| crazing | val | 0.4210 | 0.1057 | 0.1690 | 0.2935 | 0.0481 | 0.1049 |
| crazing | test | 0.5160 | 0.2414 | 0.3289 | 0.2884 | 0.0261 | 0.0815 |
| inclusion | val | 0.6251 | 0.6776 | 0.6503 | 0.7014 | 0.2336 | 0.3169 |
| inclusion | test | 0.6137 | 0.7462 | 0.6735 | 0.7357 | 0.2380 | 0.3253 |
| patches | val | 0.8098 | 0.9041 | 0.8544 | 0.9322 | 0.6078 | 0.5565 |
| patches | test | 0.7195 | 0.8916 | 0.7963 | 0.8530 | 0.4719 | 0.4839 |
| pitted_surface | val | 0.6304 | 0.7188 | 0.6717 | 0.7130 | 0.4533 | 0.4374 |
| pitted_surface | test | 0.6635 | 0.8205 | 0.7337 | 0.8362 | 0.5037 | 0.5011 |
| rolled-in_scale | val | 0.5034 | 0.4112 | 0.4526 | 0.4872 | 0.1083 | 0.1956 |
| rolled-in_scale | test | 0.4889 | 0.5072 | 0.4979 | 0.4935 | 0.1303 | 0.2011 |
| scratches | val | 0.6066 | 0.8560 | 0.7100 | 0.7765 | 0.0655 | 0.2911 |
| scratches | test | 0.4487 | 0.8315 | 0.5828 | 0.6895 | 0.1421 | 0.2717 |

## 产物与哈希

| Artifact | Path | SHA-256 |
|---|---|---|
| results.csv | `training_project/runs/neu_det_ablation_b6_b7_300ep_b8_20260722_p0/B7_seed42_e300_b8_neu_p0/results.csv` | `f773f390125245d6606b298a5fe74ba7eb33c28e52c6aea507e1a3b396bcd96f` |
| run_manifest.json | `training_project/runs/neu_det_ablation_b6_b7_300ep_b8_20260722_p0/B7_seed42_e300_b8_neu_p0/run_manifest.json` | `4ced86a91b8634e1d4e230d49874b957b6af0d1d82275a85ef26311ad4c1cebe` |
| val paper_data | `training_project/runs/neu_det_b6_b7_pt_val_test_20260722/B7/val/paper_data.txt` | `fa3f880dd196e4e0228cab602de091db3cc175ff3a0a6d24fe4598e0446649f4` |
| test paper_data | `training_project/runs/neu_det_b6_b7_pt_val_test_20260722/B7/test/paper_data.txt` | `3a0402843011d391b5c7e252186b3d9b177ae9e792210820b6697c901cf9015a` |

## 结论边界

- 本报告使用同一 `best.pt` 分别在官方 val/test split 上独立评估。
- 当前结果仅包含 seed=42，不能用于估计跨随机种子的均值和标准差。
- test 指标用于最终泛化能力陈述；val 指标用于模型选择与消融过程比较。
