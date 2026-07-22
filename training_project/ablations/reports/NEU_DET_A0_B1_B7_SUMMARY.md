# NEU-DET A0、B1–B7 消融实验汇总

## 统一实验合同

- 数据集：NEU-DET_YOLO（train=1259，val=359，test=181；跨划分重复图像=0）
- 正式训练：300 epochs，batch=8，imgsz=640，seed=42，scratch，AMP=false，RuleLoss=false
- 评估：同一外部 val.py，batch=8，imgsz=640；每个 best.pt 独立运行 val/test
- A0/B1 使用默认 patience，但实际完整运行 300 epochs；B2–B7 明确 patience=0
- 项目不存在 B0；本文按实际消融编号汇总 A0 与 B1–B7

## 验证集汇总

| Model | Structure | Split | P | R | F1 | mAP50 | mAP75 | mAP50-95 | ΔmAP50 vs A0 | ΔmAP50-95 vs A0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | YOLOv8n baseline | val | 0.6851 | 0.6370 | 0.6352 | 0.7022 | 0.2851 | 0.3408 | +0.0000 | +0.0000 |
| B1 | SADH | val | 0.5630 | 0.6925 | 0.5936 | 0.6812 | 0.2494 | 0.3224 | -0.0210 | -0.0184 |
| B2 | RepHFE | val | 0.6508 | 0.6003 | 0.6026 | 0.6586 | 0.2486 | 0.3155 | -0.0436 | -0.0253 |
| B3 | A-GFPN | val | 0.6247 | 0.6515 | 0.6247 | 0.6825 | 0.2409 | 0.3270 | -0.0197 | -0.0138 |
| B4 | A-GFPN + RepHFE | val | 0.6371 | 0.6684 | 0.6371 | 0.6891 | 0.2710 | 0.3336 | -0.0131 | -0.0072 |
| B5 | A-GFPN + RepHFE + SADH | val | 0.6241 | 0.6626 | 0.6253 | 0.6804 | 0.2431 | 0.3207 | -0.0218 | -0.0201 |
| B6 | A-GFPN + SADH | val | 0.6410 | 0.6577 | 0.6222 | 0.6889 | 0.2390 | 0.3238 | -0.0133 | -0.0170 |
| B7 | RepHFE + SADH | val | 0.5994 | 0.6122 | 0.5847 | 0.6506 | 0.2528 | 0.3171 | -0.0516 | -0.0237 |

## 测试集汇总（论文最终泛化指标）

| Model | Structure | Split | P | R | F1 | mAP50 | mAP75 | mAP50-95 | ΔmAP50 vs A0 | ΔmAP50-95 vs A0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | YOLOv8n baseline | test | 0.6517 | 0.6745 | 0.6503 | 0.7023 | 0.2855 | 0.3431 | +0.0000 | +0.0000 |
| B1 | SADH | test | 0.5653 | 0.6771 | 0.5907 | 0.6810 | 0.2362 | 0.3204 | -0.0213 | -0.0227 |
| B2 | RepHFE | test | 0.5932 | 0.6952 | 0.6300 | 0.6677 | 0.2871 | 0.3313 | -0.0346 | -0.0118 |
| B3 | A-GFPN | test | 0.6855 | 0.6643 | 0.6589 | 0.6996 | 0.2894 | 0.3494 | -0.0027 | +0.0063 |
| B4 | A-GFPN + RepHFE | test | 0.6730 | 0.6507 | 0.6499 | 0.6852 | 0.2825 | 0.3412 | -0.0171 | -0.0019 |
| B5 | A-GFPN + RepHFE + SADH | test | 0.6643 | 0.6915 | 0.6592 | 0.6911 | 0.2442 | 0.3331 | -0.0112 | -0.0100 |
| B6 | A-GFPN + SADH | test | 0.6042 | 0.6626 | 0.6106 | 0.6785 | 0.2222 | 0.3238 | -0.0238 | -0.0193 |
| B7 | RepHFE + SADH | test | 0.5750 | 0.6731 | 0.6022 | 0.6494 | 0.2520 | 0.3108 | -0.0529 | -0.0323 |

## 结论

1. 验证集 mAP50-95 最高为 A0（0.3408）。
2. 测试集 mAP50-95 最高为 B3（0.3494）。
3. 测试集 mAP50 最高为 A0（0.7023）。
4. 测试集 Precision、Recall、F1、mAP75 最优项依次为 B3、B2、B5、B3，说明不同结构存在明确的精确率—召回率权衡。
5. 仅 B3 的测试集 mAP50-95 高于 A0（+0.0063），但其 mAP50 低 0.0027；不能声称所有改进模块或完整组合都稳定优于基线。
6. 当前仅包含 seed=42；表中差值是单次实验差值，不能表述为统计显著提升。建议至少对 A0、B3、B5 运行 3 个种子并报告 mean±std。

## 证据清单

| Model | Split | paper_data SHA-256 | checkpoint SHA-256 |
|---|---|---|---|
| A0 | val | `af798d5af59e5075fd7f90e6b62a11940cb59ccebf662583940a3837abdd4d92` | `f302df2168e4ec285341888bb16258dff45da414c4519cbb1cf9d950676d4577` |
| A0 | test | `18c90b33b0f7b4f7cdcfa8f31cceb840c61a3a581c99997f035c9675d100d973` | `f302df2168e4ec285341888bb16258dff45da414c4519cbb1cf9d950676d4577` |
| B1 | val | `c798da833f6e68ba339b47db74eb8d927fc414eeb53762ebd49daf46d09f5d01` | `7171dc1e62b841d44806db4687b8ac8ae504f1e442902c99b8c2114d14b8ef98` |
| B1 | test | `afb29180a6c328f4c10e119778f9cdb694d04d8a96c627c496459ded49d0e1a8` | `7171dc1e62b841d44806db4687b8ac8ae504f1e442902c99b8c2114d14b8ef98` |
| B2 | val | `ccc51e10c50fb0f5976cb72f070c82ec6f8df9b9921b89cabb21fba046506785` | `391d0f59bc416919886218d0235cd5674ebfad05ab4881f35109354aa2754456` |
| B2 | test | `92e5beb9ae1edaa790c14ce191090eeca89e75d718888538d90bb85966822ead` | `391d0f59bc416919886218d0235cd5674ebfad05ab4881f35109354aa2754456` |
| B3 | val | `0f457c5d72c5346a152ada2c602f5811bddb9ba752aabcb451a651c13c62c3f3` | `8661c165f3802875b8955735eecc08c9191d6099beabcd4b093a97ee388a8eab` |
| B3 | test | `63e34d3ad6cb9fd8266d8099b48f2269018c11e421a75381f29b35a9687d2fbc` | `8661c165f3802875b8955735eecc08c9191d6099beabcd4b093a97ee388a8eab` |
| B4 | val | `c80d7d8b68bc58cbf27be5e295294fec594f4f8ed00c40dc325734e7a29cebd0` | `c9716e6d7aec26b9a7966798392a1da0d492f409386033c666dcfdfb16d1efca` |
| B4 | test | `ab9c3dc986114c2bc2aa0c9dd16ba95d2fec8f5b31c0761e9eedddcda514a195` | `c9716e6d7aec26b9a7966798392a1da0d492f409386033c666dcfdfb16d1efca` |
| B5 | val | `6469efad4e8e1b686d6258f4a5ba8763fd5e2c0d519145bd094fe829b1c78f29` | `e8b7a32ee6d17778db9e6a781e0f03fad343c41f3f4ab8645a1c4171ae59005d` |
| B5 | test | `b8af2f6f8f59acac7a0b9813e226223acd7bb4123359d21568aea447ca4c3789` | `e8b7a32ee6d17778db9e6a781e0f03fad343c41f3f4ab8645a1c4171ae59005d` |
| B6 | val | `ca9b7b2d619fd3f30fa14368c23f043a4977f03a78e8bb6d3b9965a0c3ff0699` | `44282bec233895cedd6fdaa227f37aadb6c2d8f0ab1a03680c6f93dc9e322027` |
| B6 | test | `35070c1d6e5c85a39dad78de781ddfa85bf4265c19c75f4872f0ab735f9d3baa` | `44282bec233895cedd6fdaa227f37aadb6c2d8f0ab1a03680c6f93dc9e322027` |
| B7 | val | `fa3f880dd196e4e0228cab602de091db3cc175ff3a0a6d24fe4598e0446649f4` | `9d0026f9a3b69c907980fe5517df1f33bb2aee654acdf52ff0a1be00aedc322a` |
| B7 | test | `3a0402843011d391b5c7e252186b3d9b177ae9e792210820b6697c901cf9015a` | `9d0026f9a3b69c907980fe5517df1f33bb2aee654acdf52ff0a1be00aedc322a` |
