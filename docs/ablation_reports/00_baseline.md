# Ablation Integration Stage 0 Baseline

阶段：0 - 分支与可追溯基线

状态：PASSED

基线提交：`7c6a7a405966782f0b71c6a5597cf73a60600aa9`

实现分支：`codex/ablation-config-integration`

环境：Python 3.10.20、PyTorch 2.1.2、CUDA 12.1、Ultralytics 8.2.50；CUDA 可用。

目标 B4 签名：2,308,655 参数、25 层、CSPStage=4、RepHFE=2。

源目录：`D:\defect_detection\History_Data\harpnet_selected_artifacts`

| 原文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `yolov8n.yaml` | 1888 | `c8d4b4fb7af33409a11b19203a5d982aec5e839841988b289c12554816cb575e` |
| `B1_SADH.yaml` | 1414 | `86bc637d886a206f7d6acfc4434dcbea1be6ae672167acf9755ec2e19cf9db7b` |
| `B2_RepHFE.yaml` | 1238 | `10c2407cc9f716db8f21285ecdd08e6258e5022a6415fff10346594c606b2339` |
| `B3_A-GFPN.yaml` | 1535 | `3d7fca3e34758da865cf4f505f0e380ebc7f1f53bf425b34faf61fa1c036795e` |
| `B4_A-GFPN_RepHFE_target.yaml` | 1296 | `32c400d0a2ee4d6619eaeeabf638a1bb5ba2bd6682affe5cca1577af6c56b4e2` |
| `B5_Full_model.yaml` | 2804 | `4b518706d2eb4b143b7352a552e87862aabcf61eb85bc1f625ee144fda10c221` |

B4 历史源与当前 `training_project/models/B4_A-GFPN_RepHFE_target.yaml` 哈希一致，当前文件继续作为唯一 canonical 真源。

验证命令：

```text
python training_project/verify_model_signature.py
python training_project/verify_all.py
Get-FileHash <source-yaml> -Algorithm SHA256
```

通过项：综合门禁 11/11；vendor 无项目反向依赖；主链路无 legacy；源 YAML 六份均已盘点。

失败项：无。

风险或残留：历史 PT 不进入 Git、不用于新消融初始化；阶段 1 需逐字节归档源 YAML 并独立复核哈希。

对抗性审查：PASSED；无必须修复项。

是否允许进入下一阶段：是。
