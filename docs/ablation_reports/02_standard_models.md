# Ablation Integration Stage 2 Standard Models

阶段：2 - 标准解析路径与 A0/B2/B3/B4 canonical 模型

状态：PASSED

修改文件：

- `ultralytics-main/ultralytics/nn/tasks.py`
- `training_project/ablations/manifest.yaml`
- `training_project/ablations/verify_stage2_models.py`
- `training_project/models/ablations/A0_yolov8n.yaml`
- `training_project/models/ablations/B2_RepHFE.yaml`
- `training_project/models/ablations/B3_A-GFPN.yaml`

验证命令：

```text
python training_project/ablations/verify_archive.py --check-source --check-git
python training_project/ablations/verify_stage2_models.py
python training_project/verify_all.py
git diff --check
```

通过项：

- 删除 vendor 解析器中失去实现的 Fusion、extra_modules、timm 和 list-backbone 分支；`nn.Upsample` 和标准 `nn.Identity` 通道传播正常。
- 未知 YAML token 会快速失败并报告 token 与 layer index；已删除的 `Warehouse_Manager` 会给出明确拒绝信息。
- A0 在空扩展 registry 下独立构建；B2/B3/B4 仅在项目侧安装扩展后构建，未加载 `extra_modules`。
- canonical 与历史 source 做完整 dict 规范化全等，只允许 `nc=5`、A0 仅保留 n scale、单 scale 的隐式/显式 `scale=n` 三项声明变换。
- 内置 `activation` 顶层篡改负向测试按预期失败，防止非拓扑顶层语义漂移伪通过。
- A0/B2/B3/B4 的参数量分别为 3,011,823、1,969,391、2,303,663、2,308,655；层数分别为 23、25、27、25。
- 四模型 stride 均为 `[8, 16, 32]`，固定输入输出均为 `[1, 9, 84]` 且数值有限。
- manifest 机器校验六实验闭集、角色、组件布尔矩阵、三组受控比较、唯一差异因子和 controls；篡改 B2 的 RepHFE 标记会按预期失败。
- 明确 A0 只是外部 YOLOv8n 参考、B1 是 standalone 架构点；仅 B4-B2、B4-B3、B5-B4 分别作为 A-GFPN、RepHFE、SADH 的受控增量证据。
- 阶段 1 source archive 未修改，外部源、工作树与 Git HEAD 三方校验继续通过；原 11 项验证无回归。

失败项：首次对抗性审查指出 parser 仅绕过 Fusion 且实验语义错误归因；第二次审查指出 canonical 顶层字段和 manifest 可伪通过。上述问题均修复并完整重跑，最终审查 PASSED，无必须修复项。

风险或残留：B1/B5 仍为 `archive_only`，需在阶段 3 迁移 `Detect_LSCSBD` 后验证；vendor 中 KernelWarehouse 示例配置仍存在但 runtime 明确拒绝，最终文档需标记为不支持；本阶段仅达到 `build_forward`，尚未代表真实训练通过。

是否允许进入下一阶段：是。
