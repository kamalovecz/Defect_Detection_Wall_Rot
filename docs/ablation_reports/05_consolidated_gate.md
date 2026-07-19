# 阶段 5：六模型自动化总门禁

阶段：5（统一构建、前向、损失与反向验证）

状态：PASSED

修改文件：

- `training_project/ablations/verify_ablation_models.py`

验证命令：

```powershell
python training_project/ablations/verify_ablation_models.py --check-reference-source --require-cuda
git diff --check
```

通过项：

- 总门禁在独立子进程中重跑阶段 1～4，任一非零退出码都会终止。
- A0～B5 的配置、manifest、模型 YAML 路径和 SHA-256 闭环一致。
- 六模型固定输入 CPU/CUDA 前向均为有限值，输出 shape 均为 `[1, 9, 84]`。
- 六模型参数量、层数、stride、CSPStage/RepHFE/Detect_LSCSBD 计数符合固定签名。
- 结构消融全部确认 RuleLoss 关闭。
- criterion 精确锁定为 `ultralytics.utils.loss.v8DetectionLoss`。
- criterion 实际使用的 box/cls/dfl 权重来自合并配置，实测为 7.5/0.5/1.5。
- 六模型损失与三个 loss item 均为有限值。
- 六模型梯度均到达首层骨干、完整检测头、cv2 回归分支与 cv3 分类分支。
- 标准头来源严格为 `ultralytics.nn.modules.head.Detect`；SADH 头严格来自 `defect_modules.sadh.Detect_LSCSBD`。
- 项目模块均来自 `defect_modules`，进程未加载 legacy `extra_modules`。
- 独立对抗性审查在修复 criterion 和分支梯度伪通过点后通过。
- 提交态曾复现一次 Windows CUDA 进程退出访问异常；增加显式同步、缓存释放和引用回收后，总门禁连续 5 次退出 0。

失败项：无。

风险或残留：

- 当前使用合成的单目标 64×64 batch，只验证计算图，不替代真实数据训练。
- 空标注 batch、优化器更新、checkpoint 保存/新进程重载在阶段 6 验证。
- 数据内容哈希和跨划分重复检测在阶段 6 验证。

是否允许进入下一阶段：是。
