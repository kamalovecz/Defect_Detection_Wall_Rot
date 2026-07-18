# Ablation Integration Stage 3 SADH Models

阶段：3 - Detect_LSCSBD 解耦迁移与 B1/B5 canonical 模型

状态：PASSED

修改文件：

- `defect_modules/sadh.py`
- `defect_modules/integration.py`
- `defect_modules/registry.py`
- `ultralytics-main/ultralytics/nn/extensions.py`
- `ultralytics-main/ultralytics/nn/tasks.py`
- `ultralytics-main/ultralytics/engine/exporter.py`
- `training_project/models/ablations/B1_SADH.yaml`
- `training_project/models/ablations/B5_Full.yaml`
- `training_project/ablations/manifest.yaml`
- `training_project/ablations/verify_stage3_sadh.py`
- `training_project/verify_registry.py`
- `training_project/verify_tasks_import_boundary.py`

验证命令：

```text
python training_project/ablations/verify_archive.py --check-source --check-git
python training_project/ablations/verify_stage2_models.py
python training_project/ablations/verify_stage3_sadh.py --check-reference-source --require-cuda
python training_project/verify_all.py
git diff --check
```

通过项：

- `Detect_LSCSBD` 的检测用途最小实现迁移到 `defect_modules/sadh.py`；legacy runtime 与 paper 参考文件的 SHA-256 均已核对并登记角色。
- vendor 仅通过通用 `multi_input_channels`、`detection_head` metadata 识别扩展，不 import `defect_modules`，也未加载 `extra_modules`。
- 未安装 integration 时 `Detect_LSCSBD` 明确失败；安装后重复注册幂等，多尺度通道、stride 初始化、task guess 与 `_apply` 均通过生产路径。
- B1/B5 canonical 与历史 source 做完整 dict 规范化全等，参数量分别锁定为 2,756,005 和 3,049,701，均为 25 个顶层 layer。
- B1/B5 均含 4 个 CSPStage 和 1 个 Detect_LSCSBD；RepHFE 数量分别为 0 和 2，与 manifest 组件矩阵一致。
- 两模型 stride 均为 `[8, 16, 32]`；固定输入预测为 `[1, 9, 84]`，训练态 raw outputs 为 `[1,69,8,8]`、`[1,69,4,4]`、`[1,69,2,2]`。
- 原生 `v8DetectionLoss` 的 box/cls/dfl 三项 loss 有限；bbox、classification、standard shared、depthwise shared 和 stem 五条梯度路径均有限且非零。
- 在 CPU 生成非空 anchors 后，参数、stride、anchors、strides 可随模型迁入 CUDA、完成有限前向并全部迁回 CPU；float64 往返也通过。
- 正式 exporter 使用同一通用 detection-head helper；设置 ONNX export 后，B1/B5 都返回单一 `[1,9,84]` Tensor。
- 阶段 1/2 门禁、原 11 项总验证和 diff-check 均无回归。

失败项：首次对抗性审查指出 exporter 未识别注册式检测头、CUDA 锚点迁移未实测、回归与共享路径梯度未覆盖。全部修复并完整重跑后，最终审查 PASSED，无必须修复项。

风险或残留：当前只验证 exporter 准备和 head 输出契约，完整 ONNX/ORT 数值一致性留到最终导出门禁；最小实现不承诺 legacy `eca_k` 参数或旧整对象 pickle 兼容，历史 PT 继续仅作 provenance；本阶段达到 `build_forward_backward`，尚未代表真实数据训练完成。

是否允许进入下一阶段：是。
