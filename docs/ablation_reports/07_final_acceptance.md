# 阶段 7：全新环境最终验收

阶段：7（工程最终验收与交付证据加固）

状态：PASSED（`engineering_acceptance`）

正式消融训练状态：BLOCKED

## 修改文件

- `training_project/run_verifier_safely.py`
- `training_project/verify_all.py`
- `training_project/ablations/verify_ablation_models.py`
- `training_project/ablations/verify_stage4_fairness.py`
- `training_project/ablations/run_stage6_smoke.py`
- `training_project/ablations/verify_final_acceptance.py`
- `training_project/ablations/verify_final_negative_cases.py`
- `export_pipeline/export_onnx.py`
- `export_pipeline/verify_onnx_consistency.py`
- `export_pipeline/canonical_artifacts.json`
- `README.md`、`RELEASES.md` 与 `docs/` 下的验收、架构、环境和交接说明

## 验证命令

```powershell
python training_project/ablations/verify_final_negative_cases.py

python training_project/ablations/verify_final_acceptance.py `
  --require-real-data `
  --smoke-state training_project/runs/ablation_smoke_final/stage6_state.json `
  --onnx-manifest export_pipeline/outputs/port_defect_smoke/artifact_manifest.json

git status --porcelain --untracked-files=all
git diff --check
```

完整最终验收必须同时提供真实数据、smoke state 和 ONNX manifest；缺少任一证据参数均退出非零，不再允许生成 `not_requested` 的伪通过结果。

## 通过项

- 验收环境：Python 3.10.20、PyTorch 2.1.2、CUDA 12.1、Ultralytics 8.2.50，CUDA 可用。
- 核心 `verify_all` 11 项、六模型总门禁和 PT/ONNX 一致性全部通过。
- A0～B5 与 L1 的合同、manifest、checkpoint SHA、strict YAML 加载和固定输入推理均被重新验证。
- 数据规模固定为 train/val/test 图像 2428/607/350，标签数量一致，无缺失、空标签或非法标签行。
- 最终验收入口同时检查进入和退出时的 Git clean，并确认 HEAD 未变化。
- 验证型 Python 子进程使用专用安全启动器，绕过 Windows/PyTorch DLL 退出期的间歇性 `0xC0000005`；验证异常仍保留 traceback 并退出非零，训练进程不使用该绕行。
- 主执行方在干净工作树提交 `cf650bb35961de50fe6e4799c7f5561e9fcdd325` 上连续运行三次完整验收，均退出 0，耗时约 70～73 秒。
- 对抗性审查方独立连续运行两次完整验收，均退出 0，耗时约 71 秒；最终判定 Stage 7 PASS。
- 负向套件明确拒绝：缺少最终证据参数、错误哈希、路径穿越、参数量/类别伪造、BGR 预处理、伪造 runtime、替换真实 checkpoint 后重算 manifest 哈希、FAKE smoke commit、伪造 strict/hash 状态。
- `export_pipeline/canonical_artifacts.json` 作为 Git 跟踪的可信锚，固定来源提交/checkpoint/run manifest、PT/ONNX/YAML 哈希、canonical YAML、数据 YAML、类别、拓扑、预处理和 runtime。
- PT 权重可严格加载到 canonical YAML 拓扑；参数量、层数、类别与 ONNX 图元数据均从实际文件和受控配置重建验证。
- CASE_C、`extra_modules` 和 legacy 主链路继续被拒绝。

## 失败项

- 正式消融训练数据门禁未通过：当前 train/val/test 间存在 26 组图像内容完全重复。

## 风险或残留

- 本阶段通过只证明工程构建、smoke、checkpoint 和 PT/ONNX 交付链可复验，不证明任何精度提升或模型优劣。
- 在重新划分数据并将跨 split 内容重复降为 0 前，不得启动或引用正式多种子消融训练。
- `export_pipeline/outputs/` 与 `training_project/runs/` 是忽略目录；在新机器验收时必须提供与 tracked lock 和 smoke contract 匹配的产物。
- 原工作区 `D:\defect_detection\repo_staging` 中存在一处未纳入本分支提交的用户工作改动：`training_project/configs/port_defect_ruleloss_smoke.yaml` 的 `plots` 值变化。最终 clean 验收在独立工作树 `D:\defect_detection\repo_staging_final_acceptance` 完成，该用户改动未被暂存、覆盖或回退。
- 安全启动器只用于短生命周期验证子进程；不得用于正式训练进程或替代对原生运行期崩溃的诊断。

## 是否允许进入下一阶段

允许进入工程分支发布、复制使用和数据治理；不允许进入正式消融训练。正式训练需先去除 26 组跨划分重复，更新并审查数据契约，再重新运行完整验收。
