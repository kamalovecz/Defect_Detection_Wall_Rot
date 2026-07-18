# HARP-Net 训练工程解耦版 Release Notes

## HARP-Net Decoupled Training v1

> 分支：`codex/decouple-training-v1`
>
> 基线：`main@937a757`
>
> 功能验收提交：`cbc9d69`
>
> 发布类型：训练工程解耦与工程闭环版本
>
> 验收日期：2026-07-18

本版本完成了 HARP-Net 训练代码从历史实验目录、隐式 monkey patch、
Ultralytics 私有模块集合以及 CASE_C 旧权重依赖中的解耦。训练仓库现在负责
模型构建、训练、checkpoint 验证和 canonical ONNX 一致性验证；RKNN 转换与
板端验证由部署仓库 `331_PC_RKNN` 负责。

## Release highlights

- 建立单向模块注册边界，vendor Ultralytics 不再反向导入项目代码。
- 将 `CSPStage`、`RepHFE` 和 RuleLoss 收敛到独立的 `defect_modules` 契约。
- 使用 criterion factory 替代跨模块全局 monkey patch，RuleLoss 默认关闭。
- 统一数据集名称为 `Port_Defect`，训练入口改为 YAML 配置驱动并支持 CLI 覆盖。
- 隔离 CASE_C、pickle alias 和剪枝诊断逻辑，主训练链路不加载 legacy 模块。
- 删除未使用的 `extra_modules`、CUTLASS、Mamba、自定义 CUDA 及历史备份代码。
- 打通真实数据训练、checkpoint 重载、PT/ONNX 导出与数值一致性验证。
- 从全新 Conda 环境完成最终安装、模型构建、训练和 ONNX 验收。

## 详细改动

### 1. 数据与配置契约

- 标准数据集名称统一为 `Port_Defect`。
- 活动配置不再使用 `DAD030`；该名称仅保留在 `legacy_compat` 历史诊断记录中。
- 新增仓库相对数据示例：
  `training_project/datasets/port_defect.example.yaml`。
- 新增配置解析模块 `training_project/config.py`。
- 新增 baseline 与 RuleLoss smoke 配置：
  - `training_project/configs/port_defect_baseline.yaml`
  - `training_project/configs/port_defect_ruleloss_smoke.yaml`
- 重构 `training_project/train.py`：
  - 使用配置文件定义模型、数据、训练参数和 loss；
  - 支持 `--data`、`--epochs`、`--batch`、`--device`、`--name` 等 CLI 覆盖；
  - 数据缺失时在模型训练前快速失败；
  - 每次运行生成包含 Git、模型、类别、随机种子和 criterion 状态的
    `run_manifest.json`。
- 修正 `environment.yml` 的 PyTorch/CUDA channel 与依赖声明，并补充
  `timm==1.0.15`、ONNX 和 ONNX Runtime。

### 2. 单向模块注册

- 新增通用 vendor 扩展接口：
  `ultralytics-main/ultralytics/nn/extensions.py`。
- 新增项目侧统一安装入口：`defect_modules.integration.install()`。
- 删除 vendor `tasks.py` 对 `defect_modules.registry` 的直接导入。
- 明确注册元数据：
  - `CSPStage` 注入输入通道并在模块内部处理 repeat；
  - `RepHFE` 注入输入通道，不消费内部 repeat。
- 未安装项目扩展时，自定义 YAML token 会给出明确错误；安装后可正常构建。
- `install()` 支持幂等重复调用，不重复注册或产生额外副作用。
- `defect_modules.blocks` 改为标准包导入，不再通过文件路径动态加载。

### 3. RuleLoss 解耦

- 删除 `defect_modules.patch` 全局 monkey patch 机制。
- 在 vendor 扩展层加入 detection criterion factory。
- baseline 使用原生 `ultralytics.utils.loss.v8DetectionLoss`。
- 仅当配置中的 `loss.rule.enabled: true` 时使用
  `defect_modules.loss.v8DetectionLoss`。
- 通过 trainer callback 同步当前 epoch 与总 epoch。
- 将 RuleLoss 版本、超参数、启用状态、实际 criterion、更新次数和最终
  `lambda_rule` 写入运行清单。
- 增加配置参数校验、epoch schedule 边界验证和相同输入 loss 差异验证。

### 4. Legacy 隔离与运行时精简

- CASE_C PT、C2f_v2、pickle 兼容和旧剪枝诊断统一移动到
  `legacy_compat`。
- 主训练、验证和导出链路不导入 `ultralytics.nn.extra_modules`。
- 删除 729 个未使用的 tracked legacy 文件，包括：
  - CUTLASS；
  - Mamba/selective scan；
  - DCNv3/DCNv4；
  - 大量历史实验 block/head；
  - 自定义 CUDA 构建代码；
  - loss 备份文件。
- 保留的 CASE_C 工具只进行依赖诊断，不参与主模型构建。
- 已知 CASE_C checkpoint 通过 SHA-256 快速拒绝，不尝试强制转换目标拓扑。
- 锁定 bundled Ultralytics 版本为 `8.2.50`，并记录项目补丁边界。

### 5. 训练闭环

- 在真实 `Port_Defect` 数据集上完成 baseline 1 epoch smoke train。
- 在真实数据上完成 RuleLoss 1 epoch smoke train。
- 在全新 `harpnet_acceptance` 环境再次完成 baseline 1 epoch smoke train。
- 验证范围包括：
  - 数据加载；
  - 前向与反向传播；
  - 优化器更新；
  - finite loss/metrics；
  - best/last checkpoint 保存；
  - 新进程 checkpoint 重载；
  - 固定输入推理；
  - legacy 模块未加载。
- 训练输出统一写入 Git 忽略目录 `training_project/runs/`。

### 6. PT/ONNX 交付链

- 重写 `export_pipeline/export_onnx.py`，只接受：
  - 已完成训练运行的 checkpoint；
  - 与目标 YAML 拓扑一致的模型；
  - 不依赖 legacy 模块的 canonical PT。
- 删除训练仓库中的 RKNN 转换实现和旧的硬编码导出脚本。
- 新增 CASE_C 拒绝测试和 PT/ONNX 数值一致性验证。
- artifact manifest 使用相对路径，记录：
  - PT、ONNX 和 YAML 哈希；
  - 类别映射；
  - 输入尺寸；
  - 输入输出名称；
  - 预处理约定；
  - 参数量和层数；
  - runtime 版本；
  - 数值验证结果。
- 导出物统一写入 Git 忽略目录 `export_pipeline/outputs/`。

### 7. 自动化验证与文档

- 新增统一验收入口：

  ```powershell
  python training_project/verify_all.py
  ```

- 验收覆盖环境版本、配置、registry、import 边界、注册幂等性、block shape、
  模型签名、RuleLoss factory、legacy 隔离和导出拒绝逻辑。
- 更新 README、架构说明、环境说明、仓库内容策略、上传检查清单和当前验证状态。
- 新增阶段 0～7 验证报告：`docs/stage_reports/`。

## 验收结果

| 检查项 | 结果 |
| --- | --- |
| 全新 Conda 环境创建 | PASSED |
| Python / PyTorch / CUDA | 3.10.20 / 2.1.2 / 12.1 |
| CUDA 可用性 | PASSED，NVIDIA GeForce RTX 4090 |
| 综合自动化检查 | PASSED，11/11 |
| 目标模型参数量 | 2,308,655 |
| 目标模型层数 | 25 |
| 自定义模块数量 | CSPStage=4，RepHFE=2 |
| vendor 到项目反向依赖 | 无 |
| 主链路 legacy import | 无 |
| baseline 真实数据 smoke | PASSED |
| RuleLoss 真实数据 smoke | PASSED |
| 全新环境 baseline smoke | PASSED |
| checkpoint 重载与推理 | PASSED |
| CASE_C 拒绝 | PASSED |
| ONNX checker | PASSED |
| PT/ONNX 数值一致性 | PASSED |
| Git 输出隔离 | PASSED |

ONNX 固定输入输出形状为 `[1, 9, 8400]`。数值一致性结果：

| 输出 | 最大绝对误差 | 平均绝对误差 | 阈值 |
| --- | ---: | ---: | ---: |
| boxes | 0.0244140625 | 0.0006894757 | 0.05 / 0.001 |
| scores | 0.0000808239 | 0.0000007331 | 0.001 / 0.0001 |

工程 smoke artifact 哈希：

| 文件 | SHA-256 |
| --- | --- |
| baseline PT | `2bfecdff4022b16ed514d4d6ee4529122742fbc4062c480b842c26eaf1925f4a` |
| ONNX | `9a97ef15ce8ef30d45179348f243de698ae93e873a14c42bec21e197b28fbddf` |
| model YAML | `32c400d0a2ee4d6619eaeeabf638a1bb5ba2bd6682affe5cca1577af6c56b4e2` |

## 安装与快速验证

在仓库根目录执行：

```powershell
conda env create -n harpnet_acceptance -f environment.yml
conda activate harpnet_acceptance
python training_project/verify_all.py
```

数据应以仓库相对结构提供：

```text
datasets/Port_Defect/
  data.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

数据位于其他位置时可通过 CLI 覆盖：

```powershell
python -m training_project.train `
  --config training_project/configs/port_defect_baseline.yaml `
  --data D:/datasets/Port_Defect/data.yaml
```

## Breaking changes

- `DAD030` 活动配置已移除，请迁移到 `Port_Defect`。
- 不再支持通过 `defect_modules.patch` 激活 RuleLoss。
- 自定义模型构建前必须调用 `defect_modules.integration.install()`；统一训练入口已自动处理。
- `ultralytics.nn.extra_modules` 不再属于主运行时，依赖其 pickle 类路径的旧 PT
  只能使用 `legacy_compat` 诊断，不能作为目标 YAML 或 canonical ONNX 的输入。
- 训练仓库不再提供 ONNX 到 RKNN 的转换入口。
- 旧的硬编码绝对路径和历史导出脚本已移除。

## 已知限制

- 本 Release 验证的是工程闭环，不包含正式长周期训练和精度调优。
- 当前 PT/ONNX 是 1 epoch smoke artifact，不应直接作为生产部署模型。
- CASE_C checkpoint 与目标 YAML 拓扑不同，本版本明确拒绝，不提供强制迁移。
- RKNN 转换、量化、板端性能和精度验证不在本仓库职责内。
- `environment.yml` 固定兼容版本线，但不是逐包 byte-for-byte lockfile。

## Commit history

| Commit | 阶段 | 内容 |
| --- | --- | --- |
| `eed3f41` | Stage 0 | 建立可追溯基线与验证报告 |
| `96f46ac` | Stage 1 | 统一数据和配置契约 |
| `7cd04ad` | Stage 2 | 建立单向模块注册机制 |
| `4f9c409` | Stage 3 | criterion factory 与 RuleLoss 解耦 |
| `ea9e8f4` | Stage 4 | legacy 隔离和 Ultralytics 精简 |
| `469c16f` | Stage 5 | 真实数据训练闭环 |
| `f3381ff` | Stage 6 | canonical PT/ONNX 交付契约 |
| `cbc9d69` | Stage 7 | 全新环境最终验收 |

## 升级建议

1. 在独立环境运行 `training_project/verify_all.py`。
2. 将数据配置名称和路径迁移到 `Port_Defect`。
3. 通过统一训练入口运行 baseline，确认 RuleLoss 为关闭状态。
4. 需要 RuleLoss 时使用显式配置，不再调用 patch API。
5. 重新训练生成与目标 YAML 一致的 checkpoint，不复用 CASE_C PT。
6. 使用 artifact manifest 将验证后的 ONNX 交接给 `331_PC_RKNN`。
