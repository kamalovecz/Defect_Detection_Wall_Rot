# HARP-Net 缺陷检测项目

这是一个面向金属/墙面缺陷检测的 HARP-Net 研究工程。仓库已经整理成可继续开发、可训练、可上传 GitHub 的源码基线版本，重点支持后续不断加入新的自定义结构模块、损失函数、检测头、模型 YAML、训练验证脚本和导出流程。

当前仓库不是数据集仓库，也不是模型权重仓库。数据集通过百度网盘等外部渠道分发，阶段性 `.pt` 模型文件通过 GitHub Release 或其他 artifact 平台发布。

## 快速启动

### 1. 克隆仓库

```powershell
git clone git@github.com:kamalovecz/Defect_Detection_Wall_Rot.git
cd Defect_Detection_Wall_Rot
```

### 2. 准备环境

推荐使用项目当前验证过的环境名 `yolo_ultra`。如果本机还没有环境，可以参考 `environment.yml` 创建：

```powershell
conda env create -f environment.yml
conda activate yolo_ultra
```

如果你已经有可用环境，至少需要 Python 3.10、PyTorch、torchvision、OpenCV、PyYAML 和 Ultralytics 相关依赖。仓库中的 `training_project/train.py` 会优先把当前仓库根目录和 `ultralytics-main` 加入 `sys.path`。

### 3. 是否需要额外放入干净原版 Ultralytics 源码？

不需要。

本仓库已经包含项目所需的最小修改版 Ultralytics 源码：

```text
ultralytics-main/
```

不要把干净原版 Ultralytics 覆盖到这个目录。原因是当前项目依赖 `ultralytics-main/ultralytics/nn/tasks.py` 中的定制修改：

- 已移除 `from ultralytics.nn.extra_modules import *`。
- YAML token 解析被收紧到官方模块和 `defect_modules.registry.YAML_BLOCKS`。
- `CSPStage`、`RepHFE` 等项目模块通过 registry/patch 接入。

如果换成干净原版 Ultralytics，模型 YAML 中的自定义模块大概率无法解析，导入边界验证也会失效。

如果后续确实需要同步新版 Ultralytics，建议新建分支，把上游源码作为参考合并，再重新迁移 `tasks.py` 的边界修改和自定义模块解析逻辑，而不是直接覆盖当前目录。

### 4. 准备数据集

数据集不进入 Git。当前数据集通过百度网盘分享：

```text
文件名: Port_Defect.zip
链接: https://pan.baidu.com/s/1I62ItneKTMSTZhb54oNGAQ?pwd=0331
提取码: 0331
```

克隆仓库后，在仓库根目录新建本地数据集目录：

```powershell
mkdir datasets
```

将 `Port_Defect.zip` 放到 `datasets/` 下并解压，推荐得到下面的结构：

```text
Defect_Detection_Wall_Rot/
  datasets/
    Port_Defect/
      data.yaml
      images/
      labels/
```

`datasets/` 是本地数据目录，已被 Git 忽略，不会上传到仓库。

解压后修改 `datasets/Port_Defect/data.yaml`。如果压缩包内的 `data.yaml` 使用旧的绝对路径，需要改成当前仓库下的相对路径，例如：

```yaml
path: datasets/Port_Defect
train: images/train
val: images/val
test: images/test
# names 字段保持压缩包内 data.yaml 的原有类别编号和类别名称
```

如果 `Port_Defect.zip` 解压后多了一层同名目录，例如 `datasets/Port_Defect/Port_Defect/data.yaml`，则按实际目录调整 `path`，或把内部文件整理到推荐结构中。

### 5. 运行基础验证

```powershell
python training_project\verify_registry.py
python training_project\verify_tasks_import_boundary.py
python training_project\verify_external_blocks.py
```

这三项用于确认：

- `YAML_BLOCKS` 只暴露当前项目允许的模块。
- 导入 `ultralytics.nn.tasks` 不会触发旧的 `extra_modules` 通配符导入。
- 目标 YAML 可以用 `defect_modules.blocks` 中的 `CSPStage` 和 `RepHFE` 正常构建。

### 6. 运行 1 epoch smoke train

```powershell
python training_project\train.py `
  --model training_project\models\B4_A-GFPN_RepHFE_target.yaml `
  --data datasets\Port_Defect\data.yaml `
  --epochs 1 `
  --batch 4 `
  --imgsz 640 `
  --workers 0 `
  --name harpnet_smoke
```

训练输出默认写入：

```text
training_project\runs\
```

`runs/` 不进入 Git。

### 7. 正式训练示例

```powershell
python training_project\train.py `
  --model training_project\models\B4_A-GFPN_RepHFE_target.yaml `
  --data datasets\Port_Defect\data.yaml `
  --epochs 100 `
  --batch 16 `
  --imgsz 640 `
  --workers 0 `
  --name harpnet_b4_exp01
```

## 仓库目录说明

```text
defect_modules/
  blocks.py       自定义结构模块，例如卷积、CSP、注意力、高频增强、特征融合
  loss.py         RuleLoss、蒸馏损失、困难样本加权损失等训练组件
  registry.py     模型 token、损失对象、参数解析类型和兼容映射
  patch.py        Ultralytics 命名空间注入和旧 PT 兼容

training_project/
  models/         不同模块组合的 YAML
  datasets/       数据集 YAML 示例，不存放真实图片和标签
  train.py        统一训练入口
  verify_*.py     结构、来源、维度、PT 加载和训练验证脚本

datasets/
  本地解压的数据集目录，例如 datasets/Port_Defect/，不进入 Git

export_pipeline/
  export_onnx.py                 ONNX 导出入口
  verify_onnx_consistency.py     ONNX 一致性验证
  load_canonical_model.py        canonical 模型加载

ultralytics-main/
  项目所需的最小修改版 Ultralytics 源码

docs/
  当前验证状态、环境说明、仓库策略和上传检查清单
```

## 自建模块训练流程

后续新增模块时，推荐固定沿用下面这条链路：

```text
defect_modules/blocks.py
        ->
defect_modules/registry.py
        ->
defect_modules/patch.py
        ->
training_project/models/*.yaml
        ->
training_project/train.py
        ->
Ultralytics parse_model
        ->
模型训练
```

### 1. 在 `blocks.py` 实现模块

例如新增一个普通高频增强模块：

```python
import torch.nn as nn


class NewHFEBlock(nn.Module):
    def __init__(self, c1, c2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

普通单输入、单输出模块通常满足：

```text
输入: B x C1 x H x W
输出: B x C2 x H x W
```

### 2. 在 `registry.py` 注册 YAML token

```python
from defect_modules.blocks import CSPStage, RepHFE, NewHFEBlock

YAML_BLOCKS = {
    "CSPStage": CSPStage,
    "RepHFE": RepHFE,
    "NewHFEBlock": NewHFEBlock,
}
```

注意：不要把历史 `extra_modules` 的大量模块名塞回 `YAML_BLOCKS`。旧 PT 兼容路径应放在 `PICKLE_COMPAT_TYPES`，不要混入 YAML 主链路。

### 3. 通过 `patch.py` 注入 Ultralytics 命名空间

训练入口会先执行：

```python
from defect_modules.patch import apply

apply(
    pickle_compat=False,
    legacy_aliases=False,
    strict=True,
)
```

这一步会把 registry 中允许的项目模块注入到 Ultralytics 构建模型时使用的命名空间。

### 4. 在 YAML 中使用新模块

例如新建：

```text
training_project/models/new_harpnet.yaml
```

YAML 示例：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, NewHFEBlock, [128, 3]]
  - [-1, 2, CSPStage, [128]]
  - [-1, 1, RepHFE, [128]]
```

这里 YAML 显式提供 `c2=128` 和 `kernel_size=3`，`c1` 需要由 `parse_model()` 根据上一层输出通道自动补入，最终构造应类似：

```python
NewHFEBlock(c1, 128, 3)
```

如果新模块涉及下面这些情况，就不能只注册 token，还需要同步扩展参数解析逻辑：

- 自动推导输入通道 `c1`
- 自动计算输出通道 `c2`
- 内部重复次数 `n`
- 多层输入索引
- `Concat` 或 `Add` 类多输入
- 输出多个特征图
- 特殊宽度/深度缩放
- 检测头输入通道列表

### 5. 通过统一入口训练

```powershell
python training_project\train.py `
  --model training_project\models\new_harpnet.yaml `
  --data datasets\Port_Defect\data.yaml `
  --epochs 100 `
  --batch 16 `
  --imgsz 640 `
  --workers 0 `
  --name new_harpnet_exp01
```

## Loss、Head 与结构模块的边界

结构模块、特征融合模块、backbone 模块和后续自定义检测头属于 YAML 结构组件，应通过 `YAML_BLOCKS` 进入模型构建链路。

损失函数、蒸馏目标、剪枝约束和困难样本加权属于训练组件。把 loss 写入 `LOSS_OBJECTS` 只表示对象可被发现，不代表训练时自动生效。它还必须被 trainer 或 criterion 实际调用，例如接入 `v8DetectionLoss` 或项目自定义 Trainer。

推荐长期保持下面的职责边界：

```text
defect_modules/blocks.py
  模型结构模块，例如卷积、CSP、注意力、高频增强、特征融合

defect_modules/heads.py
  自定义检测头，后续模块增加较多时可以从 blocks.py 单独拆出

defect_modules/loss.py
  RuleLoss、蒸馏损失、困难样本加权损失

defect_modules/registry.py
  模型 token、损失对象、参数解析类型和兼容映射

defect_modules/patch.py
  Ultralytics 命名空间注入和旧 PT 兼容

training_project/models
  不同模块组合的 YAML

training_project/train.py
  统一训练入口

training_project/verify_*.py
  结构、来源、维度、PT 加载和训练验证
```

## 新模块至少需要完成的验证

### 1. 模块 shape 验证

```python
import torch
from defect_modules.blocks import NewHFEBlock

x = torch.randn(1, 64, 80, 80)
module = NewHFEBlock(64, 128)
y = module(x)

assert y.shape == (1, 128, 80, 80)
```

### 2. registry 和 patch 验证

```python
from defect_modules.patch import apply

apply(pickle_compat=False, legacy_aliases=False, strict=True)

import ultralytics.nn.tasks as tasks

assert tasks.NewHFEBlock.__module__ == "defect_modules.blocks"
```

### 3. YAML 构建验证

```python
from ultralytics import YOLO

model = YOLO("training_project/models/new_harpnet.yaml")
```

### 4. 短训练验证

```powershell
python training_project\train.py `
  --model training_project\models\new_harpnet.yaml `
  --data datasets\Port_Defect\data.yaml `
  --epochs 1 `
  --batch 4 `
  --imgsz 640 `
  --workers 0 `
  --name new_module_smoke
```

确认 loss 正常、输出目录生成、模块真实来源为 `defect_modules.blocks`。

## 导出与部署约束

如果模块最终要部署到 RK3588S/RKNN，设计阶段就要控制算子范围。

优先使用：

```text
Conv2d
Depthwise Conv2d
BatchNorm2d
SiLU 或 ReLU
Add
Concat
Interpolate
MaxPool
普通 reshape 和 permute
```

谨慎使用：

```text
依赖输入数据的 Python 分支
自定义 CUDA 算子
复杂 einsum
动态索引
动态卷积核生成
不规则 grid_sample
ONNX 不支持的算子
RKNN 不支持的特殊激活或归一化
```

## 数据集和模型产物发布方式

### 数据集

数据集不进入 Git。当前公开数据集信息：

```text
文件名: Port_Defect.zip
链接: https://pan.baidu.com/s/1I62ItneKTMSTZhb54oNGAQ?pwd=0331
提取码: 0331
推荐解压路径: datasets/Port_Defect/
训练数据描述文件: datasets/Port_Defect/data.yaml
```

后续维护数据集时，建议同时记录：

- 数据集版本号
- 数据集目录结构说明
- `data.yaml` 的类别编号和类别名称
- 数据集 SHA256 或文件数量统计

### 阶段性 PT 模型

`.pt`、`.pth`、ONNX、RKNN、engine、TorchScript 等产物不提交到 Git。

阶段性 PT 模型建议放到 GitHub Release。命名示例：

```text
harpnet-baseline-b4-epochXXX.pt
harpnet-cspstage-rephfe-b4-epochXXX.pt
harpnet-ruleloss-b4-epochXXX.pt
harpnet-deploy-candidate-b4-epochXXX.pt
```

每个 Release 建议附带：

- `.pt` checkpoint
- 对应模型 YAML
- 数据集版本或百度网盘数据集标识
- 完整训练命令
- 指标摘要
- SHA256 校验值
- 已知限制，例如 RuleLoss 是否启用、是否通过 ONNX 导出验证

推荐 tag：

```text
weights-vYYYYMMDD
smoke-vYYYYMMDD
deploy-candidate-vYYYYMMDD
```

当前仓库提交本身不包含 Release 资产。等数据集和阶段性 PT 文件确定后，再通过 GitHub Release 上传。

## 当前真实状态

在声明结果前，请先阅读：

```text
docs/VALIDATION_STATUS.md
```

当前快照的真实状态是：

- `RuleLoss` 已存在，但当前训练路径中尚未真正激活。
- 固定 PT 是 `CASE_C`，与目标 YAML 拓扑不严格一致。
- ONNX 导出当前是 `SKIPPED_CASE_C`，没有 canonical state_dict，因此不能宣称部署导出已完成。
- 数据集和阶段性 checkpoint 均外置，不进入 Git。

## 长期维护原则

后续模块持续迭代时，尽量保持三条原则：

```text
模块实现集中在 defect_modules
模块发现和解析规则集中在 registry
训练和导出均通过固定项目入口
```

这样可以通过新增 YAML 和实验配置组合出不同模型，例如：

```text
Baseline + 新模块 A
Baseline + 新模块 B
Baseline + A + B
HARP-Net + 新模块 C
剪枝学生模型 + 蒸馏模块
部署版模型 + 可重参数化模块
```
