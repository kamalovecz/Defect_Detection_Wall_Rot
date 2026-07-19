# Ablation Integration Stage 1 Source Archive

阶段：1 - 历史 YAML 归档与 provenance manifest

状态：PASSED

修改内容：六份历史 YAML 逐字节归档到 `training_project/models/ablations/source/`；新增 `training_project/ablations/manifest.yaml` 和 `verify_archive.py`。

验证命令：

```text
python training_project/ablations/verify_archive.py
python training_project/ablations/verify_archive.py --check-source
python training_project/ablations/verify_archive.py --check-source --check-git
```

通过项：

- 六份归档 YAML 的 SHA-256 和字节数与外部历史源完全一致。
- `.gitattributes` 对归档目录禁用文本归一化，提交到 Git 的 YAML blob 也与历史源逐字节一致。
- B4 canonical 与归档源哈希一致，继续保持唯一活动真源。
- 归档为闭集：仅允许 manifest 中六份 YAML 和 README。
- 所有仓库路径必须为相对路径且解析后不得逃逸允许根目录。
- provenance basename、扩展名、SHA 格式和历史权重元数据均受校验。
- 历史 PT 未复制进 Git，仅登记 SHA 和拓扑状态。

负向测试：临时加入第七份 YAML 时验证失败；临时将 archive 路径改为 `../escape.yaml` 时验证失败；测试文件均已清理。

首次对抗性审查：FAILED，指出额外文件和路径逃逸可伪通过，以及状态语义混淆。

修复后对抗性重审：PASSED；无必须修复项。

提交后完整性复查发现通用 YAML 行尾规则会把部分 CRLF 归档规范化为 LF；已增加归档专用 `-text` 规则、重新写入 Git blob，并将 HEAD blob 校验加入验证脚本。最终对抗性重审：PASSED；确认归档规则为 `text: unset`、HEAD blob 与外部历史源逐字节一致，无必须修复项。

阶段 4 建立 canonical 哈希契约前再次审计发现，活动 B4 canonical 不在 source 通配保护范围内，其旧 HEAD blob 仍被规范化。已为该唯一活动历史同源文件增加精确 `-text` 规则、重写 blob，并把 B4 canonical HEAD 校验纳入 `--check-git`，防止新克隆中的行尾漂移。

失败项：无（修复后完整重跑通过）。

风险或残留：除 B4 外的 canonical 文件尚未生成；`source_root` 当前为 provenance 信息而逐项 `source_path` 是实际外部校验路径；source 归档、source SHA/bytes 和 `.gitattributes` 的 `-text` 保护规则在后续阶段禁止修改。

是否允许进入下一阶段：是。
