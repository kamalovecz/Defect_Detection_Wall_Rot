# Project Architecture

The dependency direction is `training_project -> defect_modules -> ultralytics-main`. The runtime exposes generic model-module and detection-criterion registration APIs; it does not know about HARP-Net classes.

`defect_modules.integration.install()` registers CSPStage with channel injection and internal repeat handling, RepHFE with channel injection, and Detect_LSCSBD as a multi-input detection head. It also selects either the native detection criterion or RuleLoss factory from validated configuration.

The ablation dependency path remains one-way: canonical YAML/config -> `defect_modules.integration.install()` -> generic vendor extension registry. Historical source YAMLs are immutable evidence under `training_project/models/ablations/source`; they are never loaded as the active runtime truth.

The active runtime contains no `extra_modules` tree. Historical pickle compatibility is isolated under `legacy_compat` and must never be imported by training or export.

The training boundary ends at a numerically validated ONNX artifact. The downstream `331_PC_RKNN` repository owns ONNX-to-RKNN conversion, device transfer, board inference, and PT/RKNN alignment.
