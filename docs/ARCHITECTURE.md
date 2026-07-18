# Project Architecture

The dependency direction is `training_project -> defect_modules -> ultralytics-main`. The runtime exposes generic model-module and detection-criterion registration APIs; it does not know about HARP-Net classes.

`defect_modules.integration.install()` registers CSPStage with channel injection and internal repeat handling, registers RepHFE with channel injection, and selects either the native detection criterion or RuleLoss factory from validated configuration.

The active runtime contains no `extra_modules` tree. Historical pickle compatibility is isolated under `legacy_compat` and must never be imported by training or export.

The training boundary ends at a numerically validated ONNX artifact. The downstream `331_PC_RKNN` repository owns ONNX-to-RKNN conversion, device transfer, board inference, and PT/RKNN alignment.
