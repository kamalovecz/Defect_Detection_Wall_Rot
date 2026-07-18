# Export boundary

This package accepts a completed, topology-compatible training checkpoint and produces a self-contained PT/ONNX engineering artifact. It does not convert ONNX to RKNN.

The artifact manifest uses relative filenames only and records hashes, classes, input size, preprocessing, topology, runtime version, and PT/ONNX consistency results. RKNN conversion and board validation belong to `331_PC_RKNN`.
