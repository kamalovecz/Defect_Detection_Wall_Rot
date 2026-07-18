from __future__ import annotations

import json
import sys

import onnx
import onnxruntime
import torch
import ultralytics


def main() -> int:
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ultralytics": ultralytics.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Expected Python 3.10, got {result['python']}")
    if not torch.__version__.startswith("2.1.") or torch.version.cuda != "12.1":
        raise RuntimeError(f"Unexpected PyTorch/CUDA compatibility line: {result}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the acceptance environment")
    if ultralytics.__version__ != "8.2.50":
        raise RuntimeError(f"Unexpected Ultralytics version: {ultralytics.__version__}")
    print(json.dumps({"status": "ok", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
