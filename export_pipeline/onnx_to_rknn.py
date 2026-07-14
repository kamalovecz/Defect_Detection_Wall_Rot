"""ONNX to RKNN placeholder.

This stage must remain free of ultralytics, defect_modules, and training-time
imports. Add rknn-toolkit2 conversion logic only when entering RKNN validation.
"""

if __name__ == "__main__":
    raise SystemExit("RKNN conversion is intentionally not executed in the MVP setup step.")
