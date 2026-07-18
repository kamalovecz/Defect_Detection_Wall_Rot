from __future__ import annotations

from export_onnx import reject_known_case_c


def main() -> int:
    rejected = False
    try:
        reject_known_case_c("BA5CC233EEA726226B3EFCED7200018F799CB702DB4A7F688BD8B06212B71656")
    except RuntimeError:
        rejected = True
    if not rejected:
        raise RuntimeError("Known CASE_C checkpoint hash was accepted")
    print('{"status":"ok","case_c_rejected":true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
