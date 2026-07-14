# ultralytics/utils/amp_compat.py
import torch

# 统一导出两个名字：AMP_CUSTOM_FWD / AMP_CUSTOM_BWD
# - 在 torch>=2.3 时转到 torch.amp.custom_* 并强制 device_type='cuda'
# - 在 torch<=2.2 时回退到 torch.cuda.amp.custom_*
try:
    from torch.amp import custom_fwd as _cfwd, custom_bwd as _cbwd

    def AMP_CUSTOM_FWD(*args, **kwargs):
        # 既支持 @AMP_CUSTOM_FWD 也支持 @AMP_CUSTOM_FWD(cast_inputs=...)
        if args and callable(args[0]) and not kwargs:
            return _cfwd(device_type='cuda')(args[0])
        return _cfwd(device_type='cuda', **kwargs)

    def AMP_CUSTOM_BWD(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return _cbwd(device_type='cuda')(args[0])
        return _cbwd(device_type='cuda', **kwargs)

except Exception:
    from torch.cuda.amp import custom_fwd as AMP_CUSTOM_FWD, custom_bwd as AMP_CUSTOM_BWD
