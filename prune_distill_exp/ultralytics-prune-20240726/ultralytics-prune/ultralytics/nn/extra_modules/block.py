import torch.nn as nn

from ultralytics.nn.modules.block import C2f, RepNCSPELAN4


class _UnavailableExtraModule(nn.Module):
    """Placeholder for optional third-party blocks not used by vanilla YOLOv8 pruning."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "This optional extra module is not available in the current pruning smoke path."
        )


Faster_Block = _UnavailableExtraModule
Fusion = _UnavailableExtraModule
IFM = _UnavailableExtraModule
InjectionMultiSum_Auto_pool = _UnavailableExtraModule
TopBasicLayer = _UnavailableExtraModule
SimFusion_3in = _UnavailableExtraModule
SimFusion_4in = _UnavailableExtraModule
AdvPoolFusion = _UnavailableExtraModule
PyramidPoolAgg = _UnavailableExtraModule
RepVGGBlock = _UnavailableExtraModule
RepConvN = _UnavailableExtraModule
Star_Block = _UnavailableExtraModule
C2f_Faster = _UnavailableExtraModule
C2f_EMBC = _UnavailableExtraModule
C2f_Star = _UnavailableExtraModule
MBConv = _UnavailableExtraModule
RepNCSP = _UnavailableExtraModule
