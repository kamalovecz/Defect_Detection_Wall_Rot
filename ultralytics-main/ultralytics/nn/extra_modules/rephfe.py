import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RepDWConv", "RepHFE"]


class RepDWConv(nn.Module):
    """
    Re-parameterizable depthwise convolution.
    Train: 3x3 DW + 1x1 DW + Identity
    Deploy: merged single 3x3 DW
    """

    def __init__(self, channels):
        super().__init__()
        self.deploy = False
        self.channels = channels

        self.rbr_reparam = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True)
        self.rbr_reparam.requires_grad_(False)

        self.rbr_dense = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_identity = nn.BatchNorm2d(channels)

    def forward(self, inputs):
        if hasattr(self, "deploy") and self.deploy:
            return self.rbr_reparam(inputs)
        return self.rbr_dense(inputs) + self.rbr_1x1(inputs) + self.rbr_identity(inputs)

    def switch_to_deploy(self):
        if hasattr(self, "deploy") and self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        self.__delattr__("rbr_identity")
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        kernel1x1_padded = F.pad(kernel1x1, [1, 1, 1, 1])
        return kernel3x3 + kernel1x1_padded + kernelid, bias3x3 + bias1x1 + biasid

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            kernel = torch.zeros(self.channels, 1, 3, 3, device=branch.weight.device)
            for i in range(self.channels):
                kernel[i, 0, 1, 1] = 1
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepHFE(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or in_channels

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.rep_dwc = RepDWConv(in_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.pwc = nn.Conv2d(in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.act1(self.rep_dwc(x))
        x = self.act2(self.bn2(self.pwc(x)))
        return x
