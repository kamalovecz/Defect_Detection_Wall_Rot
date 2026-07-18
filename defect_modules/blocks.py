"""RepHFE blocks externalized from the HARP-Net Ultralytics fork.

This file intentionally carries only RepHFE and its direct helper RepDWConv so
registry.py no longer needs to import RepHFE from extra_modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, RepConv

__all__ = ["RepDWConv", "RepHFE", "BasicBlock_3x3_Reverse", "SPP", "CSPStage"]


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

class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):
        super().__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        return x + y if self.shortcut else y


class SPP(nn.Module):
    def __init__(self, ch_in, ch_out, k, pool_size):
        super().__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module(f"pool{i}", pool)
            self.pool.append(pool)
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        return self.conv(torch.cat(outs, axis=1))


class CSPStage(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        n,
        block_fn="BasicBlock_3x3_Reverse",
        ch_hidden_ratio=1.0,
        act="silu",
        spp=False,
    ):
        super().__init__()
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == "BasicBlock_3x3_Reverse":
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in, ch_hidden_ratio, ch_mid, shortcut=True),
                )
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module("spp", SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        return self.conv3(torch.cat(mid_out, axis=1))

