"""MBRConv5: multi-branch re-parameterizable 5x5 convolution block."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("MBRConv5",)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fuse Conv2d + BatchNorm2d into equivalent conv weights and bias."""
    w = conv.weight
    if conv.bias is None:
        b = torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
    else:
        b = conv.bias
    std = torch.sqrt(bn.running_var + bn.eps)
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    w_fused = w * t
    b_fused = (b - bn.running_mean) / std * bn.weight + bn.bias
    return w_fused, b_fused


def _pad_to_5x5(weight: torch.Tensor):
    """Center-pad kernels to 5x5."""
    kh, kw = weight.shape[-2:]
    ph = (5 - kh) // 2
    pw = (5 - kw) // 2
    return F.pad(weight, (pw, pw, ph, ph))


class MBRConv5(nn.Module):
    """
    Multi-branch re-parameterization block.

    Train-time: 5x5/3x3/1x1/cross-shaped branches + BN-enhanced branches + 1x1 fusion.
    Deploy-time: equivalent single 5x5 conv via `switch_to_deploy()`.
    """

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4, deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        self.deploy = deploy
        self.mid_channels = out_channels * rep_scale

        if deploy:
            self.deploy_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True)
            return

        # Raw branches
        self.conv5 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_h = nn.Conv2d(in_channels, self.mid_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv_v = nn.Conv2d(in_channels, self.mid_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)

        # BN branches
        self.bn5 = nn.BatchNorm2d(self.mid_channels)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.bn3 = nn.BatchNorm2d(self.mid_channels)
        self.bnh = nn.BatchNorm2d(self.mid_channels)
        self.bnv = nn.BatchNorm2d(self.mid_channels)

        # Branch fusion
        self.conv_out = nn.Conv2d(self.mid_channels * 10, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_out.weight.requires_grad = False
        self.weight1 = nn.Parameter(torch.empty_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)

    def forward(self, x: torch.Tensor):
        if self.deploy:
            return self.deploy_conv(x)

        x1 = self.conv5(x)
        x2 = self.conv1(x)
        x3 = self.conv3(x)
        x4 = self.conv_h(x)
        x5 = self.conv_v(x)

        merged = torch.cat(
            [x1, x2, x3, x4, x5, self.bn5(x1), self.bn1(x2), self.bn3(x3), self.bnh(x4), self.bnv(x5)],
            dim=1,
        )
        final_weight = self.conv_out.weight + self.weight1
        return F.conv2d(merged, final_weight, self.conv_out.bias, stride=1, padding=0)

    def slim(self):
        """Export equivalent single 5x5 conv parameters (weight, bias)."""
        if self.deploy:
            return self.deploy_conv.weight.data, self.deploy_conv.bias.data

        # Raw branch kernels/biases
        k5, b5 = self.conv5.weight, self.conv5.bias
        k1, b1 = _pad_to_5x5(self.conv1.weight), self.conv1.bias
        k3, b3 = _pad_to_5x5(self.conv3.weight), self.conv3.bias
        kh, bh = _pad_to_5x5(self.conv_h.weight), self.conv_h.bias
        kv, bv = _pad_to_5x5(self.conv_v.weight), self.conv_v.bias

        # BN branch kernels/biases (conv+bn fused)
        k5b, b5b = _fuse_conv_bn(self.conv5, self.bn5)
        k1b, b1b = _fuse_conv_bn(self.conv1, self.bn1)
        k3b, b3b = _fuse_conv_bn(self.conv3, self.bn3)
        khb, bhb = _fuse_conv_bn(self.conv_h, self.bnh)
        kvb, bvb = _fuse_conv_bn(self.conv_v, self.bnv)

        k5b = _pad_to_5x5(k5b)
        k1b = _pad_to_5x5(k1b)
        k3b = _pad_to_5x5(k3b)
        khb = _pad_to_5x5(khb)
        kvb = _pad_to_5x5(kvb)

        kernel_cat = torch.cat([k5, k1, k3, kh, kv, k5b, k1b, k3b, khb, kvb], dim=0)  # [10*mid, in, 5, 5]
        bias_cat = torch.cat([b5, b1, b3, bh, bv, b5b, b1b, b3b, bhb, bvb], dim=0)     # [10*mid]

        compress = (self.conv_out.weight + self.weight1).reshape(self.out_channels, -1)  # [out, 10*mid]
        fused_w = torch.matmul(compress, kernel_cat.reshape(kernel_cat.shape[0], -1))
        fused_w = fused_w.reshape(self.out_channels, self.in_channels, 5, 5)

        fused_b = torch.matmul(compress, bias_cat)
        if self.conv_out.bias is not None:
            fused_b = fused_b + self.conv_out.bias

        return fused_w, fused_b

    def switch_to_deploy(self):
        """Convert train-time multi-branch graph to a single conv for inference."""
        if self.deploy:
            return

        kernel, bias = self.slim()
        self.deploy_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.deploy_conv = self.deploy_conv.to(device=kernel.device, dtype=kernel.dtype)
        self.deploy_conv.weight.data.copy_(kernel)
        self.deploy_conv.bias.data.copy_(bias)

        # Remove train-time branches to reduce memory/runtime overhead
        del self.conv5, self.conv1, self.conv3, self.conv_h, self.conv_v
        del self.bn5, self.bn1, self.bn3, self.bnh, self.bnv
        del self.conv_out, self.weight1

        self.deploy = True
