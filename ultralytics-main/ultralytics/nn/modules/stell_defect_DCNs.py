# custom_deform.py
# -*- coding: utf-8 -*-
"""
Custom deformable/attention modules for integrating into Ultralytics YOLOv8.

This file defines:
- RDCN family: RDCN, Bottleneck_RDCN, C3_RDCN, C2f_RDCN
- GVHA: Global Vertical & Horizontal Attention
- GRDCN family: GRDCN (deformable conv + GVHA offsets), Bottleneck_GRDCN, C3_GRDCN, C2f_GRDCN

Notes:
- GRDCN includes an export fallback so ONNX/RKNN 导出时自动退化为标准 Conv（兼容部署）。
- GVHA 使用均值池化实现“H/W方向全局聚合”，避免 AdaptiveAvgPool2d(None, ·) 的非法参数。
- 将本文件放到 ultralytics/nn/modules/ 下，并在 __init__.py 里 `from .custom_deform import *` 注册即可。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision  # 确保 torchvision.ops 注册 deform_conv2d
except Exception:  # noqa: BLE001
    torchvision = None

from ultralytics.nn.modules import Conv, C2f, C3, Bottleneck
from ultralytics.nn.modules.conv import autopad


# ------------------------------------------------------------
# RDCN: Restricted Deformable Convolution (offsets via conv)
# ------------------------------------------------------------
class RDCN(nn.Module):
    """Restricted deformable-like convolution implemented by grid sampling logic."""

    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None, modulation=True, act="silu"):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.zero_padding = nn.ZeroPad2d(padding)
        # 注意：原作将 stride 设为 kernel_size，这里按原逻辑保留（保持行为一致）
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.bn = nn.BatchNorm2d(outc)
        self.act = RDCN.get_activation(act, inplace=True)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # noqa: ANN001
        # 缩小 offset 分支的梯度
        if grad_input is not None:
            grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        if grad_output is not None:
            grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.dtype
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)  # (b, c, h*ks, w*ks)
        out = self.act(self.bn(self.conv(x_offset)))
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing="ij",
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0).view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing="ij",
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    @staticmethod
    def Fc(x):
        return (2 * (1 / (1 + torch.exp(-x))) - 1)

    def _get_p(self, offset, dtype):
        offset_limit = 4
        offset = self.Fc(offset) * offset_limit
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)          # (1, 2N, 1, 1)
        p_0 = self._get_p_0(h, w, N, dtype)    # (1, 2N, h, w)
        p = p_0 + p_n + offset
        return p

    @staticmethod
    def _get_x_q(x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)  # (b, c, h*w)
        index = q[..., :N] * padded_w + q[..., N:]  # (b, h, w, N)
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )  # (b, c, h*w*N)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat(
            [x_offset[..., s : s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
            dim=-1,
        )
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset

    @staticmethod
    def get_activation(name="silu", inplace=True):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)
        else:
            raise AttributeError(f"Unsupported act type: {name}")
        return module


class Bottleneck_RDCN(Bottleneck):
    """Standard bottleneck with RDCN replacing cv2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv2 = RDCN(c_, c2, k[1], 1)


class C3_RDCN(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_RDCN(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))


class C2f_RDCN(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 注意：ModuleList 需传入 list 而非生成器，确保注册子模块
        self.m = nn.ModuleList([Bottleneck_RDCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)])


# ------------------------------------------------------------
# GVHA: Global Vertical & Horizontal Attention
# ------------------------------------------------------------
class GVHA(nn.Module):
    """Global Vertical and Horizontal Attention.
    对 H/W 两个方向的全局聚合，并结合 channel 注意力。
    """

    def __init__(self, channels) -> None:
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels),
        )
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 沿 W 做全局池化 -> [B,C,H,1]；沿 H 做全局池化 -> [B,C,1,W]
        x_pool_h = x.mean(dim=3, keepdim=True)  # [B,C,H,1]
        x_pool_w = x.mean(dim=2, keepdim=True).transpose(2, 3)  # [B,C,1,W]
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)  # [B,C,H+W,1]

        x_pool_hw = self.conv_hw(x_pool_hw)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [H, W], dim=2)

        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [H, W], dim=2)

        x_pool_h = x_pool_h * x_pool_h_weight
        x_pool_w = x_pool_w * x_pool_w_weight

        x_pool_ch = self.gap(x) * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)

        # 将 W 分支转置回 [B,C,W,1] -> 与原 x 对齐
        return x * x_pool_h.sigmoid() * x_pool_w.transpose(2, 3).sigmoid() * x_pool_ch.sigmoid()


# ------------------------------------------------------------
# GRDCN: Deformable Conv guided by GVHA offsets + export fallback
# ------------------------------------------------------------
class GRDCN_Offset_Attention(nn.Module):
    """Generate offsets & mask for deformable conv, enhanced by GVHA."""

    def __init__(self, in_channels, kernel_size, stride, deformable_groups=1) -> None:
        super().__init__()
        padding = autopad(kernel_size, None, 1)
        self.out_channel = deformable_groups * 3 * kernel_size * kernel_size
        self.conv_offset_mask = nn.Conv2d(in_channels, self.out_channel, kernel_size, stride, padding, bias=True)
        self.attention = GVHA(self.out_channel)

    def forward(self, x):
        conv_offset_mask = self.conv_offset_mask(x)
        conv_offset_mask = self.attention(conv_offset_mask)
        return conv_offset_mask


class GRDCN(nn.Module):
    """Deformable conv + GVHA-based offsets with ONNX/RKNN export fallback."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
        deformable_groups=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        pad = autopad(kernel_size, padding, dilation)
        self.padding = (pad, pad)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.conv_offset_mask = GRDCN_Offset_Attention(in_channels, kernel_size, stride, deformable_groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        # 导出/不支持 deform_conv2d 时：退化为普通 conv（或可改为空洞卷积近似）
        exporting = torch.onnx.is_in_onnx_export() or (getattr(torch.ops, "torchvision", None) is None)
        if exporting:
            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            y = self.bn(y)
            return self.act(y)

        # 训练/推理（支持 torchvision deform_conv2d）
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        y = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            self.deformable_groups,
            True,
        )
        y = self.bn(y)
        return self.act(y)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        # 初始化 offset/mask 卷积为零，便于从普通卷积出发学习
        self.conv_offset_mask.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.conv_offset_mask.bias.data.zero_()


class Bottleneck_GRDCN(Bottleneck):
    """Standard bottleneck with GRDCN replacing cv2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv2 = GRDCN(c_, c2, k[1], 1)


class C3_GRDCN(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_GRDCN(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))


class C2f_GRDCN(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList([Bottleneck_GRDCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)])
