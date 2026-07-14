"""Feature Complementary Mapping (FCM) block."""

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

__all__ = ("FCM",)


class _FCMChannelAttention(nn.Module):
    """Channel attention that returns a (B, C, 1, 1) mask."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pool(self.depthwise(x)))


class _FCMSpatialAttention(nn.Module):
    """Spatial attention that returns a (B, 1, H, W) mask."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FCM(nn.Module):
    """
    Feature Complementary Mapping.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.sub_channels = max(1, channels // 4)
        self.main_channels = channels - self.sub_channels

        self.main_conv1 = Conv(self.main_channels, self.main_channels, k=3, s=1)
        self.main_conv2 = Conv(self.main_channels, self.main_channels, k=3, s=1)
        self.main_proj = Conv(self.main_channels, channels, k=1, s=1)

        self.sub_proj = Conv(self.sub_channels, channels, k=1, s=1)

        self.spatial_attention = _FCMSpatialAttention(channels)
        self.channel_attention = _FCMChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise RuntimeError(f"FCM expected input channels={self.channels}, got {x.shape[1]}.")

        main_feat, sub_feat = torch.split(x, [self.main_channels, self.sub_channels], dim=1)

        main_feat = self.main_conv1(main_feat)
        main_feat = self.main_conv2(main_feat)
        main_feat = self.main_proj(main_feat)

        sub_feat = self.sub_proj(sub_feat)

        fused_spatial = self.spatial_attention(sub_feat) * main_feat
        fused_channel = self.channel_attention(main_feat) * sub_feat
        return fused_spatial + fused_channel
