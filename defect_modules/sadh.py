"""Scale-adaptive decoupled detection head used by the HARP-Net ablations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, DFL
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["Detect_LSCSBD"]


def _best_gn_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Detect_LSCSBD(nn.Module):
    """Lightweight shared-convolution head with per-level normalization."""

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(
        self,
        nc: int = 6,
        hidc: int = 256,
        ch: tuple = (),
        reg_max: int | None = None,
        shared_layers: int = 2,
        dw_per_level=None,
        norm: str = "bn",
    ):
        super().__init__()
        if len(ch) not in {3, 4}:
            raise ValueError(f"Detect_LSCSBD expects 3 or 4 feature levels, got {len(ch)}")
        if norm.lower() not in {"bn", "gn"}:
            raise ValueError("Detect_LSCSBD norm must be 'bn' or 'gn'")

        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = int(reg_max) if reg_max is not None else (12 if self.nl == 4 else 16)
        self.no = self.nc + 4 * self.reg_max
        self.stride = torch.zeros(self.nl)

        if dw_per_level is None:
            self.dw_flags = [False] * (self.nl - 1) + [True]
        elif isinstance(dw_per_level, bool):
            self.dw_flags = [dw_per_level] * self.nl
        else:
            self.dw_flags = [bool(value) for value in dw_per_level]
            if len(self.dw_flags) != self.nl:
                raise ValueError(
                    f"dw_per_level length ({len(self.dw_flags)}) must match detection levels ({self.nl})"
                )

        self.stem = nn.ModuleList(Conv(c, hidc, k=1, s=1, g=1, act=False) for c in ch)
        self.shared_layers = int(shared_layers)
        self.shared_conv_std = nn.ModuleList(
            nn.Conv2d(hidc, hidc, 3, 1, 1, groups=1, bias=False) for _ in range(self.shared_layers)
        )
        self.shared_conv_dw = nn.ModuleList(
            nn.Conv2d(hidc, hidc, 3, 1, 1, groups=hidc, bias=False) for _ in range(self.shared_layers)
        )
        if norm.lower() == "bn":
            self.sep_norm = nn.ModuleList(
                nn.ModuleList(nn.BatchNorm2d(hidc) for _ in range(self.shared_layers)) for _ in ch
            )
        else:
            groups = _best_gn_groups(hidc)
            self.sep_norm = nn.ModuleList(
                nn.ModuleList(nn.GroupNorm(groups, hidc) for _ in range(self.shared_layers)) for _ in ch
            )
        self.act = nn.SiLU(inplace=True)
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1, bias=True)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1, bias=True)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.bias_init()

    def bias_init(self, probability: float = 0.01) -> None:
        with torch.no_grad():
            self.cv2.bias.fill_(0.1)
            self.cv3.bias.fill_(-math.log((1 - probability) / probability))

    @torch.no_grad()
    def decode_bboxes(self, pred_or_anchors, pred_dist=None, stride=None):
        if pred_dist is None:
            box = pred_or_anchors
            anchors = self.anchors.unsqueeze(0)
            strides = self.strides
        else:
            anchors = pred_or_anchors
            if anchors.dim() == 2:
                anchors = anchors.unsqueeze(0)
            box = pred_dist
            strides = stride

        batch = box.shape[0]
        distribution_channels = 4 * self.reg_max
        if box.shape[1] == distribution_channels:
            distances = self.dfl(box) if self.reg_max > 1 else box
        elif box.shape[-1] == distribution_channels:
            box = box.permute(0, 2, 1).contiguous()
            distances = self.dfl(box) if self.reg_max > 1 else box
        elif box.shape[1] == 4:
            distances = box
        elif box.shape[-1] == 4:
            distances = box.permute(0, 2, 1).contiguous()
        else:
            points = box.shape[1]
            probabilities = box.view(batch, points, 4, self.reg_max).softmax(-1)
            projection = torch.arange(self.reg_max, dtype=box.dtype, device=box.device)
            distances = (probabilities * projection).sum(-1).permute(0, 2, 1).contiguous()
        return dist2bbox(distances, anchors, xywh=True, dim=1) * strides

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != self.nl:
            raise ValueError(f"Detect_LSCSBD expects {self.nl} feature maps, got {len(features)}")

        outputs = []
        for level in range(self.nl):
            feature = self.stem[level](features[level])
            for layer in range(self.shared_layers):
                convolution = self.shared_conv_dw[layer] if self.dw_flags[level] else self.shared_conv_std[layer]
                feature = self.act(self.sep_norm[level][layer](convolution(feature)))
            outputs.append(torch.cat((self.cv2(feature), self.cv3(feature)), 1))

        if self.training:
            return outputs

        batch = outputs[0].shape[0]
        concatenated = torch.cat([output.reshape(batch, self.no, -1) for output in outputs], 2)
        if self.dynamic or self.shape != outputs[0].shape:
            anchors, strides = make_anchors(outputs, self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1).to(device=concatenated.device, dtype=concatenated.dtype)
            self.strides = strides.transpose(0, 1).to(device=concatenated.device, dtype=concatenated.dtype)
            self.shape = outputs[0].shape

        box, classification = concatenated.split((4 * self.reg_max, self.nc), 1)
        prediction = torch.cat((self.decode_bboxes(box), classification.sigmoid()), 1)
        return prediction if self.export else (prediction, outputs)
