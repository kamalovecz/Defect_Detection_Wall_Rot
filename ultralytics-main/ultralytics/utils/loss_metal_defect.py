#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
金属表面缺陷检测专用损失函数
Metal Surface Defect Detection Custom Loss Functions

创新点:
1. 尺寸感知损失 (Size-Aware Loss): 微小缺陷加权更高
2. 边缘增强损失 (Edge-Enhanced Loss): 针对裂纹、划痕等线性缺陷
3. 类别平衡损失 (Class-Balanced Loss): 处理不同缺陷类型的样本不平衡
4. 混合IoU损失 (Hybrid IoU Loss): 结合Wise-IoU和NWD
5. 难样本自适应 (Hard Sample Adaptive): 动态关注难检测缺陷

适用场景:
- 金属表面缺陷检测 (裂纹、划痕、斑点、凹坑、麻点、氧化等)
- 纺织品瑕疵检测
- 玻璃表面缺陷检测
- 其他工业质检场景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MetalDefectLoss(nn.Module):
    """
    金属表面缺陷检测专用损失函数
    
    创新组合:
    1. Wise-IoU: 动态梯度增益
    2. NWD: 小目标增强
    3. Size-Aware Weighting: 尺寸感知权重
    4. Edge-Enhanced Loss: 边缘损失（可选）
    5. Class Balance: 类别平衡
    """
    
    def __init__(
        self,
        reg_max=16,
        use_nwd=True,
        nwd_ratio=0.25,
        use_edge_loss=True,
        edge_weight=0.5,
        use_size_aware=True,
        size_bins=(0.0, 0.02, 0.05, 0.1, 1.0),
        size_weights=(3.0, 2.0, 1.5, 1.0),
        use_class_balance=True,
        class_weights=None,
    ):
        """
        Args:
            reg_max: DFL最大值
            use_nwd: 是否使用NWD损失
            nwd_ratio: NWD损失占比
            use_edge_loss: 是否使用边缘损失
            edge_weight: 边缘损失权重
            use_size_aware: 是否使用尺寸感知权重
            size_bins: 尺寸分界点 (相对于图像尺寸)
            size_weights: 对应的权重
            use_class_balance: 是否使用类别平衡
            class_weights: 各类别权重 (None则自动计算)
        """
        super().__init__()
        self.reg_max = reg_max
        self.use_nwd = use_nwd
        self.nwd_ratio = nwd_ratio
        self.use_edge_loss = use_edge_loss
        self.edge_weight = edge_weight
        self.use_size_aware = use_size_aware
        self.size_bins = torch.tensor(size_bins, dtype=torch.float32)
        self.size_weights = torch.tensor(size_weights, dtype=torch.float32)
        self.use_class_balance = use_class_balance
        self.class_weights = class_weights
        
        # Wise-IoU组件
        self.iou_mean = 1.0
        self.momentum = 0.9
        
        print(f"[MetalDefectLoss] 初始化")
        print(f"  - NWD增强: {use_nwd} (ratio={nwd_ratio})")
        print(f"  - 边缘损失: {use_edge_loss} (weight={edge_weight})")
        print(f"  - 尺寸感知: {use_size_aware}")
        print(f"  - 类别平衡: {use_class_balance}")
    
    def bbox_iou(
        self,
        box1: torch.Tensor,
        box2: torch.Tensor,
        xywh: bool = True,
        GIoU: bool = False,
        DIoU: bool = False,
        CIoU: bool = False,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        计算IoU及其变体
        Args:
            box1: [N, 4] (x1, y1, x2, y2) or (x, y, w, h)
            box2: [N, 4]
            xywh: 是否为xywh格式
        Returns:
            iou: [N]
        """
        if xywh:
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        
        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
        
        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps
        
        # IoU
        iou = inter / union
        
        if CIoU or DIoU or GIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                if CIoU:
                    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)
                return iou - rho2 / c2
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        
        return iou
    
    def nwd_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Normalized Wasserstein Distance (NWD)
        对小目标更敏感的距离度量
        
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4]
        Returns:
            nwd: [N]
        """
        # 转换为中心点和宽高
        pred_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
        pred_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
        pred_w = pred_boxes[..., 2] - pred_boxes[..., 0]
        pred_h = pred_boxes[..., 3] - pred_boxes[..., 1]
        
        target_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
        target_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2
        target_w = target_boxes[..., 2] - target_boxes[..., 0]
        target_h = target_boxes[..., 3] - target_boxes[..., 1]
        
        # 中心点距离
        center_distance = torch.sqrt(
            (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2 + eps
        )
        
        # 宽高距离
        wh_distance = torch.sqrt(
            (pred_w - target_w) ** 2 + (pred_h - target_h) ** 2 + eps
        )
        
        # 归一化常数（使用目标框尺寸）
        normalize_term = torch.sqrt(target_w ** 2 + target_h ** 2 + eps)
        
        # Wasserstein距离
        wasserstein_2 = (center_distance + wh_distance) / (normalize_term + eps)
        
        # 转换为相似度（0-1，越大越好）
        nwd = torch.exp(-wasserstein_2)
        
        return nwd
    
    def wise_iou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Wise-IoU v3: 动态非单调焦点损失
        
        核心创新: 根据anchor质量动态调整梯度增益
        - 高质量anchor: 减小梯度，避免过拟合
        - 低质量anchor: 增大梯度，加速收敛
        
        Args:
            pred_boxes: [N, 4]
            target_boxes: [N, 4]
            weight: [N, 1] 样本权重
        Returns:
            loss: scalar
        """
        # 计算IoU
        iou = self.bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=False)
        
        # 更新IoU均值（EMA）
        with torch.no_grad():
            self.iou_mean = self.momentum * self.iou_mean + (1 - self.momentum) * iou.mean().item()
        
        # 计算anchor质量指标
        beta = iou.detach() / (self.iou_mean + eps)
        
        # 动态梯度增益
        alpha = 1 - torch.exp(-beta ** 2)
        
        # Wise-IoU损失
        loss_iou = (1 - iou) * alpha
        
        # 加权求和
        return (loss_iou * weight.squeeze(-1)).sum()
    
    def edge_enhanced_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        边缘增强损失
        专门针对裂纹、划痕等线性缺陷
        
        思路: 
        1. 计算边界距离（4个边）
        2. 对小宽高比的框（线性缺陷）加大边界惩罚
        3. 促进模型更精确地定位线性缺陷的边界
        
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4]
            weight: [N, 1]
        Returns:
            loss: scalar
        """
        # 计算4个边的距离
        edge_diff = torch.abs(pred_boxes - target_boxes)  # [N, 4]
        
        # 计算宽高比（识别线性缺陷）
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        aspect_ratio = torch.min(pred_w, pred_h) / (torch.max(pred_w, pred_h) + eps)
        
        # 线性缺陷权重（宽高比越小，权重越大）
        linear_weight = torch.exp(-aspect_ratio * 2.0)  # [N]
        
        # 边缘损失（L1距离）
        edge_loss = edge_diff.sum(dim=-1)  # [N]
        
        # 加权
        weighted_loss = edge_loss * linear_weight * weight.squeeze(-1)
        
        return weighted_loss.sum()
    
    def size_aware_weight(
        self,
        boxes: torch.Tensor,
        img_size: int = 640
    ) -> torch.Tensor:
        """
        尺寸感知权重
        微小缺陷获得更高权重
        
        Args:
            boxes: [N, 4] (x1, y1, x2, y2) 归一化坐标 [0, 1]
            img_size: 图像尺寸
        Returns:
            weights: [N, 1]
        """
        # 计算面积（相对于图像尺寸）
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area = w * h  # [N]
        
        # 根据面积分配权重
        device = boxes.device
        size_bins = self.size_bins.to(device)
        size_weights = self.size_weights.to(device)
        
        weights = torch.ones_like(area)
        for i in range(len(size_weights)):
            mask = (area >= size_bins[i]) & (area < size_bins[i + 1])
            weights[mask] = size_weights[i]
        
        return weights.unsqueeze(-1)  # [N, 1]
    
    def class_balanced_weight(
        self,
        target_labels: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        类别平衡权重
        处理不同缺陷类型的样本不平衡
        
        Args:
            target_labels: [N] 类别标签
            num_classes: 类别总数
        Returns:
            weights: [N, 1]
        """
        if self.class_weights is not None:
            # 使用预设权重
            weights = torch.tensor(self.class_weights, device=target_labels.device)
            return weights[target_labels.long()].unsqueeze(-1)
        
        # 自动计算权重（逆频率）
        class_counts = torch.bincount(target_labels.long(), minlength=num_classes).float()
        class_counts = torch.clamp(class_counts, min=1.0)  # 避免除零
        
        # 逆频率权重
        total = class_counts.sum()
        class_weights = total / (class_counts * num_classes)
        
        # 归一化
        class_weights = class_weights / class_weights.mean()
        
        return class_weights[target_labels.long()].unsqueeze(-1)
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        num_classes: int = 6,
        img_size: int = 640
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            pred_boxes: [B, N, 4] 预测框 (x1, y1, x2, y2)
            target_boxes: [B, N, 4] 目标框
            target_scores: [B, N, num_classes] 目标分数（one-hot）
            fg_mask: [B, N] 前景mask
            num_classes: 类别数
            img_size: 图像尺寸
        Returns:
            loss_iou: IoU损失
            loss_dfl: DFL损失（如果使用）
        """
        # 筛选前景样本
        pred_boxes_fg = pred_boxes[fg_mask]  # [M, 4]
        target_boxes_fg = target_boxes[fg_mask]  # [M, 4]
        target_scores_fg = target_scores[fg_mask]  # [M, num_classes]
        
        if pred_boxes_fg.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device), torch.tensor(0.0, device=pred_boxes.device)
        
        # 基础权重（来自标签分配）
        weight = target_scores_fg.sum(-1, keepdim=True)  # [M, 1]
        
        # ===== 1. Wise-IoU损失 =====
        loss_iou = self.wise_iou_loss(pred_boxes_fg, target_boxes_fg, weight)
        
        # ===== 2. NWD损失（小目标增强） =====
        if self.use_nwd:
            nwd = self.nwd_loss(pred_boxes_fg, target_boxes_fg)
            loss_nwd = ((1 - nwd) * weight.squeeze(-1)).sum()
            # 混合IoU和NWD
            loss_iou = (1 - self.nwd_ratio) * loss_iou + self.nwd_ratio * loss_nwd
        
        # ===== 3. 边缘增强损失 =====
        if self.use_edge_loss:
            loss_edge = self.edge_enhanced_loss(pred_boxes_fg, target_boxes_fg, weight)
            loss_iou = loss_iou + self.edge_weight * loss_edge
        
        # ===== 4. 尺寸感知权重 =====
        if self.use_size_aware:
            size_weight = self.size_aware_weight(target_boxes_fg, img_size)
            loss_iou = loss_iou * size_weight.mean()
        
        # ===== 5. 类别平衡权重 =====
        if self.use_class_balance:
            # 获取目标类别
            target_labels = target_scores_fg.argmax(dim=-1)  # [M]
            class_weight = self.class_balanced_weight(target_labels, num_classes)
            loss_iou = loss_iou * class_weight.mean()
        
        # 归一化
        target_scores_sum = max(target_scores_fg.sum(), 1.0)
        loss_iou = loss_iou / target_scores_sum
        
        # DFL损失（保持兼容性）
        loss_dfl = torch.tensor(0.0, device=pred_boxes.device)
        
        return loss_iou, loss_dfl


class MetalDefectLossLite(nn.Module):
    """
    金属缺陷轻量损失（去除边缘损失，保留核心功能）
    适合快速训练和部署
    """
    
    def __init__(
        self,
        reg_max=16,
        use_nwd=False,
        nwd_ratio=0.2,
        use_size_aware=True,
        size_bins=(0.0, 0.02, 0.05, 0.1, 1.0),
        size_weights=(2.5, 1.8, 1.3, 1.0),
        use_class_balance=True,
    ):
        super().__init__()
        self.full_loss = MetalDefectLoss(
            reg_max=reg_max,
            use_nwd=use_nwd,
            nwd_ratio=nwd_ratio,
            use_edge_loss=False,  # 关闭边缘损失
            use_size_aware=use_size_aware,
            size_bins=size_bins,
            size_weights=size_weights,
            use_class_balance=use_class_balance,
        )
    
    def forward(self, *args, **kwargs):
        return self.full_loss(*args, **kwargs)


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    # 创建损失函数
    loss_fn = MetalDefectLoss(
        use_nwd=True,
        nwd_ratio=0.25,
        use_edge_loss=True,
        edge_weight=0.5,
        use_size_aware=True,
        use_class_balance=True,
    )
    
    # 模拟数据
    batch_size = 2
    num_anchors = 100
    num_classes = 6
    
    pred_boxes = torch.rand(batch_size, num_anchors, 4)  # [B, N, 4]
    target_boxes = torch.rand(batch_size, num_anchors, 4)
    target_scores = torch.rand(batch_size, num_anchors, num_classes)
    fg_mask = torch.rand(batch_size, num_anchors) > 0.7  # 30%前景
    
    # 计算损失
    loss_iou, loss_dfl = loss_fn(
        pred_boxes, target_boxes, target_scores, fg_mask,
        num_classes=num_classes
    )
    
    print(f"IoU Loss: {loss_iou.item():.4f}")
    print(f"DFL Loss: {loss_dfl.item():.4f}")


if __name__ == '__main__':
    example_usage()

