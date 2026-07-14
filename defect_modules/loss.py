"""Externalized HARP-Net RuleLoss detection loss bridge.

This module contains the current custom v8DetectionLoss implementation extracted
from ultralytics-main/ultralytics/utils/loss.py. In this project, "RuleLoss" is
not a standalone original class; it is the rule-weighted branch inside
v8DetectionLoss. The aliases below expose that implementation through the
modular defect_modules package.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ultralytics.utils.atss import ATSSAssigner, generate_anchors
from ultralytics.utils.loss import (
    BboxLoss,
    EMASlideLoss,
    FocalLoss_YOLO,
    QualityfocalLoss_YOLO,
    SlideLoss,
    VarifocalLoss_YOLO,
    build_yolo_cls_loss,
)
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce, self.cls_loss_name = build_yolo_cls_loss()
        # self.bce = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none'))  # Exponential Moving Average Slide Loss
        # self.bce = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) # Slide Loss
        # self.bce = FocalLoss_YOLO(alpha=0.25, gamma=1.5) # FocalLoss
        # self.bce = VarifocalLoss_YOLO(alpha=0.75, gamma=2.0) # VarifocalLoss
        # self.bce = QualityfocalLoss_YOLO(beta=2.0) # QualityfocalLoss
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        if hasattr(m, 'dfl_aux'):
            self.assigner_aux = TaskAlignedAssigner(topk=13, num_classes=self.nc, alpha=0.5, beta=6.0)
            self.aux_loss_ratio = 0.25
        # self.assigner = ATSSAssigner(9, num_classes=self.nc)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # ATSS use        
        self.grid_cell_offset = 0.5
        self.fpn_strides = list(self.stride.detach().cpu().numpy())
        self.grid_cell_size = 5.0

        # Rule-weighted loss:
        # - RULE_LOSS_VERSION=v2   -> keep legacy update-step schedule (RuleLossV2)
        # - RULE_LOSS_VERSION=paper -> epoch schedule lambda(e) aligned with manuscript
        self.rule_loss_enable = self._env_flag("RULE_LOSS_ENABLE", False)
        self.rule_loss_version = str(os.getenv("RULE_LOSS_VERSION", "v2")).strip().lower()
        if self.rule_loss_version in {"legacy", "rulelossv2"}:
            self.rule_loss_version = "v2"
        elif self.rule_loss_version in {"v3", "epoch", "paper_strict"}:
            self.rule_loss_version = "paper"
        elif self.rule_loss_version not in {"v2", "paper"}:
            self.rule_loss_version = "v2"
        self.rule_small_area = self._env_float("RULE_LOSS_SMALL_AREA", 32.0 * 32.0)
        self.rule_gamma_small = self._env_float(
            "RULE_LOSS_GAMMA_SMALL",
            self._env_float("RULE_LOSS_SMALL_GAIN", 0.06),
        )
        self.rule_gamma_contrast = self._env_float(
            "RULE_LOSS_GAMMA_CONTRAST",
            self._env_float("RULE_LOSS_LOW_CONTRAST_GAIN", 0.04),
        )
        self.rule_low_contrast_std = self._env_float("RULE_LOSS_LOW_CONTRAST_STD", 0.12)
        self.rule_lambda_max = self._env_float("RULE_LOSS_LAMBDA_MAX", 1.0)
        self.rule_schedule_iters = int(
            self._env_float("RULE_LOSS_SCHEDULE_ITERS", self._env_float("RULE_LOSS_RAMP_ITERS", 12000))
        )
        self.rule_stage0_ratio = max(0.0, min(1.0, self._env_float("RULE_LOSS_STAGE0_RATIO", 0.20)))
        self.rule_stage1_ratio = max(
            self.rule_stage0_ratio + 1e-6, min(1.0, self._env_float("RULE_LOSS_STAGE1_RATIO", 0.60))
        )
        self.rule_updates = 0
        self.rule_total_epochs = max(1, int(self._env_float("RULE_LOSS_TOTAL_EPOCHS", 300)))
        self.rule_current_epoch = max(0, int(self._env_float("RULE_LOSS_EPOCH", 0)))
        self.rule_t1_epoch_cfg = int(self._env_float("RULE_LOSS_T1_EPOCH", -1))
        self.rule_t2_epoch_cfg = int(self._env_float("RULE_LOSS_T2_EPOCH", -1))
        self.rule_t1_epoch = 0
        self.rule_t2_epoch = 1
        self._resolve_rule_epoch_schedule()

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _env_float_list(name: str, default):
        value = os.getenv(name)
        if not value:
            return list(default)
        try:
            parsed = [float(v.strip()) for v in str(value).split(",") if v.strip()]
            return parsed if parsed else list(default)
        except Exception:
            return list(default)

    def _resolve_rule_epoch_schedule(self):
        total_epochs = max(1, int(self.rule_total_epochs))
        t1 = self.rule_t1_epoch_cfg if self.rule_t1_epoch_cfg >= 0 else int(total_epochs * 0.33)
        t2 = self.rule_t2_epoch_cfg if self.rule_t2_epoch_cfg >= 0 else int(total_epochs * 0.67)
        t1 = max(0, t1)
        t2 = max(t1 + 1, t2)
        self.rule_t1_epoch = t1
        self.rule_t2_epoch = t2

    def set_rule_epoch(self, epoch: int, total_epochs=None):
        if total_epochs is not None:
            self.rule_total_epochs = max(1, int(total_epochs))
        self.rule_current_epoch = max(0, int(epoch))
        self._resolve_rule_epoch_schedule()

    def _sync_rule_epoch_from_env(self):
        epoch_env = os.getenv("RULE_LOSS_EPOCH")
        total_env = os.getenv("RULE_LOSS_TOTAL_EPOCHS")
        t1_env = os.getenv("RULE_LOSS_T1_EPOCH")
        t2_env = os.getenv("RULE_LOSS_T2_EPOCH")
        updated = False
        try:
            if total_env is not None:
                self.rule_total_epochs = max(1, int(float(total_env)))
                updated = True
            if epoch_env is not None:
                self.rule_current_epoch = max(0, int(float(epoch_env)))
            if t1_env is not None:
                self.rule_t1_epoch_cfg = int(float(t1_env))
                updated = True
            if t2_env is not None:
                self.rule_t2_epoch_cfg = int(float(t2_env))
                updated = True
        except Exception:
            return
        if updated:
            self._resolve_rule_epoch_schedule()

    def _lambda_rule_v2(self, increment=False):
        if not self.rule_loss_enable:
            return 0.0
        if increment:
            self.rule_updates += 1

        if self.rule_schedule_iters > 0:
            progress = min(1.0, self.rule_updates / float(self.rule_schedule_iters))
        else:
            progress = 1.0

        if progress <= self.rule_stage0_ratio:
            return 0.0
        if progress >= self.rule_stage1_ratio:
            return self.rule_lambda_max
        ramp = (progress - self.rule_stage0_ratio) / max(self.rule_stage1_ratio - self.rule_stage0_ratio, 1e-8)
        return self.rule_lambda_max * ramp

    def _lambda_rule_epoch(self):
        if not self.rule_loss_enable:
            return 0.0
        self._sync_rule_epoch_from_env()
        epoch = max(0, int(self.rule_current_epoch))
        if epoch < self.rule_t1_epoch:
            return 0.0
        if epoch < self.rule_t2_epoch:
            ramp = (epoch - self.rule_t1_epoch) / max(self.rule_t2_epoch - self.rule_t1_epoch, 1)
            return self.rule_lambda_max * ramp
        return self.rule_lambda_max

    def _lambda_rule_t(self, increment=False):
        if self.rule_loss_version == "paper":
            return self._lambda_rule_epoch()
        return self._lambda_rule_v2(increment=increment)

    def _build_rule_anchor_weights(self, batch, target_bboxes, fg_mask, dtype, lambda_rule):
        """Build per-anchor rule weights from spatial priors: area prior + low-contrast prior."""
        if not self.rule_loss_enable or lambda_rule <= 0.0:
            return None

        anchor_weights = torch.ones_like(fg_mask, dtype=dtype, device=self.device)
        if not fg_mask.any():
            return anchor_weights

        wh = (target_bboxes[..., 2:4] - target_bboxes[..., 0:2]).clamp_min(0)
        area = wh[..., 0] * wh[..., 1]
        prior_small = (area < self.rule_small_area).to(dtype=dtype) * self.rule_gamma_small

        prior_low_contrast = torch.zeros_like(prior_small, dtype=dtype, device=self.device)
        if "img" in batch:
            img_std = batch["img"].float().flatten(1).std(dim=1)
            low_contrast_img = (img_std < self.rule_low_contrast_std).to(dtype=dtype)
            prior_low_contrast = low_contrast_img.unsqueeze(1).expand_as(prior_small) * self.rule_gamma_contrast

        rule_prior = prior_small + prior_low_contrast
        rule_gain = 1.0 + lambda_rule * rule_prior
        anchor_weights = torch.where(fg_mask, rule_gain, anchor_weights)
        return anchor_weights

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        if hasattr(self, 'assigner_aux'):
            loss, batch_size = self.compute_loss_aux(preds, batch)
        else:
            loss, batch_size = self.compute_loss(preds, batch)
        return loss.sum() * batch_size, loss.detach()

    def compute_loss(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        feats = feats[:self.stride.size(0)]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        lambda_rule = self._lambda_rule_t(increment=True) if self.rule_loss_enable else 0.0
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # ATSS
        if isinstance(self.assigner, ATSSAssigner):
            anchors, _, n_anchors_list, _ = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
            target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes.detach() * stride_tensor)
        # TAL
        else:
            target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = target_scores.sum().clamp_min(1.0)
        rule_anchor_weights = self._build_rule_anchor_weights(
            batch=batch, target_bboxes=target_bboxes, fg_mask=fg_mask, dtype=dtype, lambda_rule=lambda_rule
        )
        target_scores_rule = target_scores if rule_anchor_weights is None else target_scores * rule_anchor_weights.unsqueeze(-1)

        # cls loss
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            cls_raw = self.bce(pred_scores, target_scores.to(dtype))
            if rule_anchor_weights is not None:
                cls_raw = cls_raw * rule_anchor_weights.unsqueeze(-1)
            loss[1] = cls_raw.sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            cls_raw = self.bce(pred_scores, cls_iou_targets.to(dtype))
            if rule_anchor_weights is not None:
                cls_raw = cls_raw * rule_anchor_weights.unsqueeze(-1)
            loss[1] = cls_raw.sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, QualityfocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            cls_raw = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool))
            if rule_anchor_weights is not None:
                cls_raw = cls_raw * rule_anchor_weights.unsqueeze(-1)
            loss[1] = cls_raw.sum() / max(fg_mask.sum(), 1)

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores_rule,
                                              target_scores_sum, fg_mask, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0))

        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            if fg_mask.sum():
                auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            else:
                auto_iou = -1
            cls_raw = self.bce(pred_scores, target_scores.to(dtype), auto_iou)
            if rule_anchor_weights is not None:
                cls_raw = cls_raw * rule_anchor_weights.unsqueeze(-1)
            loss[1] = cls_raw.sum() / target_scores_sum  # BCE
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return loss, batch_size
    
    def compute_loss_aux(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats_all = preds[1] if isinstance(preds, tuple) else preds
        if len(feats_all) == self.stride.size(0):
            return self.compute_loss(preds, batch)
        feats, feats_aux = feats_all[:self.stride.size(0)], feats_all[self.stride.size(0):]
        
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_distri_aux, pred_scores_aux = torch.cat([xi.view(feats_aux[0].shape[0], self.no, -1) for xi in feats_aux], 2).split((self.reg_max * 4, self.nc), 1)

        pred_scores, pred_distri = pred_scores.permute(0, 2, 1).contiguous(), pred_distri.permute(0, 2, 1).contiguous()
        pred_scores_aux, pred_distri_aux = pred_scores_aux.permute(0, 2, 1).contiguous(), pred_distri_aux.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        lambda_rule = self._lambda_rule_t(increment=True) if self.rule_loss_enable else 0.0
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes_aux = self.bbox_decode(anchor_points, pred_distri_aux)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux, _ = self.assigner_aux(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = target_scores.sum().clamp_min(1.0)
        target_scores_sum_aux = target_scores_aux.sum().clamp_min(1.0)
        rule_anchor_weights = self._build_rule_anchor_weights(
            batch=batch, target_bboxes=target_bboxes, fg_mask=fg_mask, dtype=dtype, lambda_rule=lambda_rule
        )
        rule_anchor_weights_aux = self._build_rule_anchor_weights(
            batch=batch, target_bboxes=target_bboxes_aux, fg_mask=fg_mask_aux, dtype=dtype, lambda_rule=lambda_rule
        )
        target_scores_rule = target_scores if rule_anchor_weights is None else target_scores * rule_anchor_weights.unsqueeze(-1)
        target_scores_aux_rule = (
            target_scores_aux if rule_anchor_weights_aux is None else target_scores_aux * rule_anchor_weights_aux.unsqueeze(-1)
        )

        # cls loss
        if isinstance(self.bce, nn.BCEWithLogitsLoss):
            cls_raw = self.bce(pred_scores, target_scores.to(dtype))
            cls_raw_aux = self.bce(pred_scores_aux, target_scores_aux.to(dtype))
            if rule_anchor_weights is not None:
                cls_raw = cls_raw * rule_anchor_weights.unsqueeze(-1)
            if rule_anchor_weights_aux is not None:
                cls_raw_aux = cls_raw_aux * rule_anchor_weights_aux.unsqueeze(-1)
            loss[1] = cls_raw.sum() / target_scores_sum  # BCE
            loss[1] += cls_raw_aux.sum() / target_scores_sum_aux * self.aux_loss_ratio  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            target_bboxes_aux /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores_rule,
                                            target_scores_sum, fg_mask, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0))
            aux_loss_0, aux_loss_2 = self.bbox_loss(pred_distri_aux, pred_bboxes_aux, anchor_points, target_bboxes_aux, target_scores_aux_rule,
                                            target_scores_sum_aux, fg_mask_aux, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0))
            
            loss[0] += aux_loss_0 * self.aux_loss_ratio
            loss[2] += aux_loss_2 * self.aux_loss_ratio

        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            cls_raw = self.bce(pred_scores, target_scores.to(dtype), auto_iou)
            cls_raw_aux = self.bce(pred_scores_aux, target_scores_aux.to(dtype), -1)
            if rule_anchor_weights is not None:
                cls_raw = cls_raw * rule_anchor_weights.unsqueeze(-1)
            if rule_anchor_weights_aux is not None:
                cls_raw_aux = cls_raw_aux * rule_anchor_weights_aux.unsqueeze(-1)
            loss[1] = cls_raw.sum() / target_scores_sum  # BCE
            loss[1] += cls_raw_aux.sum() / target_scores_sum_aux * self.aux_loss_ratio  # BCE
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
        return loss, batch_size


RuleLossDetectionLoss = v8DetectionLoss
RuleLoss = v8DetectionLoss


def patch_ultralytics_loss(verbose: bool = True):
    """Route Ultralytics detection training to the externalized RuleLoss class."""
    patched = []

    import ultralytics.utils.loss as loss_mod

    loss_mod.RuleLoss = RuleLoss
    loss_mod.RuleLossDetectionLoss = RuleLossDetectionLoss
    loss_mod.v8DetectionLoss = RuleLossDetectionLoss
    patched.append("ultralytics.utils.loss")

    try:
        import ultralytics.nn.tasks as tasks_mod

        tasks_mod.RuleLoss = RuleLoss
        tasks_mod.RuleLossDetectionLoss = RuleLossDetectionLoss
        tasks_mod.v8DetectionLoss = RuleLossDetectionLoss
        patched.append("ultralytics.nn.tasks")
    except Exception as exc:
        if verbose:
            print(f"[defect_modules.loss] tasks patch warning: {exc!r}")

    if verbose:
        print(f"[defect_modules.loss] RuleLossDetectionLoss patched into {', '.join(patched)}")
    return patched
