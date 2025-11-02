import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLossV2(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0, wh_weight=1.0, center_weight=1.0, conf_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.wh_weight = wh_weight
        self.center_weight = center_weight
        self.conf_weight = conf_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def focal_loss(self, pred, gt):
        pred_sigmoid = torch.sigmoid(pred)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = -torch.log(pred_sigmoid + 1e-6) * torch.pow(1 - pred_sigmoid, self.alpha) * pos_mask
        neg_loss = -torch.log(1 - pred_sigmoid + 1e-6) * torch.pow(pred_sigmoid, self.alpha) * neg_weights * neg_mask

        num_pos = pos_mask.sum()
        loss = (pos_loss.sum() + neg_loss.sum()) / torch.clamp(num_pos, min=1.0)
        return loss

    def forward(self, pred, target):
        # pred, target: [B, 5, H, W]
        conf_pred = pred[:, 0, :, :]
        center_pred = pred[:, 1:3, :, :]
        wh_pred = pred[:, 3:5, :, :]

        conf_gt = target[:, 0, :, :]
        center_gt = target[:, 1:3, :, :]
        wh_gt = target[:, 3:5, :, :]

        # 1️⃣ BCE Loss for confidence
        loss_conf = self.bce(conf_pred, conf_gt)

        # 2️⃣ Focal Loss for center
        loss_center = self.focal_loss(center_pred, center_gt)

        # 3️⃣ Smooth L1 Loss for width-height
        loss_wh = self.smooth_l1(wh_pred, wh_gt)

        total_loss = (
            self.conf_weight * loss_conf +
            self.center_weight * loss_center +
            self.wh_weight * loss_wh
        )

        return {
            "total": total_loss,
            "conf": loss_conf,
            "center": loss_center,
            "wh": loss_wh
        }
