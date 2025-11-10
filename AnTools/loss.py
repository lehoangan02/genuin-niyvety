import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Standard Focal loss for binary (0/1) targets.
    Expects pred (prob in (0, 1)) and gt (0 or 1) with same shape.

    alpha (gamma in the original paper) is the focusing parameter.
    """

    def __init__(self, alpha=2.0, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, gt):
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_mask

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        num_pos = pos_mask.sum()

        if num_pos == 0:
            # no positive samples, only negative loss
            loss = -neg_loss
        else:
            # normalize by number of positive samples
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class LossAll(nn.Module):
    """
    Loss wrapper that matches your model output layout for CombinedModelV3:

    Channel 0 -> confidence (Focal Loss)
    Channels 1-2 -> width/height (relative w, h)

    Assumes at most ONE object per frame.
    """

    def __init__(
        self, focal_alpha=2.0, lambda_conf=1.0, lambda_center=1.0, lambda_wh=1.0
    ):
        super(LossAll, self).__init__()
        self.focal = FocalLoss(alpha=focal_alpha)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="sum")

        self.lambda_conf = lambda_conf
        self.lambda_center = lambda_center
        self.lambda_wh = lambda_wh

    def forward(self, preds, targets, frame_tensor):
        """
        preds: [B, 3, H, W] (logits for conf, raw regression for box)
        targets: list of dicts (len B)
        frame_tensor: [B, C, H_frame, W_frame]
        Returns: scalar tensor (total loss)
        """

        # --- 1. Build Ground Truth Tensor ---

        B, C_pred, H, W = preds.shape
        _, _, frame_H, frame_W = frame_tensor.shape

        gt_tensor = torch.zeros(B, 3, H, W, device=preds.device)
        reg_mask = torch.zeros(B, 1, H, W, device=preds.device, dtype=torch.bool)

        num_pos = 0.0

        for b in range(B):
            target_boxes = targets[b]["boxes"]  # Shape: [num_boxes, 2]

            if target_boxes.shape[0] > 0:
                box = target_boxes[0]

                gt_width, gt_height = box

                width_rel = gt_width / frame_W
                height_rel = gt_height / frame_H
                # print(width_rel, height_rel)
                # print(targets)
                heatmap = targets[b]["heatmap"]
                # print(torch.max(heatmap))

                gt_tensor[b, 0, :, :] = heatmap
                
                coord = torch.argmax(heatmap)
                cy_feat, cx_feat = divmod(coord.item(), heatmap.shape[1])
                cx_int = int(cx_feat)
                cy_int = int(cy_feat)
                cx_int = max(0, min(cx_int, W - 1))
                cy_int = max(0, min(cy_int, H - 1))

                gt_tensor[b, 1, cy_int, cx_int] = width_rel
                gt_tensor[b, 2, cy_int, cx_int] = height_rel

                reg_mask[b, 0, cy_int, cx_int] = True
                num_pos += 1.0

        # --- 2. Calculate Losses ---

        # Confidence: Focal loss on the confidence map
        conf_pred = preds[:, 0:1, :, :]
        conf_gt = gt_tensor[:, 0:1, :, :]
        loss_conf = self.focal(conf_pred, conf_gt)

        # --- Regression Loss (Center & WH) ---

        # Split the tensors first
        wh_pred = preds[:, 1:3, :, :]  # [B, 2, H, W]
        wh_gt = gt_tensor[:, 1:3, :, :]  # [B, 2, H, W]

        # Expand the [B, 1, H, W] mask to [B, 2, H, W]
        wh_mask = reg_mask.expand_as(wh_pred)

        # Filter predictions and gts just for the positive locations
        wh_pred_pos = wh_pred[wh_mask]
        wh_gt_pos = wh_gt[wh_mask]

        # Normalize by number of positive boxes
        num_pos = max(1.0, num_pos)
        loss_wh = self.smooth_l1(wh_pred_pos, wh_gt_pos) / num_pos

        # --- 3. Combine Losses ---
        total_loss = self.lambda_conf * loss_conf + self.lambda_wh * loss_wh
        # print(loss_conf)
        # print(loss_wh)

        # For debugging, you could return a dict:
        # return {
        #     "total": total_loss,
        #     "conf": loss_conf.detach(),
        #     "center": loss_center.detach(),
        #     "wh": loss_wh.detach()
        # }

        return total_loss
