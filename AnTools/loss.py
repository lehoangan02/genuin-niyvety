import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal loss for heatmap-like targets.
    Expects pred (logits) and gt (in [0,1]) with same shape.
    Uses alpha (focusing power for preds) and beta (weighting for negatives).
    """
    def __init__(self, alpha=2.0, beta=4.0, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, gt):
        # pred: logits. Convert to probabilities safely
        pred_prob = torch.clamp(torch.sigmoid(pred), self.eps, 1.0 - self.eps)

        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = torch.log(pred_prob) * torch.pow(1 - pred_prob, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred_prob) * torch.pow(pred_prob, self.alpha) * neg_weights * neg_mask

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        num_pos = pos_mask.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

class LossAll(nn.Module):
    """
    Loss wrapper that matches your model output layout for CombinedModelV3:

    Channel 0 -> confidence (BCEWithLogits)
    Channels 1-2 -> center offsets (Focal loss on heatmaps)
    Channels 3-4 -> width/height (Smooth L1 on regression)

    The forward() expects:
      preds: tensor [B, 5, H, W]
      gts:   tensor [B, 5, H, W]   (same layout as preds)
    and returns a single scalar tensor (total loss) so it can be used directly in:
      loss = criterion(preds, gts)
      loss.backward()
    """

    def __init__(self,
                 focal_alpha=2.0,
                 focal_beta=4.0,
                 lambda_conf=1.0,
                 lambda_center=1.0,
                 lambda_wh=1.0):
        super(LossAll, self).__init__()
        self.focal = FocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_conf = lambda_conf
        self.lambda_center = lambda_center
        self.lambda_wh = lambda_wh

    def forward(self, preds, gts):
        """
        preds: [B, 5, H, W]
        gts:   [B, 5, H, W]
        Returns: scalar tensor (total loss)
        """
        if isinstance(preds, dict):
            # support old-style dict inputs (if some parts of your code still produce dict)
            # expecting key 'out' or 'pred' to contain the tensor — prefer direct tensor input though
            # fallback: try to extract main output
            if 'out' in preds:
                preds = preds['out']
            else:
                # try first tensor-like value
                for v in preds.values():
                    if torch.is_tensor(v):
                        preds = v
                        break

        # ensure tensors
        assert torch.is_tensor(preds), "preds must be a tensor [B,5,H,W]"
        assert torch.is_tensor(gts), "gts must be a tensor [B,5,H,W]"

        # split channels
        conf_pred = preds[:, 0, :, :]        # logits for confidence
        center_pred = preds[:, 1:3, :, :]    # logits (we apply focal)
        wh_pred = preds[:, 3:5, :, :]        # regression for w,h

        conf_gt = gts[:, 0, :, :].float()
        center_gt = gts[:, 1:3, :, :].float()
        wh_gt = gts[:, 3:5, :, :].float()

        # confidence: BCEWithLogits
        loss_conf = self.bce_logits(conf_pred, conf_gt)

        # center: apply focal per-channel and average
        # focal expects pred logits and gt in [0,1]
        # center_gt might be heatmaps with {0,1}. If your center channels are offsets
        # instead of heatmaps, you must adapt — this assumes heatmap-like gt.
        # Apply focal separately per channel and average
        loss_center_ch0 = self.focal(center_pred[:, 0, :, :], center_gt[:, 0, :, :])
        loss_center_ch1 = self.focal(center_pred[:, 1, :, :], center_gt[:, 1, :, :])
        loss_center = 0.5 * (loss_center_ch0 + loss_center_ch1)

        # width/height: Smooth L1 on the regression channels
        # Only compute where you have valid gt (optional). If you supply a mask, incorporate it here.
        loss_wh = self.smooth_l1(wh_pred, wh_gt)

        total_loss = (self.lambda_conf * loss_conf +
                      self.lambda_center * loss_center +
                      self.lambda_wh * loss_wh)

        return total_loss
