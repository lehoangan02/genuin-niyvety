import torch
from torchvision.ops import nms

class DecoderV1(torch.nn.Module):
    def __init__(self, iou_threshold=1.0, score_threshold=-99999, topk=1):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.topk = topk

    def forward(self, preds):
        # preds: [B, 3, H, W] = (score + width + height)
        B, C, H, W = preds.shape
        assert C >= 3, "Expected at least 3 channels (1 score + 2 width height)"

        boxes_list = []
        scores_list = []

        for b in range(B):
            pred = preds[b]
            score_map = pred[0]
            box_map = pred[1:3]

            scores = score_map.reshape(-1)
            heights = box_map[0].reshape(-1)
            widths = box_map[1].reshape(-1)

            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=pred.device),
                torch.arange(W, device=pred.device)
            )
            centers = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1).to(pred.dtype)

            keep = scores > self.score_threshold
            if keep.sum() == 0:
                boxes_list.append(pred.new_zeros((0, 4)))
                scores_list.append(pred.new_zeros((0,)))
                continue

            scores = scores[keep]
            heights = heights[keep]
            widths = widths[keep]
            centers = centers[keep]

            heights = torch.clamp_min(heights, 1e-6) * H
            widths = torch.clamp_min(widths, 1e-6) * W

            half_w = widths * 0.5
            half_h = heights * 0.5

            # convert (x_center, y_center, height, width) -> (x1, y1, x2, y2)
            x1 = centers[:, 0] - half_w
            y1 = centers[:, 1] - half_h
            x2 = centers[:, 0] + half_w
            y2 = centers[:, 1] + half_h
            boxes = torch.stack((x1, y1, x2, y2), dim=1)

            # topk
            if self.topk > 0 and scores.numel() > self.topk:
                topk_vals, topk_idx = torch.topk(scores, self.topk)
                boxes, scores = boxes[topk_idx], topk_vals

            # NMS on CPU
            keep_idx = nms(boxes.cpu(), scores.cpu(), self.iou_threshold)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

            boxes_list.append(boxes)
            scores_list.append(scores)

        return boxes_list, scores_list
