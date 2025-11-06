import torch
from torchvision.ops import nms

class DecoderV1(torch.nn.Module):
    def __init__(self, iou_threshold=0.5, score_threshold=-1, topk=1):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.topk = topk

    def forward(self, preds):
        # preds: [B, 5, H, W] = (score + 4 box coords)
        B, C, H, W = preds.shape
        assert C >= 5, "Expected at least 5 channels (1 score + 4 coords)"

        boxes_list = []
        scores_list = []

        for b in range(B):
            pred = preds[b]
            score_map = pred[0]
            box_map = pred[1:5]

            # flatten
            scores = score_map.reshape(-1)
            boxes = box_map.reshape(4, -1).permute(1, 0)

            # threshold
            keep = scores > self.score_threshold
            boxes, scores = boxes[keep], scores[keep]

            if scores.numel() == 0:
                boxes_list.append(torch.zeros((0, 4)))
                scores_list.append(torch.zeros((0,)))
                continue

            # topk
            if scores.numel() > self.topk:
                topk_vals, topk_idx = torch.topk(scores, self.topk)
                boxes, scores = boxes[topk_idx], topk_vals

            # NMS on CPU
            keep_idx = nms(boxes.cpu(), scores.cpu(), self.iou_threshold)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

            boxes_list.append(boxes)
            scores_list.append(scores)

        return boxes_list, scores_list
