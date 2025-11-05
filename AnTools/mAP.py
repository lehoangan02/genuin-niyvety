# ================================================================
# Spatio–Temporal IoU Evaluation Script (for label.txt + test_output.txt)
# ================================================================
# Format expected in both files:
# video_id img1 img2 img3 frame_file has_obj x1 y1 x2 y2
# ================================================================

import numpy as np

# ---------------------------------------------------------------
# 🔗 File paths
GROUND_TRUTH_PATH = "label.txt"       # your GT file
PREDICTIONS_PATH  = "test_output.txt" # your prediction file
IOU_THRESHOLD     = 0.5               # for optional mAP@50
# ================================================================


# ================================================================
# 1. Load data
# ================================================================
def load_txt(path):
    """
    Load text file and return dict:
    {
      video_id: {frame_number: (has_object, [x1,y1,x2,y2])}
    }
    """
    data = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            video_id = parts[0]
            frame_file = parts[4]
            # Extract frame number robustly (e.g., frame00716_1.jpg → 716)
            frame_num = int(frame_file.split("frame")[-1].split("_")[0])
            has_obj = int(parts[5])
            x1, y1, x2, y2 = map(float, parts[6:])
            data.setdefault(video_id, {})[frame_num] = (has_obj, [x1, y1, x2, y2])
    return data


# ================================================================
# 2. IoU and ST-IoU computation
# ================================================================
def iou(boxA, boxB):
    """2D IoU between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-8)


def compute_stiou(gt_frames, pred_frames):
    """
    Compute ST-IoU between ground truth and prediction per video.
    Uses union of frames in GT or prediction.
    """
    all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    intersection_frames = [
        f for f in all_frames
        if gt_frames.get(f, (0,))[0] == 1 and pred_frames.get(f, (0,))[0] == 1
    ]
    union_frames = [
        f for f in all_frames
        if gt_frames.get(f, (0,))[0] == 1 or pred_frames.get(f, (0,))[0] == 1
    ]

    if len(union_frames) == 0:
        return 0.0

    iou_sum = sum(
        iou(gt_frames[f][1], pred_frames[f][1])
        for f in intersection_frames
    )
    return iou_sum / len(union_frames)


# ================================================================
# 3. Evaluation and mAP
# ================================================================
def evaluate(gt_data, pred_data, thr=0.5):
    stiou_list = []
    binary_hits = []

    for vid in gt_data:
        gt_frames = gt_data[vid]
        pred_frames = pred_data.get(vid, {})
        stiou_val = compute_stiou(gt_frames, pred_frames)
        stiou_list.append(stiou_val)
        binary_hits.append(1 if stiou_val >= thr else 0)
        print(f"{vid}: ST-IoU={stiou_val:.4f}, Match@{thr:.2f}={bool(binary_hits[-1])}")

    mean_stiou = np.mean(stiou_list) if stiou_list else 0.0
    mean_ap = np.mean(binary_hits) if binary_hits else 0.0
    return mean_stiou, mean_ap


# ================================================================
# 4. Run evaluation
# ================================================================
if __name__ == "__main__":
    gt_data = load_txt(GROUND_TRUTH_PATH)
    pred_data = load_txt(PREDICTIONS_PATH)
    mean_stiou, mean_ap = evaluate(gt_data, pred_data, IOU_THRESHOLD)

    print("\n================ Evaluation Results ================")
    print(f"Mean ST-IoU (official metric): {mean_stiou:.4f}")
    print(f"mAP@{int(IOU_THRESHOLD*100)} (optional): {mean_ap:.4f}")
    print("===================================================")
