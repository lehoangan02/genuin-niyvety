import cv2
import json
import os

# ======= CONFIG =======
submission_path = "submission.json"   # your generated result
dataset_dir = "public_test/public_test/samples"
# =======================

# Load submission JSON
with open(submission_path, "r") as f:
    data = json.load(f)

for video_data in data:
    video_id = video_data["video_id"]
    video_path = os.path.join(dataset_dir, video_id, "drone_video.mp4")

    if not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_path}")
        continue

    # Prepare output path
    out_path = os.path.join(dataset_dir, video_id, f"{video_id}_vis.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # ======= Build frame -> list of boxes mapping =======
    frame_boxes = {}

    for det in video_data.get("detections", []):
        # Some detections contain nested "bboxes" list
        if "bboxes" in det:
            for bbox in det["bboxes"]:
                frame_boxes.setdefault(bbox["frame"], []).append(bbox)
        # (Backward compatibility) If submission used flat boxes
        elif all(k in det for k in ["frame", "x1", "y1", "x2", "y2"]):
            frame_boxes.setdefault(det["frame"], []).append(det)

    # ======= Draw and write video =======
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_boxes:
            for box in frame_boxes[frame_id]:
                x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                # Color (green) or gradient if clip_sim exists
                color = (0, 255, 0)
                if "clip_sim" in box:
                    conf = float(box["clip_sim"])
                    color = (0, int(255 * conf), int(255 * (1 - conf)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{video_id} | f{frame_id}",
                            (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"✅ Saved visualization: {out_path}")
