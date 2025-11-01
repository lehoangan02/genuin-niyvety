import torch
import numpy as np
import cv2
import time
import json
import os
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

DATA_FOLDER = "./DATA"

INFERENCE_VIDEO_PATH = os.path.join(DATA_FOLDER, "public_test/samples/CardboardBox_0/drone_video.mp4")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLOE("yoloe-11l-seg.pt").to(device)
names = ["cardboard box"]

# Compute text embeddings on CPU (MPS workaround)
model_cpu = model.to("cpu")
text_pe = model_cpu.get_text_pe(names)
model.set_classes(names, text_pe)
model = model.to(device)

video_path = INFERENCE_VIDEO_PATH
# video_id should be "BlackBox_0" for the given path, which is the name of the folder containing the video
video_id = os.path.basename(os.path.dirname(video_path))

results = model.predict(
    source=video_path,
    predictor=YOLOEVPSegPredictor,
    stream=True,
    device=device,
    save=True,
)

prev_time = time.time()
total_detections = 0
frame_index = 0
all_bboxes = []

for r in results:
    frame = r.plot()
    num_detections = len(r.boxes)
    total_detections += num_detections

    if num_detections > 0:
        bboxes = r.boxes.xyxy.cpu().numpy()
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            all_bboxes.append({
                "frame": frame_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    # FPS display
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {total_detections}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("YOLOE Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cv2.destroyAllWindows()

output_data = [
    {
        "video_id": video_id,
        "detections": [
            {
                "bboxes": all_bboxes
            }
        ]
    }
]

with open("results.json", "w") as f:
    json.dump(output_data, f, indent=4)

print(f"✅ Tracking complete. Total detections: {total_detections}")
print("📁 Results saved to results.json and 'runs/predict' folder.")
