import torch
import numpy as np
import cv2
import time
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLOE("yoloe-11l-seg.pt").to(device)

visual_prompts = dict(
    bboxes=np.array([[370, 67, 457, 116]]),
    cls=np.array([0]),
)

results = model.predict(
    source="drone_video.mp4",
    refer_image="frame.jpg",
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    stream=True,
    device=device,
    save=True,
)

prev_time = time.time()

for r in results:
    frame = r.plot()

    # Compute FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Draw FPS on frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("YOLOE Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print("✅ Tracking complete. Output video saved in 'runs/predict' folder.")
