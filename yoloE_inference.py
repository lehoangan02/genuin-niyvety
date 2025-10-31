import torch
import numpy as np
import cv2
import time
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOE segmentation model
model = YOLOE("yoloe-11l-seg.pt").to(device)

# --- Define your text prompt ---
names = ["gray blue backpack"]

# MPS workaround: temporarily move model to CPU to compute text embeddings
model_cpu = model.to("cpu")
text_pe = model_cpu.get_text_pe(names)
model.set_classes(names, text_pe)
model = model.to(device)
# -------------------------------

# Predict on video (no 'text' arg)
results = model.predict(
    source="drone_video.mov",
    predictor=YOLOEVPSegPredictor,
    stream=True,
    device=device,
    save=True,
)

prev_time = time.time()
total_detections = 0

for i, r in enumerate(results):
    frame = r.plot()

    # Count detections in this frame
    num_detections = len(r.boxes)
    total_detections += num_detections

    # FPS display
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS and total detection counter
    cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {total_detections}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("YOLOE Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print(f"✅ Tracking complete. Total detections: {total_detections}. Output video saved in 'runs/predict' folder.")
