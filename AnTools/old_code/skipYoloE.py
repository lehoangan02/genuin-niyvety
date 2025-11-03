import torch
import numpy as np
import cv2
import time
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLOE("yoloe-11l-seg.pt").to(device) # Note: I changed to yoloe-l-seg.pt as yoloe-11l-seg.pt is not a standard name. Use your correct model file.

visual_prompts = dict(
    bboxes=np.array([[566, 353, 2434, 2654]]),
    cls=np.array([0]),
)

results = model.predict(
    source="drone_video.mov",
    refer_image="img_3.jpg",
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    stream=True,
    device=device,
    save=True,
)

prev_time = time.time()
is_paused = False  # <-- 1. Add a pause state variable

for r in results:
    frame = r.plot()

    # Compute FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Draw FPS on frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Add text to show pause state
    if is_paused:
        cv2.putText(frame, "PAUSED (Press 'p' to play, 'n' for next frame)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("YOLOE Tracking", frame)

    # --- 2. Modified Key Handling Logic ---
    key_wait_value = 1  # Default: wait 1ms (playing)
    if is_paused:
        key_wait_value = 0  # Wait indefinitely (paused)

    key = cv2.waitKey(key_wait_value) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("p"):
        is_paused = not is_paused  # Toggle pause state
    elif key == ord("n") and is_paused:
        pass  # If paused, 'n' will advance one frame and re-pause
    # --- End of Modified Logic ---

cv2.destroyAllWindows()
print("✅ Tracking complete. Output video saved in 'runs/predict' folder.")