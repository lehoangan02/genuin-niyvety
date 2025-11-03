import cv2

video_path = "drone_video.mp4"  # replace with your actual path
frame_number = 3505

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Could not read frame {frame_number} from {video_path}")

cv2.imwrite("frame.jpg", frame)
print("Saved frame.jpg")
