import cv2
import torch
import timm
import time

# Load model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = timm.create_model("fastvit_sa12", pretrained=True, features_only=True)
model.eval().to(device)

# Load frame_120.jpg for testing
image_path = "frame_120.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess image
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image_tensor = image_tensor.to(device)
# print image tensor shape
print("Input image tensor shape:", image_tensor.shape)

# Inference
with torch.no_grad():
    start_time = time.time()
    outputs = model(image_tensor)
    end_time = time.time()

print("Output shape:")
for i, out in enumerate(outputs):
    print(f"Feature {i}: {out.shape}")
print("Inference time (s):", end_time - start_time)
