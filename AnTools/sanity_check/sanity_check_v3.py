import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataset import FewShotDetDataset, custom_collate_fn
from model import CombinedModelV3
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import DecoderV1

# --- 1. Define Device ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- 2. Define Transforms ---

# Get the *exact* preprocessing for CLIP
clip_model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_image_processor = processor.image_processor

# Transform for the 3 QUERY images (must match CLIP)
query_transform = T.Compose([
    T.Resize((clip_image_processor.crop_size['height'], clip_image_processor.crop_size['width'])),
    T.ToTensor(),
    T.Normalize(mean=clip_image_processor.image_mean, std=clip_image_processor.image_std)
])

# Transform for the FRAME image (No resize, just normalize)
# Using standard ImageNet mean/std for the timm model
frame_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. Create Dataset and DataLoader ---
data_root = './../DATA' # Point this to your DATA folder
batch_size = 8

train_dataset = FewShotDetDataset(
    data_root_dir=data_root,
    query_transform=query_transform,
    frame_transform=frame_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn
)

# --- 4. Initialize Model ---
model = CombinedModelV3().to(device)
model.eval() # Set to eval mode since we froze the encoders

# --- 5. Run One Batch ---
# Get a single batch from the loader to test
query_batch, frame_batch, target_list = next(iter(train_loader))
# Print input shapes
print(f"Query Batch Shape: {query_batch.shape}")  # Expected: [B, 3, 3, H, W]
print(f"Frame Batch Shape: {frame_batch.shape}")  # Expected: [B, 3, H, W]

# Move data to device
query_batch = query_batch.to(device)
frame_batch = frame_batch.to(device)
# (no need to move targets for this forward pass)

# Run the model
with torch.no_grad():
    frame_features = model(query_batch, frame_batch)
    print("--- Output Shapes ---")
    print("\nFrame Features (List of Tensors):")
    print(f"Frame Features Shape: {frame_features.shape}")
    # feed the outputs to the decoder
    decoder = DecoderV1()
    decoder_output = decoder(frame_features)
    boxes_list, scores_list = decoder_output
    print(f"Decoder output: {len(boxes_list)} batch(es)")
    print(f"First batch boxes shape: {boxes_list[0].shape}")
    print(f"First batch scores shape: {scores_list[0].shape}")
