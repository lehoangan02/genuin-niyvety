import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from eval import EvalModule, write_results

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataset import FewShotDetDataset, custom_collate_fn
from embedding_dataset import EmbeddingDetDataset
from model import *
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import DecoderV1

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

data_root = "./../DATA"
batch_size = 12

# load the first image from data_root/frames to check
first_frame_path = os.path.join(data_root, "frames", os.listdir(os.path.join(data_root, "frames"))[0])
from PIL import Image
first_frame = Image.open(first_frame_path).convert("RGB")
print("First frame size:", first_frame.size)
# apply the frame transform
frame_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
first_frame = frame_transform(first_frame).unsqueeze(0).to(device)
print("First frame tensor shape:", first_frame.shape)

# load the first 3 embedding from data_root/embeddings_SPA/spa_wo_cls to check
emb_dir = os.path.join(data_root, "embeddings_SPA", "spa_wo_cls")
emb_files = [f for f in os.listdir(emb_dir) if f.endswith(".npy")]
embedding_paths = [os.path.join(emb_dir, f) for f in emb_files[:3]]

embeddings = [np.load(p) for p in embedding_paths]
# combine to a single tensor of shape (3, C, H, W)
embeddings = np.stack(embeddings, axis=0)
embeddings = torch.from_numpy(embeddings).float()
print("Emmbeddings shapes: ", embeddings.shape)
embeddings = embeddings.to(device)

model = CombinedModelV5().to(device)
decoder = DecoderV1()

with torch.no_grad():
    res = model(embeddings, first_frame)    
    print("Model output shape:", res.shape)

