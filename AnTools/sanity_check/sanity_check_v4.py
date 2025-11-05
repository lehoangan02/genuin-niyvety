import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

clip_model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_image_processor = processor.image_processor

query_transform = T.Compose([
    T.Resize((clip_image_processor.crop_size["height"], clip_image_processor.crop_size["width"])),
    T.ToTensor(),
    T.Normalize(mean=clip_image_processor.image_mean, std=clip_image_processor.image_std)
])

frame_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_root = "./../DATA"
batch_size = 8

dataset = EmbeddingDetDataset(
    data_root_dir=data_root,
    frame_transform=frame_transform
)

loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn
)

model = CombinedModelV4().to(device)
decoder = DecoderV1()

# Initialize and run evaluation
eval_module = EvalModule(model, decoder, device, batch_size=2)
eval_module.evaluate(dataset, result_dir="results", resume_path="weights/model_last_v4.pth")