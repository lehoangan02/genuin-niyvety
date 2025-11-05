import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval import EvalModule, write_results

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataset import FewShotDetDataset, custom_collate_fn
from model import *
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import DecoderV1

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# load model
model = CombinedModelV4().to(device)
# save model weights
torch.save(model.state_dict(), "weights/model_last_v4.pth")