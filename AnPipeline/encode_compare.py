from mobileClip import MobileClipEncoder
from AnPipeline.ViTClip import ViTClipEncoder
import torch
import numpy as np
import cv2
import time
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


encoder = MobileClipEncoder() # use mobileclip encoder
encoder2 = ViTClipEncoder() # use clip encoder


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLOE("yoloe-11l-seg.pt").to(device)

names = ["gray blue backpack"]
model_cpu = model.to("cpu")

text_emb1 = model_cpu.get_text_pe(names)
text_emb2 = encoder.embedTextNormalized(names[0])
text_emb3 = encoder.embedText(names[0])
text_emb4 = encoder2.embedTextNormalized(names[0])
text_emb5 = encoder2.embedText(names[0])

text_emb1 = text_emb1.to(device).squeeze(0).squeeze(0)
text_emb2 = text_emb2.to(device).squeeze(0).squeeze(0)
text_emb3 = text_emb3.to(device).squeeze(0).squeeze(0)
text_emb4 = text_emb4.to(device).squeeze(0).squeeze(0)
text_emb5 = text_emb5.to(device).squeeze(0).squeeze(0)

# Verify that the embeddings are similar
print("Text Embedding from YOLOE model:", text_emb1)
print("Text Embedding from MobileClipEncoder (Normalized):", text_emb2)
print("Text Embedding from MobileClipEncoder (Unnormalized):", text_emb3)
print("Text Embedding from ClipEncoder (Normalized):", text_emb4)
print("Text Embedding from ClipEncoder (Unnormalized):", text_emb5)

# check cosine similarity
cos = torch.nn.CosineSimilarity(dim=-1)
similarity1 = cos(text_emb1, text_emb2)
similarity2 = cos(text_emb1, text_emb3)
similarity3 = cos(text_emb1, text_emb4)
similarity4 = cos(text_emb1, text_emb5)
print(f"Cosine Similarity between YOLOE and MobileClipEncoder (Normalized): {similarity1.item()}")
print(f"Cosine Similarity between YOLOE and MobileClipEncoder (Unnormalized): {similarity2.item()}")
print(f"Cosine Similarity between YOLOE and ViTClipEncoder (Normalized): {similarity3.item()}")
print(f"Cosine Similarity between YOLOE and ViTClipEncoder (Unnormalized): {similarity4.item()}")