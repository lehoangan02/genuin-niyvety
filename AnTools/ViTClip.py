import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import numpy as np

class ViTClipEncoder:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # List of model id:
        model_id = "openai/clip-vit-base-patch32"

        print(f"Loading model: {model_id}...")
        self.model = CLIPModel.from_pretrained(model_id)
        # CLIPProcessor handles all the pre-processing (resizing, cropping, tokenizing)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        print("Model and processor loaded.")
    
    def embedImage(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features.unsqueeze(1)
    def embedImageNormalized(self, image: Image.Image) -> torch.Tensor:
        image_embedding = self.embedImage(image)
        image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
        return image_embedding

    def embedText(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features.unsqueeze(1)
    def embedTextNormalized(self, text: str) -> torch.Tensor:
        text_embedding = self.embedText(text)
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
        return text_embedding

if __name__ == "__main__":
    encoder = ClipEncoder()

    image_path = "frame.jpg"
    text_prompt = "gray blue backpack"
    image = Image.open(image_path).convert("RGB")
    image_embedding = encoder.embedImage(image)
    text_embedding = encoder.embedText(text_prompt)
    print(f"Image embedding shape: {image_embedding.shape}")
    print(f"Text embedding shape: {text_embedding.shape}")