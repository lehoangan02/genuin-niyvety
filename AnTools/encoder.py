import torch
from PIL import Image
import open_clip
from transformers import CLIPProcessor, CLIPModel
from mobileclip.modules.common.mobileone import reparameterize_model

class MobileClipBLTEncoder:
    def __init__(self):
        model_name="MobileCLIP-B"
        pretrained="datacompdr_lt"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading {model_name} ({pretrained}) ...")

        # Load model & transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Load tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Reparameterize model for inference
        self.model = reparameterize_model(self.model)

        # Move to device & eval mode
        self.model.to(self.device)
        self.model.eval()
        print("✅ MobileCLIP-B (LT) loaded and reparameterized.")

    def embedImage(self, image: Image.Image):
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        return image_features.unsqueeze(1)

    def embedText(self, text: str):
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.unsqueeze(1)

    def embedImageNormalized(self, image: Image.Image):
        image_features = self.embedImage(image)
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    def embedTextNormalized(self, text: str):
        text_features = self.embedText(text)
        return text_features / text_features.norm(p=2, keepdim=True)
class MobileClipEncoder:
    def __init__(self):
        model_name="MobileCLIP-B"
        pretrained="datacompdr_lt"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading {model_name} ({pretrained}) ...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully.")

    def embedImage(self, image: Image.Image):
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        return image_features.unsqueeze(1)

    def embedText(self, text: str):
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.unsqueeze(1)

    def embedImageNormalized(self, image: Image.Image):
        image_features = self.embedImage(image)
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    def embedTextNormalized(self, text: str):
        text_features = self.embedText(text)
        return text_features / text_features.norm(p=2, keepdim=True)
class ViTClipEncoder:
    def __init__(self):
        model_id: str = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # List of model id:
        model_id = "openai/clip-vit-base-patch32"

        print(f"Loading model: {model_id}...")
        self.model = CLIPModel.from_pretrained(model_id)
        # CLIPProcessor handles all the pre-processing (resizing, cropping, tokenizing)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
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
    encoder = ViTClipEncoder()

    image_path = "frame.jpg"
    text_prompt = "gray blue backpack"
    image = Image.open(image_path).convert("RGB")
    image_embedding = encoder.embedImage(image)
    text_embedding = encoder.embedText(text_prompt)
    print(f"Image embedding shape: {image_embedding.shape}")
    print(f"Text embedding shape: {text_embedding.shape}")