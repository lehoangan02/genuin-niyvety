import torch
from PIL import Image
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

class MobileClipEncoder:
    def __init__(self, model_name="MobileCLIP-B", pretrained="datacompdr_lt"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading {model_name} ({pretrained}) ...")

        # Load MobileCLIP-B (LT) from OpenCLIP interface
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Reparameterize for inference
        self.model = reparameterize_model(self.model)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("✅ MobileCLIP-B-LT loaded successfully.")

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
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)


if __name__ == "__main__":
    image_path = "frame.jpg"
    text_prompt = "gray blue backpack"

    encoder = MobileClipEncoder()
    image = Image.open(image_path).convert("RGB")

    image_embedding = encoder.embedImageNormalized(image)
    text_embedding = encoder.embedTextNormalized(text_prompt)

    print(f"Image embedding shape: {image_embedding.shape}")
    print(f"Text embedding shape: {text_embedding.shape}")
