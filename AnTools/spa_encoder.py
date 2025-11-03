import torch
from torchvision import transforms
from spa.models import spa_vit_base_patch16, spa_vit_large_patch16
from PIL import Image

class spa_encoder_impl:
    def __init__(self):
        self.model = spa_vit_large_patch16(pretrained=True)
        self.model.eval()
        self.model.freeze()

        # Define the device first
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Move model to device
        if self.device == "cuda":
            self.model = self.model.cuda()

        # --- THIS IS THE FIX ---
        # You MUST resize the image to 224x224, which is what the model expects.
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        # ---

    def embedImage(self, image: Image.Image):
        # 1. Apply preprocessing (this will now resize AND convert to tensor)
        image_tensor = self.preprocess(image)
        
        # 2. Add the batch dimension (B, C, H, W)
        image_tensor = image_tensor.unsqueeze(0) 
        
        # 3. Move tensor to the same device as the model
        image_tensor = image_tensor.to(self.device)
        
        # ---

        # Obtain the reshaped feature map concatenated with [CLS] token
        feature_map_cat_cls = self.model(
            image_tensor, feature_map=True, cat_cls=True
        )  # torch.Size([1, 2048, 14, 14])

        # Obtain the reshaped feature map without [CLS] token
        feature_map_wo_cls = self.model(
            image_tensor, feature_map=True, cat_cls=False
        )  # torch.Size([1, 1024, 14, 14])
        
        return feature_map_cat_cls, feature_map_wo_cls