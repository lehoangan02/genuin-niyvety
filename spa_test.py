import torch
from torchvision import transforms
from spa.models import spa_vit_base_patch16, spa_vit_large_patch16
from PIL import Image

# load the image
image_path = "img_1.jpg"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
image = transform(image).unsqueeze(0)

# load the model
model = spa_vit_large_patch16(pretrained=True)
model.eval()
model.freeze()

# move to CUDA if available
if torch.cuda.is_available():
    image = image.cuda()

# Obtain the reshaped feature map concatenated with [CLS] token
feature_map_cat_cls = model(
    image, feature_map=True, cat_cls=True
)  # torch.Size([1, 2048, 14, 14])

# Obtain the reshaped feature map without [CLS] token
feature_map_wo_cls = model(
    image, feature_map=True, cat_cls=False
)  # torch.Size([1, 1024, 14, 14])