import argparse
from model.combined_model import CombinedModelV3
from train import TrainModule
from dataset import FewShotDetDataset, custom_collate_fn
from model import CombinedModelV3
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import DecoderV1
import encoder as encoder
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='banpath model')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--resume_train', type=str, default=None, help='Path to resume training from a checkpoint')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, encode}')
    parser.add_argument('--data_dir', type=str, default='./../DATA', help='Path to dataset root directory')
    parser.add_argument('--encoder', type=str, default='clip-vit-base-patch32', help='Phase choice= {clip-vit-base-patch32, mobile-clip}')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = CombinedModelV3() 

    # --- 1. Define Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers
    )
    decoder = DecoderV1()

    if args.phase == 'train':
        # NOW pass the train_loader and the decoder
        trainer = TrainModule(train_loader, model, decoder=decoder) 
        trainer.train_network(args) 
    elif args.phase == 'test':
        # Testing code to be implemented
        pass
    elif args.phase == 'encode':
        if args.encoder == 'clip-vit-base-patch32':
            encoder_model = encoder.ViTClipEncoder(model_id="openai/clip-vit-base-patch32")
        elif args.encoder == 'mobile-clip':
            encoder_model = encoder.MobileClipEncoder(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            
    else: 
        print(f"Phase '{args.phase}' not implemented yet.")

    