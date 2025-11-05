import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='banpath model')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training from a checkpoint')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, encode}')
    parser.add_argument('--data_dir', type=str, default='./../DATA', help='Path to dataset root directory')
    parser.add_argument('--encoder', type=str, default='clip-vit-base-patch32', help='Phase choice= {clip-vit-base-patch32, mobile-clip-B, mobile-clip-BLT, SPA}')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == 'train':
        from model.combined_model import *
        from train import TrainModule
        # from dataset import FewShotDetDataset, custom_collate_fn
        from embedding_dataset import EmbeddingDetDataset, custom_collate_fn
        from model import *
        from transformers import CLIPProcessor
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
        from model import DecoderV1
        import torch
        model = CombinedModelV4() 
        # --- 1. Define Device ---
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # Transform for the FRAME image (No resize, just normalize)
        # Using standard ImageNet mean/std for the timm model
        frame_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- 3. Create Dataset and DataLoader ---
        data_root = args.data_dir
        batch_size = args.batch_size

        train_dataset = EmbeddingDetDataset(
            data_root_dir=data_root,
            frame_transform=frame_transform
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=args.num_workers
        )
        trainer = TrainModule(train_loader, model) 
        trainer.train_network(args) 
    elif args.phase == 'test':
        from model.combined_model import *
        from eval import EvalModule, write_results
        # from dataset import FewShotDetDataset, custom_collate_fn
        from embedding_dataset import EmbeddingDetDataset, custom_collate_fn
        from model import *
        from transformers import CLIPProcessor
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
        from model import DecoderV1
        import torch
        model = CombinedModelV4() 
        # --- 1. Define Device ---
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # Transform for the FRAME image (No resize, just normalize)
        # Using standard ImageNet mean/std for the timm model
        frame_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- 3. Create Dataset and DataLoader ---
        data_root = args.data_dir
        batch_size = args.batch_size

        train_dataset = EmbeddingDetDataset(
            data_root_dir=data_root,
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
        eval_module = EvalModule(model, decoder, device, batch_size=2) 
        eval_module.evaluate(train_dataset, result_dir="results", resume_path=args.resume)
    elif args.phase == 'encode':
        import encoder as encoder
        if args.encoder == 'clip-vit-base-patch32':
            encoder_model = encoder.ViTClipEncoder()
        elif args.encoder == 'mobile-clip-B':
            encoder_model = encoder.MobileClipEncoder()
        elif args.encoder == 'mobile-clip-BLT':
            encoder_model = encoder.MobileClipBLTEncoder()
        elif args.encoder == 'SPA':
            encoder_model = encoder.SPAEncoder()
        
        # embed all iamges in the dataroot + /images folder and save the embeddings to DATA/embeddings+{encoder}
        import os
        from PIL import Image
        import numpy as np
        image_dir = os.path.join(args.data_dir, 'images')
        embedding_dir = os.path.join(args.data_dir, f'embeddings_{args.encoder}')
        os.makedirs(embedding_dir, exist_ok=True)
        for img_filename in os.listdir(image_dir):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(image_dir, img_filename)
                image = Image.open(img_path).convert('RGB')
                # print image filename and size
                print(f"Processing image: {img_filename}, size: {image.size}")
                base_filename = os.path.splitext(img_filename)[0]
            
            if isinstance(encoder_model, encoder.SPAEncoder):
                # SPAEncoder returns a tuple of 2 feature maps
                emb_cat_cls, emb_wo_cls = encoder_model.embedImageNormalized(image)
                
                # --- 1. Define the two sub-folders ---
                emb1_dir = os.path.join(embedding_dir, "spa_cat_cls")
                emb2_dir = os.path.join(embedding_dir, "spa_wo_cls")

                # --- 2. Create these directories if they don't exist ---
                os.makedirs(emb1_dir, exist_ok=True)
                os.makedirs(emb2_dir, exist_ok=True)

                # --- 3. Define the common filename ---
                common_filename = f"{base_filename}_embedding.npy"

                # --- 4. Define the full paths ---
                emb1_path = os.path.join(emb1_dir, common_filename)
                emb2_path = os.path.join(emb2_dir, common_filename)

                # Squeeze the batch dimension (dim 0) to get [C, H, W]
                emb1_np = emb_cat_cls.squeeze(0).cpu().numpy()
                emb2_np = emb_wo_cls.squeeze(0).cpu().numpy()
                
                # --- 5. Save to the new paths ---
                np.save(emb1_path, emb1_np)
                np.save(emb2_path, emb2_np)
                
                print(f"Saved SPA embeddings for {img_filename} to {emb1_dir} and {emb2_dir}")

            else:
                # Other encoders (ViTClipEncoder, etc.)
                embedding = encoder_model.embedImageNormalized(image)
                
                # Squeeze dim 1 to get [B, D]
                embedding_np = embedding.squeeze(1).cpu().numpy()
                
                # Save directly into the main embedding_dir
                embedding_path = os.path.join(embedding_dir, f"{base_filename}_embedding.npy")
                np.save(embedding_path, embedding_np)
                print(f"Saved embedding for {img_filename} to {embedding_path}")
        
            
    else: 
        print(f"Phase '{args.phase}' not implemented yet.")

    