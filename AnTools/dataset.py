import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as T
from transformers import CLIPProcessor # We need this to get CLIP's transforms

class FewShotDetDataset(Dataset):
    """
    Modified Dataset to handle separate transforms for 
    query images (for CLIP) and frame images (for FastViT).
    """
    def __init__(self, data_root_dir, query_transform=None, frame_transform=None, phase='train'):
        self.phase = phase
        self.frame_dir = os.path.join(data_root_dir, 'frames')
        self.image_dir = os.path.join(data_root_dir, 'images')
        annot_file_path = os.path.join(data_root_dir, 'label.txt')
        
        self.query_transform = query_transform
        self.frame_transform = frame_transform
        
        with open(annot_file_path, 'r') as f:
            self.annotations = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        line_parts = self.annotations[idx].split()
        video_name = line_parts[0]
        query_names = line_parts[1:4]
        frame_name = line_parts[4]
        bbox_data = [float(coord) for coord in line_parts[5:]]
        
        # Load query images
        query_images = []
        for name in query_names:
            path = os.path.join(self.image_dir, name + ".jpg")
            img = Image.open(path).convert('RGB')
            # Apply the query-specific transform
            if self.query_transform:
                img = self.query_transform(img)
            query_images.append(img)
            
        # Load frame image
        frame_path = os.path.join(self.frame_dir, frame_name)
        frame_image = Image.open(frame_path).convert('RGB')
        # Apply the frame-specific transform
        if self.frame_transform:
            frame_image = self.frame_transform(frame_image)

        # Stack the 3 query images
        query_tensor = torch.stack(query_images)

        # --- Convert bbox from (x1, y1, x2, y2) → (cx, cy, w, h) ---
        x1, y1, x2, y2 = bbox_data[1:]
        # cx = (x1 + x2) / 2.0
        # cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        # bbox_converted = [cx, cy, w, h]
        bbox_converted = [w, h]
        
        # Format the target
        if self.phase == 'train':
            target = {}
            target['boxes'] = torch.tensor([bbox_converted], dtype=torch.float32)
            target['labels'] = torch.tensor([bbox_data[0]], dtype=torch.int64)
            return query_tensor, frame_image, target
        elif self.phase == 'inference':
            return video_name, query_tensor, frame_image

# --- Collate function (Same as before) ---
def custom_collate_fn(batch):
    query_batches, frame_batches, target_batches = zip(*batch)
    query_batch_tensor = torch.stack(query_batches)
    frame_batch_tensor = torch.stack(frame_batches)
    target_list = list(target_batches)
    return query_batch_tensor, frame_batch_tensor, target_list