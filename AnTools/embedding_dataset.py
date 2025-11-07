import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from gaussian import calculate_gaussian_radius, apply_gaussian


class EmbeddingDetDataset(Dataset):
    """
    Modified Dataset to handle separate transforms for
    query images (for CLIP) and frame images (for FastViT).
    """

    def __init__(self, data_root_dir, frame_transform=None, phase="train", query_type="embeddings_clip-vit-base-patch32"):
        self.phase = phase
        print("Loading embeddings from type:", query_type)
        self.frame_dir = os.path.join(data_root_dir, "frames")
        self.embedding_dir = os.path.join(
            data_root_dir, query_type
        )
        if self.phase == "train":
            annot_file_path = os.path.join(data_root_dir, "label_train.txt")
        elif self.phase == "inference":
            annot_file_path = os.path.join(data_root_dir, "test_list.txt")
        self.frame_transform = frame_transform

        with open(annot_file_path, "r") as f:
            self.annotations = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        line_parts = self.annotations[idx].split()
        video_name = line_parts[0]
        query_names = line_parts[1:4]
        frame_name = line_parts[4]
        bbox_data = [float(coord) for coord in line_parts[5:]]

        # Load query embeddings instead of raw RGB images
        query_embeddings = []
        for name in query_names:
            path = os.path.join(self.embedding_dir, f"{name}_embedding.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing embedding for query '{name}' at {path}"
                )

            embedding = np.load(path)
            # Ensure embeddings are torch tensors with float32 dtype
            embedding_tensor = torch.from_numpy(embedding).float()
            query_embeddings.append(embedding_tensor)

        # Load frame image
        frame_path = os.path.join(self.frame_dir, frame_name)
        frame_image = Image.open(frame_path).convert("RGB")
        # Apply the frame-specific transform
        if self.frame_transform:
            frame_image = self.frame_transform(frame_image)

        # Stack the 3 query embeddings to build (num_queries, embedding_dim)
        query_tensor = torch.concat(query_embeddings)

        

        # Format the target
        if self.phase == 'train':
            # --- Convert bbox from (x1, y1, x2, y2) → (cx, cy, w, h) ---
            x1, y1, x2, y2 = bbox_data[1:]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            bbox_converted = [cx, cy, w, h]
            target = {}
            target['boxes'] = torch.tensor([bbox_converted], dtype=torch.float32)
            target['labels'] = torch.tensor([bbox_data[0]], dtype=torch.int64)
            
            output_height = h // 4
            output_width = w // 4
            heatmap = torch.zeros((1, output_height, output_width), dtype=torch.float32)
            gaussian_radius = calculate_gaussian_radius(output_height, output_width)
            apply_gaussian(heatmap, (cx / 4.0, cy / 4.0), gaussian_radius)
            target["heatmap"] = heatmap
            return query_tensor, frame_image, target
        elif self.phase == 'inference':
            return video_name, query_names, query_tensor, frame_name, frame_image


# --- Collate function (Same as before) ---
def custom_collate_fn(batch):
    query_batches, frame_batches, target_batches = zip(*batch)
    query_batch_tensor = torch.stack(query_batches)
    frame_batch_tensor = torch.stack(frame_batches)
    target_list = list(target_batches)
    return query_batch_tensor, frame_batch_tensor, target_list
