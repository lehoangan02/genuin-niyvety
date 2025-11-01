import torch
import torch.nn as nn
import timm
from transformers import CLIPModel

class CombinedModel(nn.Module):
    def __init__(self, 
                 clip_model_id="openai/clip-vit-base-patch32", 
                 fastvit_model_id="fastvit_sa12",
                 freeze_encoders=True):
        
        super().__init__()
        
        # 1. Load FastViT backbone
        self.fastvit_backbone = timm.create_model(
            fastvit_model_id, 
            pretrained=True, 
            features_only=True  # This gives us a list of feature maps
        )
        
        # 2. Load CLIP model (we only need the image encoder)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id, use_safetensors=True)
        
        # Freeze parameters if we are not fine-tuning the backbones
        if freeze_encoders:
            for param in self.fastvit_backbone.parameters():
                param.requires_grad = False
            for param in self.clip_model.parameters():
                param.requires_grad = False
                
    def forward(self, query_batch, frame_batch):
        """
        Processes the batch of query images and frame images.
        
        Args:
            query_batch (torch.Tensor): Shape [B, 3, C, H_q, W_q]
            frame_batch (torch.Tensor): Shape [B, C, H_f, W_f]
        
        Returns:
            query_features (torch.Tensor): Shape [B, 3, 512]
            frame_features (list[torch.Tensor]): List of feature maps
        """
        
        # --- 1. Process Query Images with CLIP ---
        
        # Get dimensions
        B, N_queries, C, H_q, W_q = query_batch.shape 
        
        # The CLIP model expects a batch of single images [B*N, C, H, W]
        # So, we "unroll" the batch dimension
        # New shape: [B * 3, C, H_q, W_q]
        query_batch_flat = query_batch.view(B * N_queries, C, H_q, W_q)
        
        # Get image features from CLIP
        # Note: We use .get_image_features() not the full .forward()
        # Output shape: [B * 3, 512] (for vit-base-patch32)
        query_features_flat = self.clip_model.get_image_features(
            pixel_values=query_batch_flat
        )
        
        # "Roll up" the features back to our batch structure
        # Output shape: [B, 3, 512]
        query_features = query_features_flat.view(B, N_queries, -1)
        
        
        # --- 2. Process Frame Image with FastViT ---
        
        # FastViT (with features_only=True) outputs a list of feature maps
        # (e.g., [B, 48, H/4, W/4], [B, 96, H/8, W/8], ...)
        frame_features = self.fastvit_backbone(frame_batch)
        
        
        return query_features, frame_features