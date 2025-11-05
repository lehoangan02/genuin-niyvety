import torch
import torch.nn as nn
import timm
from transformers import CLIPModel
from .up_scale_module import NoPromptUpScaleModule


class CombinedModelV1(nn.Module):
    def __init__(
        self,
        clip_model_id="openai/clip-vit-base-patch32",
        fastvit_model_id="fastvit_sa12",
        freeze_encoders=True,
    ):

        super().__init__()

        # 1. Load FastViT backbone
        self.fastvit_backbone = timm.create_model(
            fastvit_model_id,
            pretrained=True,
            features_only=True,  # This gives us a list of feature maps
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


class CombinedModelV2(nn.Module):
    def __init__(
        self,
        clip_model_id="openai/clip-vit-base-patch32",
        fastvit_model_id="fastvit_sa12",
        freeze_encoders=True,
    ):

        super().__init__()

        # 1. Load FastViT backbone
        self.fastvit_backbone = timm.create_model(
            fastvit_model_id,
            pretrained=True,
            features_only=True,  # This gives us a list of feature maps
        )

        # 2. Load CLIP model (we only need the image encoder)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id, use_safetensors=True)

        # Freeze parameters if we are not fine-tuning the backbones
        if freeze_encoders:
            for param in self.fastvit_backbone.parameters():
                param.requires_grad = False
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.up1 = NoPromptUpScaleModule(c_low=512, c_high=256, c_out=256)
        self.up2 = NoPromptUpScaleModule(c_low=256, c_high=128, c_out=128)
        self.up3 = NoPromptUpScaleModule(c_low=128, c_high=64, c_out=64)

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
        f3, f2, f1, f0 = (
            frame_features[3],
            frame_features[2],
            frame_features[1],
            frame_features[0],
        )

        x = self.up1(f3, f2)
        x = self.up2(x, f1)
        x = self.up3(x, f0)
        frame_features = x

        return query_features, frame_features


class CombinedModelV3(nn.Module):
    def __init__(
        self,
        clip_model_id="openai/clip-vit-base-patch32",
        fastvit_model_id="fastvit_sa12",
        freeze_encoders=True,
    ):

        super().__init__()

        # 1. Load FastViT backbone
        self.fastvit_backbone = timm.create_model(
            fastvit_model_id,
            pretrained=True,
            features_only=True,  # This gives us a list of feature maps
        )

        # 2. Load CLIP model (we only need the image encoder)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id, use_safetensors=True)

        # Freeze parameters if we are not fine-tuning the backbones
        if freeze_encoders:
            for param in self.fastvit_backbone.parameters():
                param.requires_grad = False
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.up1 = NoPromptUpScaleModule(c_low=512, c_high=256, c_out=256)
        self.up2 = NoPromptUpScaleModule(c_low=256, c_high=128, c_out=128)
        self.up3 = NoPromptUpScaleModule(c_low=128, c_high=64, c_out=64)

        # simple head to reduce channels to 5
        self.head = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=3, padding=1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.01, inplace=True),
        )

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
        f3, f2, f1, f0 = (
            frame_features[3],
            frame_features[2],
            frame_features[1],
            frame_features[0],
        )

        x = self.up1(f3, f2)
        x = self.up2(x, f1)
        x = self.up3(x, f0)
        frame_features = x
        frame_features = self.head(frame_features)

        return frame_features


class CombinedModelV4(nn.Module):
    """Variant that consumes precomputed query embeddings instead of raw images."""

    def __init__(
        self,
        embedding_dim: int = 512,
        fastvit_model_id: str = "fastvit_sa12",
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        # FastViT backbone that returns intermediate feature maps.
        self.fastvit_backbone = timm.create_model(
            fastvit_model_id,
            pretrained=True,
            features_only=True,
        )

        if freeze_encoder:
            for param in self.fastvit_backbone.parameters():
                param.requires_grad = False

        # Grab channel dimensions to guarantee correct projections.
        feature_channels = self.fastvit_backbone.feature_info.channels()
        if len(feature_channels) < 4:
            raise RuntimeError("Expected at least 4 feature maps from FastViT backbone")

        c3, c2, c1, c0 = (
            feature_channels[3],
            feature_channels[2],
            feature_channels[1],
            feature_channels[0],
        )

        self.query_proj = nn.Linear(embedding_dim, c3)

        # Reuse the same decoder stack as V3.
        self.up1 = NoPromptUpScaleModule(c_low=c3, c_high=c2, c_out=c2)
        self.up2 = NoPromptUpScaleModule(c_low=c2, c_high=c1, c_out=c1)
        self.up3 = NoPromptUpScaleModule(c_low=c1, c_high=c0, c_out=c0)
        self.head = nn.Sequential(
            nn.Conv2d(c0, 5, kernel_size=3, padding=1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(
        self, query_embeddings: torch.Tensor, frame_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: Tensor shaped [B, N_queries, embedding_dim].
            frame_batch: Tensor shaped [B, C, H, W].

        Returns:
            Tensor shaped [B, 5, H_out, W_out] identical to CombinedModelV3.
        """

        if query_embeddings.dim() != 3:
            raise ValueError(
                f"Expected query embeddings with shape [B, N, D], got {tuple(query_embeddings.shape)}"
            )
        if query_embeddings.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {query_embeddings.size(-1)}"
            )

        # Produce backbone features as in previous variants.
        frame_features = self.fastvit_backbone(frame_batch)
        f3, f2, f1, f0 = (
            frame_features[3],
            frame_features[2],
            frame_features[1],
            frame_features[0],
        )

        # Aggregate query embeddings and project to the channel space of f3.
        query_embeddings = query_embeddings.to(frame_batch.device, frame_batch.dtype)
        pooled_query = query_embeddings.mean(dim=1)
        query_modulation = self.query_proj(pooled_query).unsqueeze(-1).unsqueeze(-1)

        # Fuse query information into the coarsest feature map.
        f3 = f3 + query_modulation

        x = self.up1(f3, f2)
        x = self.up2(x, f1)
        x = self.up3(x, f0)
        output = self.head(x)

        return output


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    batch_size, num_queries, embedding_dim = 2, 3, 512
    height, width = 224, 224

    query_embeddings = torch.randn(
        batch_size, num_queries, embedding_dim, device=device
    )
    frame_batch = torch.randn(batch_size, 3, height, width, device=device)

    model = CombinedModelV4(embedding_dim=embedding_dim).to(device)
    model.eval()

    with torch.no_grad():
        output = model(query_embeddings, frame_batch)

    print("CombinedModelV4 output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
