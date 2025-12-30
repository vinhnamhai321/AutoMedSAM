"""
Model architecture for Automatic MedSAM.

This module implements:
1. PromptModule: Lightweight trainable module that generates prompts from embeddings
2. AutoMedSAM: Wrapper combining frozen MedSAM decoder with trainable PromptModule

Architecture follows Section 2.2 of the paper:
- Dense Branch: Generates dense embeddings (256x64x64)
- Sparse Branch: Generates sparse prompts (points/box coordinates)

Note: Image encoder is NOT used during training. Embeddings are pre-computed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np

# Try to import segment_anything for MedSAM
try:
    from segment_anything import sam_model_registry
    SEGMENT_ANYTHING_AVAILABLE = True
except ImportError:
    SEGMENT_ANYTHING_AVAILABLE = False
    print("Warning: segment_anything not installed. Install with: pip install segment-anything")


class DenseBranch(nn.Module):
    """
    Dense Branch of the Prompt Module.
    
    Generates dense embeddings from image embeddings through a lightweight CNN.
    Architecture: 1x1 Conv -> ReLU -> 3x3 Conv
    
    Input: Image Embeddings (B, 256, 64, 64)
    Output: Dense Embeddings (B, 256, 64, 64)
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for module in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dense branch.
        
        Args:
            image_embeddings: Image embeddings [B, 256, 64, 64].
            
        Returns:
            Dense embeddings [B, 256, 64, 64].
        """
        x = self.conv1(image_embeddings)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class SparseBranch(nn.Module):
    """
    Sparse Branch of the Prompt Module.
    
    Generates sparse prompts (point coordinates + labels) from image embeddings.
    Architecture: 1x1 Conv -> ReLU -> MaxPool -> Flatten -> FC
    
    Input: Image Embeddings (B, 256, 64, 64)
    Output: 
        - Point coordinates (B, num_points, 2) in normalized [0, 1] range
        - Point labels (B, num_points) - 1 for foreground, 0 for background
        - Box coordinates (B, 4) as (x1, y1, x2, y2) normalized
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        embedding_size: int = 64,
        hidden_dim: int = 128,
        num_points: int = 5
    ):
        super().__init__()
        
        self.num_points = num_points
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool2d((8, 8))
        
        pool_output_size = hidden_dim * 8 * 8
        
        self.fc_points = nn.Sequential(
            nn.Linear(pool_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_points * 2)
        )
        
        self.fc_labels = nn.Sequential(
            nn.Linear(pool_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_points)
        )
        
        self.fc_box = nn.Sequential(
            nn.Linear(pool_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.conv1.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        
        for module in [self.fc_points, self.fc_labels, self.fc_box]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through sparse branch.
        
        Args:
            image_embeddings: Image embeddings [B, 256, 64, 64].
            
        Returns:
            Tuple of (point_coords, point_labels, box_coords).
        """
        batch_size = image_embeddings.shape[0]
        
        x = self.conv1(image_embeddings)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        
        # Generate point coordinates normalized to [0, 1]
        point_coords = self.fc_points(x)
        point_coords = point_coords.view(batch_size, self.num_points, 2)
        point_coords = torch.sigmoid(point_coords)
        
        # Generate point labels
        point_labels = self.fc_labels(x)
        point_labels = torch.sigmoid(point_labels)
        
        # Generate box coordinates
        box_coords = self.fc_box(x)
        box_coords = torch.sigmoid(box_coords)
        
        # Ensure valid box (x1 < x2, y1 < y2)
        x1 = torch.min(box_coords[:, 0], box_coords[:, 2])
        x2 = torch.max(box_coords[:, 0], box_coords[:, 2])
        y1 = torch.min(box_coords[:, 1], box_coords[:, 3])
        y2 = torch.max(box_coords[:, 1], box_coords[:, 3])
        box_coords = torch.stack([x1, y1, x2, y2], dim=1)
        
        return point_coords, point_labels, box_coords


class PromptModule(nn.Module):
    """
    Prompt Module for Automatic MedSAM (Section 2.2).
    
    This lightweight, trainable module generates both dense and sparse prompts
    from pre-computed image embeddings. It enables automatic segmentation
    without manual prompt input.
    
    Components:
    1. Dense Branch: Generates dense embeddings for the mask decoder
    2. Sparse Branch: Generates point prompts and bounding box coordinates
    
    Input: Image Embeddings (B, 256, 64, 64) from frozen MedSAM encoder
    Output:
        - Dense embeddings (B, 256, 64, 64)
        - Sparse embeddings (point coordinates and labels)
        - Predicted bounding box coordinates
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        embedding_size: int = 64,
        hidden_dim: int = 128,
        num_sparse_points: int = 5
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.embedding_size = embedding_size
        
        self.dense_branch = DenseBranch(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        self.sparse_branch = SparseBranch(
            embedding_dim=embedding_dim,
            embedding_size=embedding_size,
            hidden_dim=hidden_dim,
            num_points=num_sparse_points
        )
    
    def forward(
        self,
        image_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the prompt module.
        
        Args:
            image_embeddings: Image embeddings [B, 256, 64, 64].
            
        Returns:
            Dictionary containing dense_embeddings, point_coords, point_labels, box_coords.
        """
        dense_embeddings = self.dense_branch(image_embeddings)
        point_coords, point_labels, box_coords = self.sparse_branch(image_embeddings)
        
        return {
            'dense_embeddings': dense_embeddings,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'box_coords': box_coords
        }


class AutoMedSAM(nn.Module):
    """
    Automatic MedSAM Model.
    
    Uses pre-computed image embeddings and positional encoding for efficient training.
    Only the PromptModule is trainable; the MedSAM decoder is frozen.
    
    Args:
        medsam_checkpoint_path: Path to MedSAM checkpoint file.
        image_pe: Pre-computed positional encoding tensor [1, 256, 64, 64].
        device: Device to load the model on.
        image_size: Output image size (default 1024).
    """
    
    def __init__(
        self,
        medsam_checkpoint_path: Optional[str] = None,
        image_pe: Optional[torch.Tensor] = None,
        device: str = "cuda",
        image_size: int = 1024
    ):
        super().__init__()
        
        self.device = device
        self.image_size = image_size
        self.medsam = None
        
        # Register positional encoding as buffer (not trainable)
        if image_pe is not None:
            self.register_buffer('image_pe', image_pe)
        else:
            self.image_pe = None
        
        # Initialize PromptModule
        self.prompt_module = PromptModule()
        
        # Load MedSAM decoder if checkpoint provided
        if medsam_checkpoint_path is not None and SEGMENT_ANYTHING_AVAILABLE:
            self._load_medsam(medsam_checkpoint_path)
    
    def _load_medsam(self, checkpoint_path: str) -> None:
        """
        Load MedSAM model from checkpoint.
        
        Only the prompt_encoder and mask_decoder are needed for inference.
        The image_encoder is NOT loaded since we use pre-computed embeddings.
        """
        print(f"Loading MedSAM from {checkpoint_path}...")
        
        self.medsam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.medsam = self.medsam.to(self.device)
        
        # Freeze all MedSAM parameters
        for param in self.medsam.parameters():
            param.requires_grad = False
        
        self.medsam.eval()
        
        # Extract positional encoding if not provided
        if self.image_pe is None:
            pe = self.medsam.prompt_encoder.get_dense_pe()
            self.register_buffer('image_pe', pe.detach())
            print(f"Extracted positional encoding from MedSAM, shape: {pe.shape}")
        
        print("MedSAM loaded and frozen successfully!")
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        return_prompts: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using pre-computed image embeddings.
        
        Args:
            image_embeddings: Pre-computed image embeddings [B, 256, 64, 64].
            return_prompts: Whether to return the generated prompts.
            
        Returns:
            Dictionary containing 'masks' and optionally 'prompts'.
        """
        # Generate prompts using the PromptModule
        prompts = self.prompt_module(image_embeddings)
        
        # Decode masks
        if self.medsam is not None:
            masks = self._decode_masks(image_embeddings, prompts)
        else:
            masks = self._decode_masks_lightweight(prompts['dense_embeddings'])
        
        output = {'masks': masks}
        if return_prompts:
            output['prompts'] = prompts
        
        return output
    
    def _decode_masks(
        self,
        image_embeddings: torch.Tensor,
        prompts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode masks using MedSAM's mask decoder.
        
        Args:
            image_embeddings: Image embeddings [B, 256, 64, 64].
            prompts: Dictionary of prompts from PromptModule.
            
        Returns:
            Predicted masks [B, 1, H, W].
        """
        batch_size = image_embeddings.shape[0]
        
        # Scale point coordinates from [0, 1] to image size
        point_coords = prompts['point_coords'] * self.image_size
        point_labels = prompts['point_labels']
        box_coords = prompts['box_coords'] * self.image_size
        
        masks_list = []
        
        for b in range(batch_size):
            # Get sparse embeddings from prompt encoder
            sparse_embeddings, dense_embeddings = self.medsam.prompt_encoder(
                points=(point_coords[b:b+1], point_labels[b:b+1].round()),
                boxes=box_coords[b:b+1],
                masks=None
            )
            
            # Add learned dense embeddings from PromptModule
            prompt_dense = prompts['dense_embeddings'][b:b+1]
            prompt_dense = F.interpolate(
                prompt_dense,
                size=dense_embeddings.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            dense_embeddings = dense_embeddings + prompt_dense
            
            # Decode masks using pre-loaded positional encoding
            low_res_masks, _ = self.medsam.mask_decoder(
                image_embeddings=image_embeddings[b:b+1],
                image_pe=self.image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            # Upscale masks to original resolution
            masks = F.interpolate(
                low_res_masks,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
            
            masks_list.append(masks)
        
        masks = torch.cat(masks_list, dim=0)
        masks = torch.sigmoid(masks)
        
        return masks
    
    def _decode_masks_lightweight(
        self,
        dense_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Lightweight mask decoding without MedSAM (fallback).
        """
        if not hasattr(self, '_fallback_decoder'):
            self._fallback_decoder = nn.Sequential(
                nn.Conv2d(256, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            ).to(dense_embeddings.device)
        
        masks = self._fallback_decoder(dense_embeddings)
        masks = F.interpolate(
            masks,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        masks = torch.sigmoid(masks)
        
        return masks
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (PromptModule)."""
        return self.prompt_module.parameters()
    
    def train(self, mode: bool = True):
        """Set training mode. Only PromptModule is trainable."""
        super().train(mode)
        if self.medsam is not None:
            self.medsam.eval()
        return self
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint (only PromptModule weights)."""
        torch.save({
            'prompt_module_state_dict': self.prompt_module.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")


def create_model(
    medsam_checkpoint_path: Optional[str] = None,
    image_pe: Optional[torch.Tensor] = None,
    device: str = "cuda"
) -> nn.Module:
    """
    Factory function to create the AutoMedSAM model.
    
    Args:
        medsam_checkpoint_path: Path to MedSAM checkpoint.
        image_pe: Pre-computed positional encoding tensor.
        device: Device to use.
        
    Returns:
        Model instance.
    """
    if not SEGMENT_ANYTHING_AVAILABLE or medsam_checkpoint_path is None:
        print("Warning: MedSAM not available, using lightweight decoder")
    
    model = AutoMedSAM(
        medsam_checkpoint_path=medsam_checkpoint_path,
        image_pe=image_pe,
        device=device
    )
    
    return model.to(device)
