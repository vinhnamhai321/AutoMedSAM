"""
Dataset module for Automatic MedSAM.

This module provides data loading for training with pre-computed image embeddings.

Expected directory structure:
data/
├── <dataset_name>/ (e.g., ACDC)
│   ├── train/
│   │   ├── images/
│   │   │   ├── sample1.png
│   │   │   └── ...
│   │   ├── masks/
│   │   │   ├── sample1.png
│   │   │   └── ...
│   │   └── image_embeddings/
│   │       ├── sample1.pt
│   │       └── ...
│   ├── val/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── masks/
│   │   │   └── ...
│   │   └── image_embeddings/
│   │       └── ...
│   └── positional_encoding/
│       └── pe.pt  (single file for all images)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union, Any
import numpy as np
from PIL import Image


class PrecomputedEmbeddingDataset(Dataset):
    """
    Dataset for training with pre-computed image embeddings.
    
    Loads pre-computed image embeddings and corresponding masks from the 
    specified directory structure. This enables efficient training without
    running the heavy image encoder at each iteration.
    
    Args:
        data_dir: Root directory of the dataset (e.g., data/ACDC).
        split: Dataset split ('train' or 'val').
        image_size: Target image size for masks (default 1024x1024).
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        image_size: Tuple[int, int] = (1024, 1024)
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        
        # Set up paths
        self.split_dir = self.data_dir / split
        self.images_dir = self.split_dir / "images"
        self.masks_dir = self.split_dir / "masks"
        self.embeddings_dir = self.split_dir / "image_embeddings"
        
        # Validate directories
        if not self.embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {self.embeddings_dir}")
        
        # Load sample names from embeddings directory
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split from {self.data_dir}")
    
    def _load_samples(self) -> List[str]:
        """
        Load sample names from the embeddings directory.
        
        Returns:
            List of sample names (without extension).
        """
        embedding_files = sorted(self.embeddings_dir.glob("*.pt"))
        samples = [f.stem for f in embedding_files]
        return samples
    
    def _load_mask(self, name: str) -> np.ndarray:
        """
        Load mask from file.
        
        Args:
            name: Sample name.
            
        Returns:
            Binary mask as numpy array [1, H, W].
        """
        # Try different extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = self.masks_dir / f"{name}{ext}"
            if mask_path.exists():
                pil_mask = Image.open(mask_path)
                mask = np.array(pil_mask)
                
                # Handle multi-channel masks
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                
                # Binarize and resize
                mask = (mask > 0).astype(np.float32)
                
                # Resize if needed
                if mask.shape != self.image_size:
                    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
                    pil_mask = pil_mask.resize(self.image_size, Image.Resampling.NEAREST)
                    mask = np.array(pil_mask).astype(np.float32) / 255.0
                    mask = (mask > 0.5).astype(np.float32)
                
                # Add channel dimension [1, H, W]
                mask = mask[np.newaxis, :, :]
                return mask
        
        raise FileNotFoundError(f"Mask not found for sample: {name}")
    
    def _load_image(self, name: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            name: Sample name.
            
        Returns:
            Image as numpy array [1, H, W] (grayscale) or [3, H, W] (RGB).
        """
        # Try different extensions
        for ext in ['.png', '.jpg', '.jpeg', '.nii.gz']:
            image_path = self.images_dir / f"{name}{ext}"
            if image_path.exists():
                # Handle medical imaging formats
                if ext == '.nii.gz':
                    try:
                        import nibabel as nib
                        img = nib.load(image_path).get_fdata()
                        # Normalize to [0, 1]
                        if img.max() > 1.0:
                            img = img / img.max()
                        # Handle 3D to 2D (take middle slice)
                        if img.ndim == 3:
                            img = img[:, :, img.shape[2] // 2]
                        if img.ndim == 2:
                            img = img[np.newaxis, :, :]  # [1, H, W]
                        return img.astype(np.float32)
                    except ImportError:
                        pass
                
                # Handle standard image formats
                pil_image = Image.open(image_path)
                image = np.array(pil_image).astype(np.float32)
                
                # Normalize to [0, 1]
                if image.max() > 1.0:
                    image = image / 255.0
                
                # Resize if needed
                if image.shape[:2] != self.image_size:
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
                    pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
                    image = np.array(pil_image).astype(np.float32) / 255.0
                
                # Handle channel dimension
                if image.ndim == 2:
                    image = image[np.newaxis, :, :]  # [1, H, W]
                elif image.ndim == 3:
                    image = np.transpose(image, (2, 0, 1))  # [C, H, W]
                
                return image
        
        raise FileNotFoundError(f"Image not found for sample: {name}")
    
    def _compute_tight_bbox(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute tight bounding box from binary mask.
        
        Args:
            mask: Binary mask [1, H, W].
            
        Returns:
            Bounding box as [x1, y1, x2, y2] normalized to [0, 1].
        """
        mask_2d = mask[0]  # [H, W]
        height, width = mask_2d.shape
        
        # Find non-zero coordinates
        nonzero = np.nonzero(mask_2d)
        
        if len(nonzero[0]) == 0:
            # Empty mask - return full image as bbox
            return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        
        y_min, y_max = nonzero[0].min(), nonzero[0].max()
        x_min, x_max = nonzero[1].min(), nonzero[1].max()
        
        # Normalize to [0, 1]
        bbox = np.array([
            x_min / width,
            y_min / height,
            (x_max + 1) / width,
            (y_max + 1) / height
        ], dtype=np.float32)
        
        return bbox
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with pre-computed embedding.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing:
                - 'embedding': Image embedding [256, 64, 64]
                - 'image': Original image [C, H, W]
                - 'mask': Ground truth mask [1, H, W]
                - 'bbox': Bounding box [4]
                - 'name': Sample name
        """
        name = self.samples[idx]
        
        # Load embedding
        embedding_path = self.embeddings_dir / f"{name}.pt"
        embedding = torch.load(embedding_path, map_location='cpu', weights_only=True)
        
        # Load image
        image = self._load_image(name)
        
        # Load mask
        mask = self._load_mask(name)
        
        # Compute bounding box
        bbox = self._compute_tight_bbox(mask)
        
        return {
            'embedding': embedding,
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).float(),
            'bbox': torch.from_numpy(bbox).float(),
            'name': name
        }


def load_positional_encoding(data_dir: Union[str, Path]) -> torch.Tensor:
    """
    Load the pre-computed positional encoding from the dataset directory.
    
    Args:
        data_dir: Root directory of the dataset (e.g., data/ACDC).
        
    Returns:
        Positional encoding tensor [1, 256, 64, 64].
    """
    pe_path = Path(data_dir) / "positional_encoding" / "pe.pt"
    
    if not pe_path.exists():
        raise FileNotFoundError(f"Positional encoding not found: {pe_path}")
    
    pe = torch.load(pe_path, map_location='cpu', weights_only=True)
    print(f"Loaded positional encoding from {pe_path}, shape: {pe.shape}")
    
    return pe


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Root directory of the dataset.
        batch_size: Batch size for dataloaders.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = PrecomputedEmbeddingDataset(
        data_dir=data_dir,
        split="train"
    )
    
    val_dataset = PrecomputedEmbeddingDataset(
        data_dir=data_dir,
        split="val"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
