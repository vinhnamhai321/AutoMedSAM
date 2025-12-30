"""
Utility functions for Automatic MedSAM.

This module provides helper functions for:
- Seed setting for reproducibility
- GPU memory monitoring
- File path handling
- Metric computation
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import logging
from datetime import datetime


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms (may cause CUDA errors).
                      Default False because some MedSAM operations don't support determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For PyTorch 1.8+
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except RuntimeError:
                pass  # Some operations don't have deterministic implementations


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information.
    
    Returns:
        Dictionary with memory info in GB.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    cached = torch.cuda.memory_reserved(device) / (1024 ** 3)
    free = total - allocated
    
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(device),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "cached_gb": round(cached, 2),
        "free_gb": round(free, 2),
        "utilization_percent": round((allocated / total) * 100, 1)
    }


def print_gpu_memory(prefix: str = "") -> None:
    """Print GPU memory usage."""
    info = get_gpu_memory_info()
    if info["available"]:
        print(f"{prefix}GPU: {info['device_name']}")
        print(f"{prefix}  Allocated: {info['allocated_gb']:.2f} GB / {info['total_gb']:.2f} GB ({info['utilization_percent']:.1f}%)")
        print(f"{prefix}  Free: {info['free_gb']:.2f} GB")
    else:
        print(f"{prefix}GPU not available")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model.
        trainable_only: Only count trainable parameters.
        
    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(num_params: int) -> str:
    """
    Format parameter count for display.
    
    Args:
        num_params: Number of parameters.
        
    Returns:
        Formatted string (e.g., "1.5M").
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def setup_logging(
    log_dir: Path,
    name: str = "automedasm"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files.
        name: Logger name.
        
    Returns:
        Configured logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def normalize_to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor values to [0, 1] range.
    
    Args:
        tensor: Input tensor.
        
    Returns:
        Normalized tensor.
    """
    t_min = tensor.min()
    t_max = tensor.max()
    
    if t_max - t_min > 0:
        return (tensor - t_min) / (t_max - t_min)
    return torch.zeros_like(tensor)


def resize_tensor(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Resize a tensor to target size.
    
    Args:
        tensor: Input tensor [B, C, H, W] or [C, H, W].
        size: Target (H, W).
        mode: Interpolation mode.
        
    Returns:
        Resized tensor.
    """
    import torch.nn.functional as F
    
    needs_batch = tensor.dim() == 3
    if needs_batch:
        tensor = tensor.unsqueeze(0)
    
    resized = F.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )
    
    if needs_batch:
        resized = resized.squeeze(0)
    
    return resized


def compute_bbox_from_mask(
    mask: torch.Tensor,
    margin: int = 0,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute tight bounding box from binary mask.
    
    Args:
        mask: Binary mask [H, W] or [1, H, W].
        margin: Margin to add around the box.
        normalize: Whether to normalize to [0, 1].
        
    Returns:
        Bounding box [4] as (x1, y1, x2, y2).
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    h, w = mask.shape
    
    # Find non-zero coordinates
    nonzero = torch.nonzero(mask > 0.5, as_tuple=True)
    
    if len(nonzero[0]) == 0:
        # Empty mask
        return torch.tensor([0.0, 0.0, 1.0, 1.0])
    
    y_min = nonzero[0].min().item()
    y_max = nonzero[0].max().item()
    x_min = nonzero[1].min().item()
    x_max = nonzero[1].max().item()
    
    # Add margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w - 1, x_max + margin)
    y_max = min(h - 1, y_max + margin)
    
    if normalize:
        bbox = torch.tensor([
            x_min / w,
            y_min / h,
            (x_max + 1) / w,
            (y_max + 1) / h
        ], dtype=torch.float32)
    else:
        bbox = torch.tensor([x_min, y_min, x_max + 1, y_max + 1], dtype=torch.float32)
    
    return bbox


def bbox_to_mask(
    bbox: torch.Tensor,
    size: Tuple[int, int]
) -> torch.Tensor:
    """
    Convert bounding box to binary mask.
    
    Args:
        bbox: Bounding box [4] as (x1, y1, x2, y2) normalized to [0, 1].
        size: Output mask size (H, W).
        
    Returns:
        Binary mask [1, H, W].
    """
    h, w = size
    mask = torch.zeros(1, h, w)
    
    x1 = int(bbox[0].item() * w)
    y1 = int(bbox[1].item() * h)
    x2 = int(bbox[2].item() * w)
    y2 = int(bbox[3].item() * h)
    
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    mask[0, y1:y2, x1:x2] = 1.0
    
    return mask


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset all values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.
        
        Args:
            val: New value.
            n: Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value.
            
        Returns:
            Whether to stop training.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of model architecture.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Model Summary")
    lines.append("=" * 60)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    lines.append(f"Total parameters: {format_parameters(total_params)}")
    lines.append(f"Trainable parameters: {format_parameters(trainable_params)}")
    lines.append(f"Non-trainable parameters: {format_parameters(total_params - trainable_params)}")
    lines.append("")
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        lines.append(f"  {name}: {format_parameters(params)} ({format_parameters(trainable)} trainable)")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
