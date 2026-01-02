"""
Configuration module for Automatic MedSAM.

This module defines all hyperparameters and paths using a dataclass
for clean configuration management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import torch


@dataclass
class AutoMedSAMConfig:
    """
    Configuration class for Automatic MedSAM training.
    
    This dataclass manages all hyperparameters following the paper's specifications:
    - Image size: 1024x1024 (MedSAM default)
    - Pre-computed embeddings loaded from disk
    - Weak supervision with tight bounding boxes
    
    Attributes:
        image_size: Target image size for MedSAM (default 1024x1024).
        embedding_size: Size of image embeddings from MedSAM encoder (64x64).
        embedding_dim: Dimension of embedding channels (256).
        batch_size: Training batch size.
        learning_rate: Learning rate for PromptModule optimizer.
        num_epochs: Total training epochs.
        lambda_tightness: Weight for tightness loss (λ₁ in paper).
        lambda_size: Weight for size constraint loss (λ₂ in paper).
        barrier_t: Parameter t for pseudo log-barrier function.
        use_amp: Whether to use Automatic Mixed Precision.
        num_workers: DataLoader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for reproducibility.
    """
    
    # ==================== Task Configuration ====================
    # Task to train: 'LV' (Left Ventricle) or 'RV' (Right Ventricle)
    task: str = "LV"
    
    # ==================== Model Architecture ====================
    image_size: Tuple[int, int] = (1024, 1024)
    embedding_size: Tuple[int, int] = (64, 64)
    embedding_dim: int = 256
    num_sparse_points: int = 5
    
    # ==================== Training Hyperparameters ====================
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # ==================== Loss Weights (from paper Section 2.3) ====================
    lambda_tightness: float = 1e-4  # λ₁ = 0.0001 for tightness constraint
    lambda_size: float = 1e-2       # λ₂ = 0.01 for size constraint
    lambda_bbox: float = 1      # λ₃ = 0.001 for bounding box MSE loss
    barrier_t: float = 5.0          # t parameter for pseudo log-barrier ψ_t(x)
    
    # ==================== Size Constraint Parameters ====================
    size_lower_bound: float = 0.01  # Minimum foreground ratio (a in paper)
    size_upper_bound: float = 0.99  # Maximum foreground ratio (b in paper)
    
    # ==================== Memory Optimization ====================
    use_amp: bool = True
    
    # ==================== DataLoader Settings ====================
    num_workers: int = 4
    pin_memory: bool = True
    
    # ==================== Paths ====================
    # Updated for new ACDC structure: processed_data/ACDC/{split}/{task}/
    data_dir: Path = field(default_factory=lambda: Path("./processed_data/ACDC"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./output/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("./output/logs"))
    snapshot_dir: Path = field(default_factory=lambda: Path("./output/debug_snapshots"))
    medsam_checkpoint_path: Path = field(default_factory=lambda: Path("./medsam_vit_b.pth"))
    checkpoint_path: Optional[Path] = None
    
    # ==================== Visualization ====================
    save_snapshot_every_n_batches: int = 10
    log_every_n_batches: int = 5
    validate_every_n_epochs: int = 1
    
    # ==================== Reproducibility ====================
    seed: int = 42
    
    # ==================== Device ====================
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self) -> None:
        """Create directories and validate configuration after initialization."""
        # Convert string paths to Path objects if needed
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        if isinstance(self.snapshot_dir, str):
            self.snapshot_dir = Path(self.snapshot_dir)
        if isinstance(self.medsam_checkpoint_path, str):
            self.medsam_checkpoint_path = Path(self.medsam_checkpoint_path)
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration parameters."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 < self.lambda_tightness < 1, "λ₁ should be small (0 < λ₁ < 1)"
        assert 0 < self.lambda_size < 1, "λ₂ should be small (0 < λ₂ < 1)"
        assert self.barrier_t > 0, "Barrier parameter t must be positive"
        assert 0 < self.size_lower_bound < self.size_upper_bound < 1, \
            "Size bounds must satisfy 0 < a < b < 1"
        assert self.image_size[0] == self.image_size[1] == 1024, \
            "MedSAM requires 1024x1024 input images"
        assert self.task in ["LV", "RV"], \
            "Task must be 'LV' (Left Ventricle) or 'RV' (Right Ventricle)"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "task": self.task,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "lambda_tightness": self.lambda_tightness,
            "lambda_size": self.lambda_size,
            "barrier_t": self.barrier_t,
            "use_amp": self.use_amp,
            "device": self.device,
            "data_dir": str(self.data_dir),
        }
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["=" * 60, "Automatic MedSAM Configuration", "=" * 60]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


# Default configuration instance
default_config = AutoMedSAMConfig()
