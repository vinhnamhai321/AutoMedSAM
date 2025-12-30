"""
Main entry point for Automatic MedSAM training.

This script provides a command-line interface for training the Automatic MedSAM
model using pre-computed image embeddings.

Expected data directory structure:
data/
├── <dataset_name>/ (e.g., ACDC)
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   │   └── image_embeddings/
│   ├── val/
│   │   ├── images/
│   │   ├── masks/
│   │   └── image_embeddings/
│   └── positional_encoding/
│       └── pe.pt

Usage:
    python main.py --data_dir ./data/ACDC --epochs 100 --batch_size 4
"""

# Fix OpenMP duplicate library issue (must be before other imports)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
import torch

from config import AutoMedSAMConfig
from train import train_automatic_medsam
from utils import set_seed, print_gpu_memory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automatic MedSAM - Weak Supervision Medical Image Segmentation"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., ./data/ACDC)"
    )
    parser.add_argument(
        "--medsam_checkpoint",
        type=str,
        default="./medsam_vit_b.pth",
        help="Path to MedSAM checkpoint"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    # Loss weights
    parser.add_argument(
        "--lambda_tightness",
        type=float,
        default=1e-4,
        help="Weight for tightness constraint (λ₁)"
    )
    parser.add_argument(
        "--lambda_size",
        type=float,
        default=1e-2,
        help="Weight for size constraint (λ₂)"
    )
    
    # Memory optimization
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable Automatic Mixed Precision"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=10,
        help="Save visualization snapshot every N batches"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Print header
    print("=" * 70)
    print("  Automatic MedSAM - Weak Supervision Medical Image Segmentation")
    print("=" * 70)
    print()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    snapshot_dir = output_dir / "debug_snapshots"
    
    for d in [checkpoint_dir, log_dir, snapshot_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Verify data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for required subdirectories
    required_dirs = [
        data_dir / "train" / "image_embeddings",
        data_dir / "train" / "masks",
        data_dir / "val" / "image_embeddings",
        data_dir / "val" / "masks",
        data_dir / "positional_encoding"
    ]
    
    for req_dir in required_dirs:
        if not req_dir.exists():
            raise FileNotFoundError(f"Required directory not found: {req_dir}")
    
    # Create configuration
    config = AutoMedSAMConfig(
        # Data paths
        data_dir=data_dir,
        medsam_checkpoint_path=Path(args.medsam_checkpoint),
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=Path(args.resume) if args.resume else None,
        log_dir=log_dir,
        snapshot_dir=snapshot_dir,
        
        # Training parameters
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        
        # Loss weights
        lambda_tightness=args.lambda_tightness,
        lambda_size=args.lambda_size,
        
        # Memory optimization
        use_amp=not args.no_amp,
        
        # Other settings
        save_snapshot_every_n_batches=args.snapshot_every,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Print configuration
    print(config)
    print()
    
    # Print GPU info
    print("Hardware Information:")
    print_gpu_memory(prefix="  ")
    print()
    
    # Check if MedSAM checkpoint exists
    if not config.medsam_checkpoint_path.exists():
        print(f"Warning: MedSAM checkpoint not found at {config.medsam_checkpoint_path}")
        print("Model will use lightweight decoder without MedSAM backbone")
        print()
    
    # Run training
    print("Starting training...")
    print()
    
    history = train_automatic_medsam(config=config)
    
    # Print final results
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    if history['train_dice']:
        print(f"Final Training Dice: {history['train_dice'][-1]:.4f}")
    if history['val_dice']:
        print(f"Final Validation Dice: {history['val_dice'][-1]:.4f}")
    
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"Snapshots saved to: {snapshot_dir}")


if __name__ == "__main__":
    main()
