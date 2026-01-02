"""
Test/Evaluation script for Automatic MedSAM.

This script evaluates a trained model on the test set and computes
segmentation metrics (Dice, IoU, Hausdorff Distance, etc.).

Expected data directory structure (task-separated for LV/RV):
processed_data/
├── <dataset_name>/ (e.g., ACDC)
│   ├── test/
│   │   ├── LV/
│   │   │   ├── images/
│   │   │   ├── masks/
│   │   │   └── image_embeddings/
│   │   └── RV/
│   │       └── ...
│   └── positional_encoding/
│       └── pe.pt

Usage:
    # Test LV (Left Ventricle) segmentation
    python test.py --data_dir ./processed_data/ACDC --task LV --checkpoint ./output/LV/checkpoints/best_model.pth
    
    # Test RV (Right Ventricle) segmentation
    python test.py --data_dir ./processed_data/ACDC --task RV --checkpoint ./output/RV/checkpoints/best_model.pth
    
    # Save predictions
    python test.py --data_dir ./processed_data/ACDC --task LV --checkpoint ./output/LV/checkpoints/best_model.pth --save_predictions
"""

# Fix OpenMP duplicate library issue (must be before other imports)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from config import AutoMedSAMConfig
from model import AutoMedSAM, create_model
from dataset import PrecomputedEmbeddingDataset, load_positional_encoding
from loss import compute_dice_score, compute_iou_score
from utils import set_seed, print_gpu_memory
from visualization import Visualizer


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0
) -> float:
    """
    Compute Hausdorff Distance between prediction and target masks.
    
    Args:
        pred: Binary prediction mask [H, W].
        target: Binary target mask [H, W].
        percentile: Percentile for HD95 (default 95.0).
        
    Returns:
        Hausdorff distance value.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        print("Warning: scipy not installed. Hausdorff distance will return 0.")
        return 0.0
    
    # Handle edge cases
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)  # Max possible distance
    
    # Compute distance transforms
    pred_boundary = pred.astype(bool)
    target_boundary = target.astype(bool)
    
    # Distance from prediction boundary to target
    dist_pred_to_target = distance_transform_edt(~target_boundary)
    dist_target_to_pred = distance_transform_edt(~pred_boundary)
    
    # Get distances at boundary points
    pred_to_target_distances = dist_pred_to_target[pred_boundary]
    target_to_pred_distances = dist_target_to_pred[target_boundary]
    
    if len(pred_to_target_distances) == 0 or len(target_to_pred_distances) == 0:
        return 0.0
    
    # Compute HD95
    hd_pred_to_target = np.percentile(pred_to_target_distances, percentile)
    hd_target_to_pred = np.percentile(target_to_pred_distances, percentile)
    
    return max(hd_pred_to_target, hd_target_to_pred)


def compute_surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: float = 2.0
) -> float:
    """
    Compute Surface Dice (Normalized Surface Distance) between prediction and target.
    
    Args:
        pred: Binary prediction mask [H, W].
        target: Binary target mask [H, W].
        tolerance: Tolerance distance in pixels.
        
    Returns:
        Surface Dice score.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return 0.0
    
    # Handle edge cases
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0
    
    pred_boundary = pred.astype(bool)
    target_boundary = target.astype(bool)
    
    # Distance transforms
    dist_pred_to_target = distance_transform_edt(~target_boundary)
    dist_target_to_pred = distance_transform_edt(~pred_boundary)
    
    # Get boundary points
    pred_surface = pred_boundary & ~np.roll(pred_boundary, 1, axis=0)
    target_surface = target_boundary & ~np.roll(target_boundary, 1, axis=0)
    
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return 0.0
    
    # Count points within tolerance
    pred_to_target_close = (dist_pred_to_target[pred_boundary] <= tolerance).sum()
    target_to_pred_close = (dist_target_to_pred[target_boundary] <= tolerance).sum()
    
    total_points = pred_boundary.sum() + target_boundary.sum()
    
    return (pred_to_target_close + target_to_pred_close) / total_points


class Tester:
    """
    Testing engine for Automatic MedSAM.
    
    Evaluates model on test set and computes comprehensive metrics.
    
    Args:
        model: Trained AutoMedSAM model.
        test_dataloader: Test data loader.
        device: Device to run evaluation on.
        save_predictions: Whether to save prediction masks.
        save_visualizations: Whether to save visual snapshots for each sample.
        output_dir: Directory to save results.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        device: str = "cuda",
        save_predictions: bool = False,
        save_visualizations: bool = False,
        output_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_dataloader = test_dataloader
        self.device = device
        self.save_predictions = save_predictions
        self.save_visualizations = save_visualizations
        self.output_dir = output_dir
        
        if save_predictions and output_dir:
            self.predictions_dir = output_dir / "predictions"
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer for saving visual snapshots
        if save_visualizations and output_dir:
            self.visualizer = Visualizer(
                snapshot_dir=output_dir / "visualizations",
                log_dir=output_dir / "logs"
            )
        else:
            self.visualizer = None
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the test set.
        
        Returns:
            Dictionary of aggregated metrics.
        """
        # Per-sample metrics
        all_dice = []
        all_iou = []
        all_hd95 = []
        all_surface_dice = []
        
        # Sample-level results for detailed analysis
        sample_results = []
        
        pbar = tqdm(self.test_dataloader, desc="Testing")
        
        for batch_idx, batch in enumerate(pbar):
            # Load data
            embeddings = batch['embedding'].to(self.device)
            masks = batch['mask'].to(self.device)
            names = batch['name']
            
            # Forward pass (with prompts for visualization)
            output = self.model(embeddings, return_prompts=True)
            predictions = output['masks']
            prompts = output.get('prompts', {})
            
            # Process each sample in batch
            batch_size = embeddings.shape[0]
            for i in range(batch_size):
                pred = predictions[i:i+1]
                target = masks[i:i+1]
                name = names[i]
                
                # Get predicted bbox for visualization
                pred_bbox = prompts.get('box_coords', None)
                if pred_bbox is not None:
                    pred_bbox = pred_bbox[i]
                
                # Compute metrics
                dice = compute_dice_score(pred, target)
                iou = compute_iou_score(pred, target)
                
                # Convert to numpy for HD95 and surface dice
                pred_np = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                target_np = (target[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                
                hd95 = compute_hausdorff_distance(pred_np, target_np, percentile=95.0)
                surf_dice = compute_surface_dice(pred_np, target_np, tolerance=2.0)
                
                # Store metrics
                all_dice.append(dice)
                all_iou.append(iou)
                all_hd95.append(hd95)
                all_surface_dice.append(surf_dice)
                
                sample_results.append({
                    'name': name,
                    'dice': dice,
                    'iou': iou,
                    'hd95': hd95,
                    'surface_dice': surf_dice
                })
                
                # Save prediction mask if requested
                if self.save_predictions and self.output_dir:
                    pred_img = (pred_np * 255).astype(np.uint8)
                    Image.fromarray(pred_img).save(
                        self.predictions_dir / f"{name}_pred.png"
                    )
                
                # Save visualization snapshot if requested
                if self.save_visualizations and self.visualizer is not None:
                    # Get image for visualization
                    image = batch['image'][i] if 'image' in batch else pred[0]
                    gt_bbox = batch['bbox'][i] if 'bbox' in batch else None
                    
                    # Create metrics dict for display
                    metrics_dict = {
                        'dice': dice,
                        'iou': iou,
                        'hd95': hd95
                    }
                    
                    self.visualizer.save_snapshot(
                        image=image,
                        prediction=pred[0],
                        ground_truth=target[0],
                        bbox_gt=gt_bbox,
                        bbox_pred=pred_bbox,
                        epoch=0,  # Not applicable for testing
                        batch_idx=batch_idx,
                        sample_idx=i,
                        loss_dict=metrics_dict,
                        dice_score=dice,
                        sample_name=name
                    )
                
            # Update progress bar
            pbar.set_postfix({
                'dice': f'{np.mean(all_dice):.4f}',
                'iou': f'{np.mean(all_iou):.4f}'
            })
        
        # Compute aggregated metrics
        metrics = {
            'dice_mean': float(np.mean(all_dice)),
            'dice_std': float(np.std(all_dice)),
            'dice_median': float(np.median(all_dice)),
            'dice_min': float(np.min(all_dice)),
            'dice_max': float(np.max(all_dice)),
            'iou_mean': float(np.mean(all_iou)),
            'iou_std': float(np.std(all_iou)),
            'hd95_mean': float(np.mean(all_hd95)),
            'hd95_std': float(np.std(all_hd95)),
            'surface_dice_mean': float(np.mean(all_surface_dice)),
            'surface_dice_std': float(np.std(all_surface_dice)),
            'num_samples': len(all_dice)
        }
        
        # Save detailed results
        if self.output_dir:
            self._save_results(metrics, sample_results)
        
        return metrics
    
    def _save_results(
        self,
        metrics: Dict[str, float],
        sample_results: List[Dict]
    ) -> None:
        """Save evaluation results to files."""
        # Save summary metrics
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
        
        # Save per-sample results
        samples_path = self.output_dir / "test_samples.json"
        with open(samples_path, 'w') as f:
            json.dump(sample_results, f, indent=2)
        print(f"Saved sample results to {samples_path}")
        
        # Save CSV for easy analysis
        csv_path = self.output_dir / "test_results.csv"
        with open(csv_path, 'w') as f:
            f.write("sample_name,dice,iou,hd95,surface_dice\n")
            for r in sample_results:
                f.write(f"{r['name']},{r['dice']:.6f},{r['iou']:.6f},{r['hd95']:.4f},{r['surface_dice']:.6f}\n")
        print(f"Saved CSV results to {csv_path}")


def create_test_dataloader(
    data_dir: Path,
    task: str,
    batch_size: int = 1,
    num_workers: int = 4
) -> DataLoader:
    """
    Create test dataloader for a specific task.
    
    Args:
        data_dir: Root directory of the dataset.
        task: Segmentation task ('LV' or 'RV').
        batch_size: Batch size (default 1 for per-sample metrics).
        num_workers: Number of workers.
        
    Returns:
        Test DataLoader.
    """
    test_dataset = PrecomputedEmbeddingDataset(
        data_dir=data_dir,
        split="test",
        task=task
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader


def load_trained_model(
    checkpoint_path: Path,
    medsam_checkpoint_path: Path,
    image_pe: torch.Tensor,
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to trained model checkpoint.
        medsam_checkpoint_path: Path to MedSAM base checkpoint.
        image_pe: Pre-computed positional encoding.
        device: Device to load model on.
        
    Returns:
        Loaded model ready for evaluation.
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = create_model(
        medsam_checkpoint_path=str(medsam_checkpoint_path),
        image_pe=image_pe,
        device=device
    )
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.prompt_module.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}, "
              f"best dice: {checkpoint.get('best_dice', 'unknown')}")
    else:
        # Direct state dict
        model.prompt_module.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Automatic MedSAM on test set"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., ./processed_data/ACDC)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="LV",
        choices=["LV", "RV"],
        help="Segmentation task: 'LV' (Left Ventricle) or 'RV' (Right Ventricle)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--medsam_checkpoint",
        type=str,
        default="./medsam_vit_b.pth",
        help="Path to MedSAM base checkpoint"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./output/<task>/test_results)"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction masks as images"
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visual snapshots with image, GT, prediction, and bounding boxes"
    )
    
    # Other arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for testing (default 1 for per-sample metrics)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main testing entry point."""
    args = parse_args()
    
    # Print header
    print("=" * 70)
    print("  Automatic MedSAM - Test Evaluation")
    print(f"  Task: {args.task} ({'Left Ventricle' if args.task == 'LV' else 'Right Ventricle'})")
    print("=" * 70)
    print()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("./output") / args.task / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Verify data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for required subdirectories
    task = args.task
    required_dirs = [
        data_dir / "test" / task / "image_embeddings",
        data_dir / "test" / task / "masks",
        data_dir / "positional_encoding"
    ]
    
    for req_dir in required_dirs:
        if not req_dir.exists():
            raise FileNotFoundError(f"Required directory not found: {req_dir}")
    
    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Print GPU info
    print("Hardware Information:")
    print_gpu_memory(prefix="  ")
    print()
    
    # Load positional encoding
    print("Loading positional encoding...")
    image_pe = load_positional_encoding(data_dir)
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_trained_model(
        checkpoint_path=checkpoint_path,
        medsam_checkpoint_path=Path(args.medsam_checkpoint),
        image_pe=image_pe,
        device=device
    )
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    test_loader = create_test_dataloader(
        data_dir=data_dir,
        task=task,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create tester
    tester = Tester(
        model=model,
        test_dataloader=test_loader,
        device=device,
        save_predictions=args.save_predictions,
        save_visualizations=args.save_visualizations,
        output_dir=output_dir
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    print("-" * 70)
    metrics = tester.evaluate()
    print("-" * 70)
    
    # Print results
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    print(f"  Task: {args.task}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Number of samples: {metrics['num_samples']}")
    print()
    print("  Dice Score:")
    print(f"    Mean:   {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"    Median: {metrics['dice_median']:.4f}")
    print(f"    Range:  [{metrics['dice_min']:.4f}, {metrics['dice_max']:.4f}]")
    print()
    print("  IoU Score:")
    print(f"    Mean:   {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
    print()
    print("  Hausdorff Distance (HD95):")
    print(f"    Mean:   {metrics['hd95_mean']:.2f} ± {metrics['hd95_std']:.2f} pixels")
    print()
    print("  Surface Dice:")
    print(f"    Mean:   {metrics['surface_dice_mean']:.4f} ± {metrics['surface_dice_std']:.4f}")
    print("=" * 70)
    
    print(f"\nResults saved to: {output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
