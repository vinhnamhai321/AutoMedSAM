"""
Visualization module for Automatic MedSAM.

This module provides comprehensive visualization tools for monitoring
training progress, including:
- Snapshot generation with predictions, ground truth, and metrics
- Loss curve plotting
- Dice score tracking
- Bounding box visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, Any
from datetime import datetime
import json


class Visualizer:
    """
    Visualization class for training progress tracking.
    
    Creates detailed visual snapshots showing:
    - Original image with bounding boxes (GT in green, predicted in red)
    - Ground truth segmentation mask
    - Predicted soft mask (heatmap)
    - Predicted binary mask
    - Metrics overlay (Dice score, loss values)
    
    Args:
        snapshot_dir: Directory to save snapshots.
        log_dir: Directory to save training logs.
        figsize: Figure size for snapshots.
        dpi: DPI for saved figures.
    """
    
    def __init__(
        self,
        snapshot_dir: Union[str, Path],
        log_dir: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 100
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir) if log_dir else self.snapshot_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        
        # Training history for plotting curves
        self.history = {
            'epoch': [],
            'batch': [],
            'total_loss': [],
            'l_empty': [],
            'l_tightbox': [],
            'l_size': [],
            'l_bbox': [],
            'dice_score': [],
            'learning_rate': []
        }
        
        # Custom colormaps
        self._setup_colormaps()
    
    def _setup_colormaps(self) -> None:
        """Setup custom colormaps for visualization."""
        # Heatmap colormap (blue to red)
        colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        self.heatmap_cmap = LinearSegmentedColormap.from_list('heatmap', colors)
        
        # Mask overlay colormap (transparent to green)
        colors = [(0, 0, 0, 0), (0, 1, 0, 0.5)]
        self.mask_cmap = LinearSegmentedColormap.from_list('mask', colors)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Prepare image for visualization.
        
        Args:
            image: Image tensor [C, H, W] or [H, W].
            
        Returns:
            Image array [H, W, C] or [H, W] for display.
        """
        image = self._to_numpy(image)
        
        if image.ndim == 3:
            # [C, H, W] -> [H, W, C]
            if image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))
            
            # Single channel to grayscale
            if image.shape[-1] == 1:
                image = image[:, :, 0]
            elif image.shape[-1] == 3:
                # Already RGB, normalize if needed
                if image.max() <= 1.0:
                    image = image
                else:
                    image = image / 255.0
        
        # Ensure in [0, 1] range
        if image.max() > 1.0:
            image = image / image.max()
        
        return image
    
    def _prepare_mask(self, mask: torch.Tensor) -> np.ndarray:
        """
        Prepare mask for visualization.
        
        Args:
            mask: Mask tensor [1, H, W] or [H, W].
            
        Returns:
            Mask array [H, W] in [0, 1].
        """
        mask = self._to_numpy(mask)
        
        if mask.ndim == 3:
            mask = mask[0]
        
        return mask
    
    def save_snapshot(
        self,
        image: torch.Tensor,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        bbox_gt: torch.Tensor,
        bbox_pred: Optional[torch.Tensor] = None,
        epoch: int = 0,
        batch_idx: int = 0,
        sample_idx: int = 0,
        loss_dict: Optional[Dict[str, float]] = None,
        dice_score: Optional[float] = None,
        sample_name: str = ""
    ) -> str:
        """
        Save a comprehensive visual snapshot.
        
        Creates a figure with 4 subplots:
        1. Original image with bounding boxes
        2. Ground truth mask
        3. Predicted soft mask (heatmap)
        4. Predicted binary mask
        
        Args:
            image: Original image [C, H, W] or [H, W].
            prediction: Predicted soft mask [1, H, W] or [H, W].
            ground_truth: Ground truth mask [1, H, W] or [H, W].
            bbox_gt: Ground truth bounding box [4] as (x1, y1, x2, y2).
            bbox_pred: Predicted bounding box [4] (optional).
            epoch: Current epoch number.
            batch_idx: Current batch index.
            sample_idx: Sample index within batch.
            loss_dict: Dictionary of loss values.
            dice_score: Dice coefficient score.
            sample_name: Name of the sample.
            
        Returns:
            Path to saved snapshot.
        """
        # Prepare data
        image = self._prepare_image(image)
        pred_mask = self._prepare_mask(prediction)
        gt_mask = self._prepare_mask(ground_truth)
        bbox_gt = self._to_numpy(bbox_gt)
        
        if bbox_pred is not None:
            bbox_pred = self._to_numpy(bbox_pred)
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # ============ Plot 1: Original Image with Bounding Boxes ============
        ax1 = axes[0, 0]
        if image.ndim == 2:
            ax1.imshow(image, cmap='gray')
        else:
            ax1.imshow(image)
        
        # Draw GT bounding box (green)
        x1, y1, x2, y2 = bbox_gt * np.array([w, h, w, h])
        rect_gt = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor='lime', facecolor='none',
            linestyle='-', label='GT BBox'
        )
        ax1.add_patch(rect_gt)
        
        # Draw predicted bounding box (red) if available
        if bbox_pred is not None:
            x1_p, y1_p, x2_p, y2_p = bbox_pred * np.array([w, h, w, h])
            rect_pred = patches.Rectangle(
                (x1_p, y1_p), x2_p - x1_p, y2_p - y1_p,
                linewidth=3, edgecolor='red', facecolor='none',
                linestyle='--', label='Pred BBox'
            )
            ax1.add_patch(rect_pred)
        
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_title(f'Input Image - {sample_name}', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # ============ Plot 2: Ground Truth Mask ============
        ax2 = axes[0, 1]
        ax2.imshow(image, cmap='gray', alpha=0.5)
        ax2.imshow(gt_mask, cmap='Greens', alpha=0.7, vmin=0, vmax=1)
        ax2.set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # ============ Plot 3: Predicted Soft Mask (Heatmap) ============
        ax3 = axes[1, 0]
        im = ax3.imshow(pred_mask, cmap=self.heatmap_cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title('Predicted Soft Mask (Probability)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # ============ Plot 4: Predicted Binary Mask ============
        ax4 = axes[1, 1]
        pred_binary = (pred_mask > 0.5).astype(float)
        ax4.imshow(image, cmap='gray', alpha=0.5)
        ax4.imshow(pred_binary, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
        ax4.set_title('Predicted Binary Mask (threshold=0.5)', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # ============ Add Metrics Text ============
        metrics_text = f"Epoch: {epoch} | Batch: {batch_idx}"
        
        if dice_score is not None:
            metrics_text += f"\nDice Score: {dice_score:.4f}"
        
        if loss_dict is not None:
            metrics_text += f"\n\nLosses:"
            for key, value in loss_dict.items():
                metrics_text += f"\n  {key}: {value:.6f}"
        
        # Add text box
        fig.text(
            0.02, 0.02, metrics_text,
            fontsize=11,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            family='monospace'
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_e{epoch:03d}_b{batch_idx:04d}_s{sample_idx:02d}_{timestamp}.png"
        filepath = self.snapshot_dir / filename
        
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def log_metrics(
        self,
        epoch: int,
        batch_idx: int,
        loss_dict: Dict[str, float],
        dice_score: float,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Log metrics for later visualization.
        
        Args:
            epoch: Current epoch.
            batch_idx: Current batch index.
            loss_dict: Dictionary of loss values.
            dice_score: Dice score.
            learning_rate: Current learning rate.
        """
        self.history['epoch'].append(epoch)
        self.history['batch'].append(batch_idx)
        self.history['total_loss'].append(loss_dict.get('total_loss', 0))
        self.history['l_empty'].append(loss_dict.get('l_empty', 0))
        self.history['l_tightbox'].append(loss_dict.get('l_tightbox', 0))
        self.history['l_size'].append(loss_dict.get('l_size', 0))
        self.history['l_bbox'].append(loss_dict.get('l_bbox', 0))
        self.history['dice_score'].append(dice_score)
        
        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)
    
    def save_loss_curves(self, save_path: Optional[str] = None) -> str:
        """
        Save loss curve plots.
        
        Args:
            save_path: Path to save the figure.
            
        Returns:
            Path to saved figure.
        """
        if len(self.history['total_loss']) == 0:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(self.history['total_loss']))
        
        # Plot 1: Total Loss
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.history['total_loss'], 'b-', linewidth=1.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual Losses
        ax2 = axes[0, 1]
        ax2.plot(iterations, self.history['l_empty'], 'r-', label='L_empty', linewidth=1)
        ax2.plot(iterations, self.history['l_tightbox'], 'g-', label='L_tightbox', linewidth=1)
        ax2.plot(iterations, self.history['l_size'], 'b-', label='L_size', linewidth=1)
        ax2.plot(iterations, self.history['l_bbox'], 'c-', label='L_bbox', linewidth=1)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Dice Score
        ax3 = axes[1, 0]
        ax3.plot(iterations, self.history['dice_score'], 'g-', linewidth=1.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Dice Score')
        ax3.set_title('Dice Score', fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate (if available)
        ax4 = axes[1, 1]
        if len(self.history['learning_rate']) > 0:
            ax4.plot(iterations[:len(self.history['learning_rate'])],
                    self.history['learning_rate'], 'm-', linewidth=1.5)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Learning Rate not logged',
                    ha='center', va='center', fontsize=12)
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = self.log_dir / "loss_curves.png"
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def save_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Save epoch summary plot.
        
        Args:
            epoch: Epoch number.
            train_metrics: Training metrics dictionary.
            val_metrics: Validation metrics dictionary.
            
        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        metrics_names = list(train_metrics.keys())
        train_values = list(train_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='steelblue')
        
        if val_metrics is not None:
            val_values = [val_metrics.get(k, 0) for k in metrics_names]
            bars2 = ax.bar(x + width/2, val_values, width, label='Validation', color='coral')
        
        # Customize
        ax.set_ylabel('Value')
        ax.set_title(f'Epoch {epoch} Summary', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        if val_metrics is not None:
            autolabel(bars2)
        
        plt.tight_layout()
        
        # Save
        save_path = self.log_dir / f"epoch_{epoch:03d}_summary.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def save_history(self) -> str:
        """
        Save training history to JSON.
        
        Returns:
            Path to saved JSON file.
        """
        save_path = self.log_dir / "training_history.json"
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return str(save_path)
    
    def create_comparison_grid(
        self,
        images: List[torch.Tensor],
        predictions: List[torch.Tensor],
        ground_truths: List[torch.Tensor],
        names: List[str],
        save_path: Optional[str] = None,
        max_samples: int = 16
    ) -> str:
        """
        Create a grid comparison of multiple samples.
        
        Args:
            images: List of input images.
            predictions: List of predicted masks.
            ground_truths: List of ground truth masks.
            names: List of sample names.
            save_path: Path to save the figure.
            max_samples: Maximum number of samples to display.
            
        Returns:
            Path to saved figure.
        """
        n_samples = min(len(images), max_samples)
        n_cols = min(4, n_samples)
        n_rows = int(np.ceil(n_samples / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(4 * n_cols * 3, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            
            image = self._prepare_image(images[i])
            pred = self._prepare_mask(predictions[i])
            gt = self._prepare_mask(ground_truths[i])
            
            # Original image
            ax_img = axes[row, col * 3]
            ax_img.imshow(image, cmap='gray' if image.ndim == 2 else None)
            ax_img.set_title(f'{names[i][:15]}', fontsize=8)
            ax_img.axis('off')
            
            # Ground truth
            ax_gt = axes[row, col * 3 + 1]
            ax_gt.imshow(gt, cmap='Greens', vmin=0, vmax=1)
            ax_gt.set_title('GT', fontsize=8)
            ax_gt.axis('off')
            
            # Prediction
            ax_pred = axes[row, col * 3 + 2]
            ax_pred.imshow(pred, cmap='Reds', vmin=0, vmax=1)
            ax_pred.set_title('Pred', fontsize=8)
            ax_pred.axis('off')
        
        # Hide unused axes
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            for j in range(3):
                axes[row, col * 3 + j].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.snapshot_dir / "comparison_grid.png"
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)


def visualize_batch(
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    epoch: int,
    batch_idx: int,
    visualizer: Visualizer,
    loss_dict: Optional[Dict[str, float]] = None,
    dice_scores: Optional[List[float]] = None,
    max_samples: int = 4
) -> List[str]:
    """
    Convenience function to visualize a batch of predictions.
    
    Args:
        batch: Batch dictionary with 'image', 'mask', 'bbox', 'name'.
        predictions: Predicted masks [B, 1, H, W].
        epoch: Current epoch.
        batch_idx: Current batch index.
        visualizer: Visualizer instance.
        loss_dict: Loss dictionary.
        dice_scores: List of dice scores per sample.
        max_samples: Maximum samples to visualize.
        
    Returns:
        List of paths to saved snapshots.
    """
    paths = []
    batch_size = min(predictions.shape[0], max_samples)
    
    for i in range(batch_size):
        dice = dice_scores[i] if dice_scores is not None else None
        name = batch['name'][i] if 'name' in batch else f"sample_{i}"
        
        # Check for predicted bbox from prompts
        bbox_pred = None
        if 'prompts' in batch and 'box_coords' in batch['prompts']:
            bbox_pred = batch['prompts']['box_coords'][i]
        
        path = visualizer.save_snapshot(
            image=batch['image'][i],
            prediction=predictions[i],
            ground_truth=batch['mask'][i],
            bbox_gt=batch['bbox'][i],
            bbox_pred=bbox_pred,
            epoch=epoch,
            batch_idx=batch_idx,
            sample_idx=i,
            loss_dict=loss_dict,
            dice_score=dice,
            sample_name=name
        )
        paths.append(path)
    
    return paths
