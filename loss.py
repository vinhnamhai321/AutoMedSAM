"""
Loss functions for Automatic MedSAM.

This module implements the mathematical constraints from Section 2.3 of the paper:
- L_empty: Cross-entropy on background region (outside bounding box)
- L_tightbox: Tightness constraint using pseudo log-barrier function
- L_size: Foreground size constraint

Total Loss: L_total = L_empty + λ₁ * L_tightbox + λ₂ * L_size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Container for loss values and components."""
    total_loss: torch.Tensor
    l_empty: torch.Tensor
    l_tightbox: torch.Tensor
    l_size: torch.Tensor
    l_bbox: torch.Tensor
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of scalar values for logging."""
        return {
            "total_loss": self.total_loss.item(),
            "l_empty": self.l_empty.item(),
            "l_tightbox": self.l_tightbox.item(),
            "l_size": self.l_size.item(),
            "l_bbox": self.l_bbox.item(),
        }

class PseudoLogBarrier(nn.Module):
    """
    Pseudo Log-Barrier Function ψ_t(x) from Equation (3) in the paper.
    
    This function approximates the indicator function for constraint satisfaction:
    - ψ_t(x) ≈ 0 when x ≤ 0 (constraint satisfied)
    - ψ_t(x) → ∞ when x > 0 (constraint violated)
    
    The pseudo log-barrier is defined as:
        ψ_t(x) = { -1/t * log(-x)           if x ≤ -1/t²
                 { t*x - 1/t*log(1/t²) + 1/t  otherwise
    
    This provides a smooth, differentiable penalty for constraint violations.
    
    Args:
        t: Barrier parameter controlling the sharpness of the transition.
           Larger t makes the function more like a hard constraint.
    """
    
    def __init__(self, t: float = 5.0):
        super().__init__()
        self.t = t
        self.threshold = -1.0 / (t ** 2)  # -1/t²
        self.log_term = (1.0 / self.t) * torch.log(torch.tensor(1.0 / (self.t ** 2)))
        self.constant = 1.0 / self.t
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the pseudo log-barrier function.
        
        Args:
            x: Input tensor. Negative values satisfy the constraint.
            
        Returns:
            Barrier penalty values.
        """
        # Ensure numerical stability
        eps = 1e-7
        
        # Create masks for the two regions
        mask_log = x <= self.threshold  # Region 1: -1/t * log(-x)
        mask_linear = ~mask_log          # Region 2: linear approximation
        
        result = torch.zeros_like(x)
        
        # Region 1: -1/t * log(-x) for x ≤ -1/t²
        if mask_log.any():
            result[mask_log] = -(1.0 / self.t) * torch.log(-x[mask_log] + eps)
        
        # Region 2: t*x - 1/t*log(1/t²) + 1/t for x > -1/t²
        if mask_linear.any():
            log_term = (1.0 / self.t) * torch.log(
                torch.tensor(1.0 / (self.t ** 2), device=x.device, dtype=x.dtype) + eps
            )
            result[mask_linear] = self.t * x[mask_linear] - log_term + (1.0 / self.t)
        
        return result


class EmptyLoss(nn.Module):
    """
    Background Region Loss (L_empty).
    
    Enforces that pixels outside the bounding box should be classified as background.
    Uses binary cross-entropy to push predictions outside the box towards 0.
    
    L_empty = BCE(P_outside, 0) = -mean(log(1 - P_outside))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predictions: torch.Tensor,
        bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the background region loss.
        
        Args:
            predictions: Predicted soft masks [B, 1, H, W] in range [0, 1].
            bboxes: Bounding boxes [B, 4] as (x1, y1, x2, y2) normalized to [0, 1].
            
        Returns:
            Background region loss (scalar).
        """
        batch_size, _, height, width = predictions.shape
        device = predictions.device
        eps = 1e-7
        
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            pred = predictions[b, 0]  # [H, W]
            bbox = bboxes[b]  # [4]
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0].item() * width)
            y1 = int(bbox[1].item() * height)
            x2 = int(bbox[2].item() * width)
            y2 = int(bbox[3].item() * height)
            
            # Clamp to valid range
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            
            # Create mask for outside region
            outside_mask = torch.ones_like(pred, dtype=torch.bool)
            outside_mask[y1:y2, x1:x2] = False
            
            # Get predictions outside the bounding box
            outside_preds = pred[outside_mask]
            
            if outside_preds.numel() > 0:
                # BCE loss: we want these to be 0 (background)
                # L = -log(1 - p) when target is 0
                loss = -torch.log(1 - outside_preds + eps).mean()
                total_loss = total_loss + loss
        
        return total_loss / batch_size


class TightnessLoss(nn.Module):
    """
    Tightness Constraint Loss (L_tightbox) from Equation (2-4) in the paper.
    
    This loss ensures the predicted mask "touches" all four sides of the bounding box.
    For a tight bounding box, the maximum probability along each row/column inside
    the box should sum up to the width/height respectively.
    
    Mathematical formulation:
    - For rows: Σᵢ max_j(P[i,j]) should equal the height of the box
    - For columns: Σⱼ max_i(P[i,j]) should equal the width of the box
    
    We penalize deviations using the pseudo log-barrier function.
    """
    
    def __init__(self, barrier_t: float = 5.0):
        super().__init__()
        self.barrier = PseudoLogBarrier(t=barrier_t)
    
    def forward(
        self,
        predictions: torch.Tensor,
        bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the tightness constraint loss.
        
        Args:
            predictions: Predicted soft masks [B, 1, H, W] in range [0, 1].
            bboxes: Bounding boxes [B, 4] as (x1, y1, x2, y2) normalized to [0, 1].
            
        Returns:
            Tightness loss (scalar).
        """
        batch_size, _, height, width = predictions.shape
        device = predictions.device
        
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            pred = predictions[b, 0]  # [H, W]
            bbox = bboxes[b]  # [4]
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0].item() * width)
            y1 = int(bbox[1].item() * height)
            x2 = int(bbox[2].item() * width)
            y2 = int(bbox[3].item() * height)
            
            # Clamp to valid range
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width <= 0 or box_height <= 0:
                continue
            
            # Extract predictions inside the bounding box
            pred_inside = pred[y1:y2, x1:x2]  # [box_height, box_width]
            
            # Row-wise constraint: sum of max probabilities along each row
            # Should equal box_height for a tight box
            row_max = pred_inside.max(dim=1)[0]  # [box_height]
            row_sum = row_max.sum()
            
            # Column-wise constraint: sum of max probabilities along each column
            # Should equal box_width for a tight box
            col_max = pred_inside.max(dim=0)[0]  # [box_width]
            col_sum = col_max.sum()
            
            # The constraint is: row_sum ≥ box_height and col_sum ≥ box_width
            # Reformulated as: box_height - row_sum ≤ 0 and box_width - col_sum ≤ 0
            # We penalize when the sum is less than expected
            
            row_violation = box_height - row_sum  # Should be ≤ 0
            col_violation = box_width - col_sum   # Should be ≤ 0
            
            # Apply pseudo log-barrier penalty
            loss = self.barrier(row_violation) + self.barrier(col_violation)
            total_loss = total_loss + loss
        
        return total_loss / batch_size


class SizeLoss(nn.Module):
    """
    Foreground Size Constraint Loss (L_size) from Equation (5) in the paper.
    
    This loss constrains the predicted foreground size to be within a reasonable range:
    - Lower bound (a): Prevents degenerate solutions with too small foreground
    - Upper bound (b): Prevents trivial solutions that fill the entire box
    
    L_size = ψ_t(a - S/|B|) + ψ_t(S/|B| - b)
    
    where S is the sum of probabilities inside the box, and |B| is the box area.
    """
    
    def __init__(
        self,
        lower_bound: float = 0.01,
        upper_bound: float = 0.99,
        barrier_t: float = 5.0
    ):
        super().__init__()
        self.lower_bound = lower_bound  # a in the paper
        self.upper_bound = upper_bound  # b in the paper
        self.barrier = PseudoLogBarrier(t=barrier_t)
    
    def forward(
        self,
        predictions: torch.Tensor,
        bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the size constraint loss.
        
        Args:
            predictions: Predicted soft masks [B, 1, H, W] in range [0, 1].
            bboxes: Bounding boxes [B, 4] as (x1, y1, x2, y2) normalized to [0, 1].
            
        Returns:
            Size constraint loss (scalar).
        """
        batch_size, _, height, width = predictions.shape
        device = predictions.device
        
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            pred = predictions[b, 0]  # [H, W]
            bbox = bboxes[b]  # [4]
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0].item() * width)
            y1 = int(bbox[1].item() * height)
            x2 = int(bbox[2].item() * width)
            y2 = int(bbox[3].item() * height)
            
            # Clamp to valid range
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            
            box_area = (x2 - x1) * (y2 - y1)
            
            if box_area <= 0:
                continue
            
            # Extract predictions inside the bounding box
            pred_inside = pred[y1:y2, x1:x2]
            
            # Compute foreground ratio: S / |B|
            foreground_sum = pred_inside.sum()
            foreground_ratio = foreground_sum / box_area
            
            # Lower bound constraint: a - S/|B| ≤ 0
            lower_violation = self.lower_bound - foreground_ratio
            
            # Upper bound constraint: S/|B| - b ≤ 0
            upper_violation = foreground_ratio - self.upper_bound
            
            # Apply pseudo log-barrier penalty
            loss = self.barrier(lower_violation) + self.barrier(upper_violation)
            total_loss = total_loss + loss
        
        return total_loss / batch_size

class BoxLoss(nn.Module):
    """
    Bounding Box Loss (L_bbox).
    
    Computes Mean Squared Error (MSE) between predicted and ground truth bounding boxes.
    This encourages the PromptModule to generate accurate bounding box predictions.
    
    Loss: L_bbox = MSE(pred_bbox, gt_bbox)
    
    where pred_bbox and gt_bbox are normalized to [0, 1].
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(
        self,
        pred_bbox: torch.Tensor,
        gt_bbox: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bounding box MSE loss.
        
        Args:
            pred_bbox: Predicted bounding boxes [B, 4] as (x1, y1, x2, y2) normalized to [0, 1].
            gt_bbox: Ground truth bounding boxes [B, 4] as (x1, y1, x2, y2) normalized to [0, 1].
            
        Returns:
            MSE loss as a scalar tensor.
        """
        # Ensure both are clamped to [0, 1]
        pred_bbox = torch.clamp(pred_bbox, 0, 1)
        gt_bbox = torch.clamp(gt_bbox, 0, 1)
        
        # Compute MSE loss
        loss = self.mse_loss(pred_bbox, gt_bbox)
        
        return loss


class AutoMedSAMLoss(nn.Module):
    """
    Combined loss function for Automatic MedSAM.
    
    Total Loss: L_total = L_empty + λ₁ * L_tightbox + λ₂ * L_size
    
    This implements the complete loss function from Section 2.3 of the paper,
    combining weak supervision signals from tight bounding boxes with
    mathematical constraints to guide the segmentation.
    
    Args:
        lambda_tightness: Weight for tightness constraint (λ₁).
        lambda_size: Weight for size constraint (λ₂).
        barrier_t: Parameter for pseudo log-barrier function.
        size_lower_bound: Lower bound for foreground ratio (a).
        size_upper_bound: Upper bound for foreground ratio (b).
    """
    
    def __init__(
        self,
        lambda_tightness: float = 1e-4,
        lambda_size: float = 1e-2,
        lambda_bbox: float = 1e-3,
        barrier_t: float = 5.0,
        size_lower_bound: float = 0.01,
        size_upper_bound: float = 0.99
    ):
        super().__init__()
        self.lambda_tightness = lambda_tightness
        self.lambda_size = lambda_size
        self.lambda_bbox = lambda_bbox
        
        self.empty_loss = EmptyLoss()
        self.tightness_loss = TightnessLoss(barrier_t=barrier_t)
        self.size_loss = SizeLoss(
            lower_bound=size_lower_bound,
            upper_bound=size_upper_bound,
            barrier_t=barrier_t
        )
        self.box_loss = BoxLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        bboxes: torch.Tensor,
        pred_bboxes: Optional[torch.Tensor] = None,
        return_components: bool = True
    ) -> LossOutput:
        """
        Compute the total loss with all components.
        
        Args:
            predictions: Predicted soft masks [B, 1, H, W] in range [0, 1].
            bboxes: Ground truth bounding boxes [B, 4] as (x1, y1, x2, y2) normalized to [0, 1].
            pred_bboxes: Predicted bounding boxes [B, 4] (optional).
            return_components: Whether to return individual loss components.
            
        Returns:
            LossOutput containing total loss and individual components.
        """
        # Ensure predictions are in valid range [0, 1]
        predictions = torch.clamp(predictions, 0, 1)
        
        # Compute individual loss components
        l_empty = self.empty_loss(predictions, bboxes)
        l_tightbox = self.tightness_loss(predictions, bboxes)
        l_size = self.size_loss(predictions, bboxes)
        
        # Compute bbox loss if predictions are provided
        if pred_bboxes is not None:
            l_bbox = self.box_loss(pred_bboxes, bboxes)
        else:
            l_bbox = torch.tensor(0.0, device=predictions.device)
        
        # Combine losses
        total_loss = (
            l_empty +
            self.lambda_tightness * l_tightbox +
            self.lambda_size * l_size +
            self.lambda_bbox * l_bbox
        )
        
        return LossOutput(
            total_loss=total_loss,
            l_empty=l_empty,
            l_tightbox=l_tightbox,
            l_size=l_size,
            l_bbox=l_bbox
        )


class DiceLoss(nn.Module):
    """
    Dice Loss for optional supervised training comparison.
    
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    DiceLoss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Predicted masks [B, 1, H, W].
            targets: Ground truth masks [B, 1, H, W].
            
        Returns:
            Dice loss (scalar).
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


def compute_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute Dice score for evaluation.
    
    Args:
        predictions: Predicted soft masks [B, 1, H, W].
        targets: Ground truth binary masks [B, 1, H, W].
        threshold: Threshold for binarizing predictions.
        
    Returns:
        Dice score (float between 0 and 1).
    """
    eps = 1e-7
    
    # Binarize predictions
    pred_binary = (predictions > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + eps) / (union + eps)
    
    return dice.item()


def compute_iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute IoU (Intersection over Union) score for evaluation.
    
    Args:
        predictions: Predicted soft masks [B, 1, H, W].
        targets: Ground truth binary masks [B, 1, H, W].
        threshold: Threshold for binarizing predictions.
        
    Returns:
        IoU score (float between 0 and 1).
    """
    eps = 1e-7
    
    # Binarize predictions
    pred_binary = (predictions > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + eps) / (union + eps)
    
    return iou.item()
