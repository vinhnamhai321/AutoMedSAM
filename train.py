"""
Training engine for Automatic MedSAM.

This module implements training using pre-computed image embeddings.
The embeddings and positional encoding are loaded from disk at startup.

Features:
- Automatic Mixed Precision (AMP) for memory efficiency
- Pre-computed embeddings loaded from disk
- Progress tracking with tqdm
- Visual debugging with snapshot generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime

from config import AutoMedSAMConfig
from model import AutoMedSAM, create_model
from loss import AutoMedSAMLoss, compute_dice_score
from visualization import Visualizer, visualize_batch
from dataset import PrecomputedEmbeddingDataset, load_positional_encoding, create_dataloaders


class Trainer:
    """
    Training engine for Automatic MedSAM.
    
    Uses pre-computed image embeddings loaded from disk for efficient training.
    
    Args:
        config: Training configuration.
        model: AutoMedSAM model.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader (optional).
    """
    
    def __init__(
        self,
        config: AutoMedSAMConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Loss function
        # Loss function
        self.criterion = AutoMedSAMLoss(
            lambda_tightness=config.lambda_tightness,
            lambda_size=config.lambda_size,
            lambda_bbox=config.lambda_bbox,
            barrier_t=config.barrier_t,
            size_lower_bound=config.size_lower_bound,
            size_upper_bound=config.size_upper_bound
        )
        
        # Optimizer (only trainable parameters)
        self.optimizer = optim.Adam(
            self.model.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # AMP scaler
        self.scaler = GradScaler("cuda") if config.use_amp else None
        
        # Visualizer
        self.visualizer = Visualizer(
            snapshot_dir=config.snapshot_dir,
            log_dir=config.log_dir
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_dice = 0.0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run the complete training pipeline.
        
        Returns:
            Training history dictionary.
        """
        print("=" * 60)
        print("Automatic MedSAM Training")
        print("=" * 60)
        print(self.config)
        
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Training PromptModule with Pre-computed Embeddings")
        print("=" * 60)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Validation epoch
            val_metrics = None
            if self.val_dataloader is not None and (epoch + 1) % self.config.validate_every_n_epochs == 0:
                val_metrics = self._validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch results
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)
            
            # Save checkpoint
            dice_to_check = val_metrics['dice'] if val_metrics else train_metrics['dice']
            if dice_to_check > self.best_dice:
                self.best_dice = dice_to_check
                self._save_checkpoint(is_best=True)
            
            # Save latest checkpoint
            self._save_checkpoint(is_best=False, suffix="latest_epoch")
        
        # Save final artifacts
        self._save_training_artifacts()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        
        return self.history
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_l_empty = 0.0
        epoch_l_tightbox = 0.0
        epoch_l_size = 0.0
        epoch_l_bbox = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Load pre-computed embeddings
            embeddings = batch['embedding'].to(self.config.device)
            masks = batch['mask'].to(self.config.device)
            bboxes = batch['bbox'].to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast("cuda"):
                    output = self.model(embeddings, return_prompts=True)
                    predictions = output['masks']
                    pred_bboxes = output['prompts']['box_coords']
                    loss_output = self.criterion(predictions, bboxes, pred_bboxes)
                
                # Backward pass with AMP
                self.scaler.scale(loss_output.total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(embeddings, return_prompts=True)
                predictions = output['masks']
                pred_bboxes = output['prompts']['box_coords']
                loss_output = self.criterion(predictions, bboxes, pred_bboxes)
                
                # Backward pass
                loss_output.total_loss.backward()
                self.optimizer.step()
            
            # Compute Dice score
            dice = compute_dice_score(predictions.detach(), masks)
            
            # Update running metrics
            epoch_loss += loss_output.total_loss.item()
            epoch_dice += dice
            epoch_l_empty += loss_output.l_empty.item()
            epoch_l_tightbox += loss_output.l_tightbox.item()
            epoch_l_size += loss_output.l_size.item()
            epoch_l_bbox += loss_output.l_bbox.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_output.total_loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
            
            # Log metrics for visualization
            self.visualizer.log_metrics(
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                loss_dict=loss_output.to_dict(),
                dice_score=dice,
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
            
            # Save visual snapshots
            if (batch_idx + 1) % self.config.save_snapshot_every_n_batches == 0:
                with torch.no_grad():
                    batch_with_prompts = {
                        'image': batch['image'].cpu(),
                        'mask': masks.cpu(),
                        'bbox': bboxes.cpu(),
                        'name': batch.get('name', [f'sample_{i}' for i in range(masks.shape[0])]),
                        'prompts': {k: v.cpu() for k, v in output['prompts'].items()} if 'prompts' in output else {}
                    }
                    
                    # visualize_batch(
                    #     batch=batch_with_prompts,
                    #     predictions=predictions.cpu(),
                    #     epoch=self.current_epoch,
                    #     batch_idx=batch_idx,
                    #     visualizer=self.visualizer,
                    #     loss_dict=loss_output.to_dict(),
                    #     max_samples=2
                    # )
            
            self.global_step += 1
            
            # Memory management
            del embeddings, predictions, loss_output
            torch.cuda.empty_cache()
        
        # Compute epoch averages
        metrics = {
            'loss': epoch_loss / num_batches,
            'dice': epoch_dice / num_batches,
            'l_empty': epoch_l_empty / num_batches,
            'l_tightbox': epoch_l_tightbox / num_batches,
            'l_size': epoch_l_size / num_batches,
            'l_bbox': epoch_l_bbox / num_batches
        }
        
        self.history['train_loss'].append(metrics['loss'])
        self.history['train_dice'].append(metrics['dice'])
        
        return metrics
    
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        val_loss = 0.0
        val_dice = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validation"):
            embeddings = batch['embedding'].to(self.config.device)
            masks = batch['mask'].to(self.config.device)
            bboxes = batch['bbox'].to(self.config.device)
            
            if self.config.use_amp:
                with autocast("cuda"):
                    output = self.model(embeddings, return_prompts=True)
                    predictions = output['masks']
                    pred_bboxes = output['prompts']['box_coords']
                    loss_output = self.criterion(predictions, bboxes, pred_bboxes)
            else:
                output = self.model(embeddings, return_prompts=True)
                predictions = output['masks']
                pred_bboxes = output['prompts']['box_coords']
                loss_output = self.criterion(predictions, bboxes, pred_bboxes)
            
            dice = compute_dice_score(predictions, masks)
            
            val_loss += loss_output.total_loss.item()
            val_dice += dice
            num_batches += 1
        
        metrics = {
            'loss': val_loss / num_batches,
            'dice': val_dice / num_batches
        }
        
        self.history['val_loss'].append(metrics['loss'])
        self.history['val_dice'].append(metrics['dice'])
        
        return metrics
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        learning_rate: float
    ) -> None:
        """Log epoch results."""
        self.history['learning_rate'].append(learning_rate)
        
        log_str = f"\nEpoch {epoch + 1}/{self.config.num_epochs}"
        log_str += f"\n  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}"
        
        if val_metrics is not None:
            log_str += f"\n  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}"
        
        log_str += f"\n  LR: {learning_rate:.6f}"
        
        print(log_str)
        
        # Save epoch summary visualization
        self.visualizer.save_epoch_summary(
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )
    
    def _save_checkpoint(self, is_best: bool = False, suffix: str = "") -> None:
        """Save model checkpoint."""
        if is_best:
            path = self.config.checkpoint_dir / "best_model.pth"
        elif suffix:
            path = self.config.checkpoint_dir / f"model_{suffix}.pth"
        else:
            path = self.config.checkpoint_dir / f"model_epoch_{self.current_epoch + 1}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.prompt_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'config': self.config.to_dict()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def _save_training_artifacts(self) -> None:
        """Save training artifacts (loss curves, history, etc.)."""
        self.visualizer.save_loss_curves()
        self.visualizer.save_history()
        
        history_path = self.config.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training artifacts saved to {self.config.log_dir}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint to resume training."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.prompt_module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_dice = checkpoint['best_dice']
        
        print(f"Resumed from epoch {self.current_epoch}, best dice: {self.best_dice:.4f}")


def train_automatic_medsam(config: AutoMedSAMConfig) -> Dict[str, List[float]]:
    """
    Main training function.
    
    Args:
        config: Training configuration.
        
    Returns:
        Training history.
    """
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print(f"\n{'='*60}")
    print(f"Training Task: {config.task}")
    print(f"{'='*60}")
    
    # Load positional encoding
    image_pe = load_positional_encoding(config.data_dir)
    
    # Create dataloaders with task parameter
    train_dataloader, val_dataloader = create_dataloaders(
        data_dir=config.data_dir,
        task=config.task,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create model with pre-loaded positional encoding
    model = create_model(
        medsam_checkpoint_path=str(config.medsam_checkpoint_path),
        image_pe=image_pe,
        device=config.device
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    # Resume from checkpoint if provided
    if config.checkpoint_path and Path(config.checkpoint_path).exists():
        print(f"Resuming from checkpoint: {config.checkpoint_path}")
        trainer.load_checkpoint(str(config.checkpoint_path))
    
    # Train
    history = trainer.train()
    
    return history
