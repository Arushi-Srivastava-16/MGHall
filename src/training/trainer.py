"""
Multi-task GNN Trainer.

This module implements training loops with multi-task learning for node classification,
origin detection, and error type prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import json

from .wandb_config import log_metrics, log_training_metrics


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class MultiTaskTrainer:
    """Trainer for multi-task GNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        node_loss_weight: float = 1.0,
        origin_loss_weight: float = 2.0,
        error_type_loss_weight: float = 0.5,
        max_epochs: int = 100,
        patience: int = 10,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: GNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            node_loss_weight: Weight for node classification loss
            origin_loss_weight: Weight for origin detection loss
            error_type_loss_weight: Weight for error type classification loss
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
        )
        
        # Loss functions
        self.node_criterion = nn.BCELoss()
        self.origin_criterion = FocalLoss(alpha=0.25, gamma=2.0)  # Focal loss for imbalanced data
        self.error_type_criterion = nn.CrossEntropyLoss()
        
        # Loss weights
        self.node_loss_weight = node_loss_weight
        self.origin_loss_weight = origin_loss_weight
        self.error_type_loss_weight = error_type_loss_weight
        
        # Training config
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            
        Returns:
            Dictionary with individual and total losses
        """
        # Node classification loss
        node_loss = self.node_criterion(outputs['node_pred'], batch.y)
        
        # Origin detection loss
        origin_loss = self.origin_criterion(outputs['origin_pred'], batch.y_origin)
        
        # Error type classification loss
        error_type_loss = self.error_type_criterion(
            outputs['error_type_pred'],
            batch.y_error_type,
        )
        
        # Total loss
        total_loss = (
            self.node_loss_weight * node_loss +
            self.origin_loss_weight * origin_loss +
            self.error_type_loss_weight * error_type_loss
        )
        
        return {
            'total': total_loss,
            'node': node_loss,
            'origin': origin_loss,
            'error_type': error_type_loss,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_node_loss = 0.0
        total_origin_loss = 0.0
        total_error_type_loss = 0.0
        total_correct = 0
        total_nodes = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Compute loss
            losses = self.compute_loss(outputs, batch)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += losses['total'].item()
            total_node_loss += losses['node'].item()
            total_origin_loss += losses['origin'].item()
            total_error_type_loss += losses['error_type'].item()
            
            # Compute accuracy
            node_pred = (outputs['node_pred'] > 0.5).float()
            total_correct += (node_pred == batch.y).sum().item()
            total_nodes += batch.y.size(0)
        
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'node_loss': total_node_loss / num_batches,
            'origin_loss': total_origin_loss / num_batches,
            'error_type_loss': total_error_type_loss / num_batches,
            'accuracy': total_correct / total_nodes,
        }
    
    @torch.no_grad()
    def validate(self, loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        total_origin_correct = 0
        total_origins = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Compute loss
            losses = self.compute_loss(outputs, batch)
            total_loss += losses['total'].item()
            
            # Compute accuracy
            node_pred = (outputs['node_pred'] > 0.5).float()
            total_correct += (node_pred == batch.y).sum().item()
            total_nodes += batch.y.size(0)
            
            # Origin detection accuracy
            origin_pred = (outputs['origin_pred'] > 0.5).float()
            origin_mask = batch.y_origin == 1
            if origin_mask.sum() > 0:
                total_origin_correct += (origin_pred[origin_mask] == batch.y_origin[origin_mask]).sum().item()
                total_origins += origin_mask.sum().item()
        
        num_batches = len(loader)
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_nodes if total_nodes > 0 else 0,
            'origin_accuracy': total_origin_correct / total_origins if total_origins > 0 else 0,
        }
    
    def train(self, use_wandb: bool = False) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            use_wandb: Whether to log to Weights & Biases
            
        Returns:
            Training history
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.max_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Log to wandb
            if use_wandb:
                log_metrics({
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/origin_accuracy': val_metrics['origin_accuracy'],
                    'lr': self.optimizer.param_groups[0]['lr'],
                }, step=epoch)
            
            # Print progress
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Origin Acc: {val_metrics['origin_accuracy']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                
                # Save best model
                if self.checkpoint_dir:
                    self.save_checkpoint('best_model.pth')
            else:
                self.epochs_without_improvement += 1
                
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        # Test on best model
        if self.checkpoint_dir:
            self.load_checkpoint('best_model.pth')
        
        test_metrics = self.validate(self.test_loader)
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Origin Accuracy: {test_metrics['origin_accuracy']:.4f}")
        
        return {
            'history': self.history,
            'test_metrics': test_metrics,
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            print(f"Checkpoint {checkpoint_path} not found")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']


