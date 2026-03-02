"""
Improved training script for Code GNN with better origin detection.

Addresses class imbalance issue (only 6.5% origin nodes) with:
- Higher origin loss weight
- Better focal loss parameters
- Class-weighted loss
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
from tqdm import tqdm


class ImprovedFocalLoss(nn.Module):
    """Improved Focal Loss with better handling for extreme class imbalance."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 3.0, pos_weight: float = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply focal weighting
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        # Apply pos_weight if provided
        if self.pos_weight is not None:
            weight = self.pos_weight * targets + (1 - targets)
            focal_loss = focal_loss * weight
        
        return focal_loss.mean()


def create_code_splits():
    """Create test splits for HumanEval with 1000 samples."""
    from src.data_processing.humaneval_converter import convert_humaneval_file
    
    project_root = Path(__file__).parent.parent
    raw_file = project_root / "data/raw/human-eval/data/HumanEval.jsonl.gz"
    processed_file = project_root / "data/processed/humaneval_full.jsonl"
    
    if not processed_file.exists():
        print("Converting HumanEval dataset...")
        stats = convert_humaneval_file(
            input_path=raw_file,
            output_path=processed_file,
            augment_to=3000,
        )
        print(f"Converted {stats['converted']} samples")
    
    test_dir = project_root / "data" / "processed" / "code_test_splits"
    test_dir.mkdir(exist_ok=True)
    
    def subsample_file(input_path, output_path, max_samples):
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i >= max_samples:
                    break
                f_out.write(line)
        return i + 1
    
    train_count = subsample_file(processed_file, test_dir / "train.jsonl", 700)
    val_count = subsample_file(processed_file, test_dir / "val.jsonl", 150)
    test_count = subsample_file(processed_file, test_dir / "test.jsonl", 150)
    
    print(f"Created code test splits: {train_count} train, {val_count} val, {test_count} test")
    return test_dir


def compute_class_weights(train_loader):
    """Compute class weights for origin detection from training data."""
    total_origin = 0
    total_non_origin = 0
    
    for batch in train_loader:
        total_origin += batch.y_origin.sum().item()
        total_non_origin += (1 - batch.y_origin).sum().item()
    
    total = total_origin + total_non_origin
    if total == 0:
        return None
    
    # Weight inversely proportional to frequency
    pos_weight = total_non_origin / (total_origin + 1e-8)
    
    print(f"Origin class distribution: {total_origin} origin, {total_non_origin} non-origin")
    print(f"Computed pos_weight: {pos_weight:.2f}")
    
    return pos_weight


def main():
    print("="*70)
    print("IMPROVED CODE GNN TRAINING (Better Origin Detection)")
    print("="*70)
    
    # Create test splits
    print("\nPreparing HumanEval data...")
    test_dir = create_code_splits()
    
    # Import after creating splits
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    from src.training.trainer import MultiTaskTrainer
    
    # Paths
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints" / "code_improved_run"
    
    # Config with improved settings
    config = {
        "domain": "code",
        "model": "ConfidenceGatedGAT",
        "input_dim": 384 + 5 + 6,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 16,
        "max_epochs": 30,  # More epochs
        "patience": 8,  # More patience
        "origin_loss_weight": 10.0,  # Much higher weight for origin detection
    }
    
    print("\nCreating data loaders...")
    feature_extractor = FeatureExtractor(embedding_dim=384)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_path=test_dir / "train.jsonl",
            val_path=test_dir / "val.jsonl",
            test_path=test_dir / "test.jsonl",
            batch_size=config["batch_size"],
            num_workers=0,
            feature_extractor=feature_extractor,
            cache_dir=Path(__file__).parent.parent / "data" / "graphs" / "code_cache",
        )
        
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute class weights
    print("\nComputing class weights...")
    pos_weight = compute_class_weights(train_loader)
    
    # Create model
    print("\nInitializing Code GNN model...")
    model = ConfidenceGatedGAT(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        use_confidence_gating=True,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with improved loss
    print("\nInitializing trainer with improved loss settings...")
    
    # Patch the trainer to use improved focal loss
    from src.training.trainer import MultiTaskTrainer, FocalLoss
    
    # Create custom trainer class with improved loss
    class ImprovedCodeTrainer(MultiTaskTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Replace origin criterion with improved version
            if pos_weight:
                self.origin_criterion = ImprovedFocalLoss(
                    alpha=0.75,  # Higher alpha for positive class
                    gamma=3.0,   # Higher gamma for harder examples
                    pos_weight=pos_weight
                )
            else:
                self.origin_criterion = ImprovedFocalLoss(alpha=0.75, gamma=3.0)
    
    trainer = ImprovedCodeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        origin_loss_weight=config["origin_loss_weight"],  # Much higher weight
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        checkpoint_dir=checkpoint_dir,
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING IMPROVED CODE GNN TRAINING...")
    print("="*70)
    print(f"Key improvements:")
    print(f"  - Origin loss weight: {config['origin_loss_weight']}x (vs 2.0x default)")
    print(f"  - Improved Focal Loss: alpha=0.75, gamma=3.0")
    print(f"  - Pos weight: {pos_weight:.2f}" if pos_weight else "  - Pos weight: None")
    print(f"  - More epochs: {config['max_epochs']} (vs 20)")
    print("="*70)
    
    try:
        results = trainer.train(use_wandb=False)
        
        print("\n" + "="*70)
        print("✅ IMPROVED CODE GNN TRAINING COMPLETE!")
        print("="*70)
        print(f"\nTest Results:")
        print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"  Origin Detection Accuracy: {results['test_metrics']['origin_accuracy']:.4f}")
        print(f"  Test Loss: {results['test_metrics']['loss']:.4f}")
        
        # Save results
        results_path = checkpoint_dir / "code_improved_training_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "domain": "code",
                "config": config,
                "pos_weight": pos_weight,
                "test_metrics": results["test_metrics"],
                "final_train_loss": results["history"]["train_loss"][-1],
                "final_val_loss": results["history"]["val_loss"][-1],
                "num_epochs": len(results["history"]["train_loss"]),
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        print("\n" + "="*70)
        print("IMPROVED CODE GNN TRAINING COMPLETE! ✅")
        print("="*70)
        print(f"\nTarget Metrics:")
        print(f"  Origin Detection: >75% (Previous: 15.52%)")
        print(f"  Node Classification F1: >70% (Previous: 95.07%)")
        print(f"\nRun comprehensive evaluation to see detailed metrics:")
        print(f"  ./venv/bin/python experiments/comprehensive_evaluation.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

