"""
Training script for Code domain GNN (HumanEval).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data_processing.graph_dataloader import create_dataloaders
from src.graph_construction.feature_extractor import FeatureExtractor
from models.gnn_architectures.gat_model import ConfidenceGatedGAT
from src.training.trainer import MultiTaskTrainer
from src.training.wandb_config import init_wandb


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "splits"
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints" / "code"
    
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    
    # Check if data exists
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Please run data conversion and splitting first.")
        return
    
    # Config (similar to math but potentially different hyperparameters)
    config = {
        "domain": "code",
        "model": "ConfidenceGatedGAT",
        "input_dim": 384 + 5 + 6,
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.15,  # Slightly higher dropout for code
        "learning_rate": 5e-4,  # Lower LR for code
        "weight_decay": 1e-5,
        "batch_size": 16,
        "max_epochs": 100,
        "patience": 15,  # More patience for code
    }
    
    # Initialize W&B
    init_wandb(
        project_name="chg-framework",
        experiment_name="code-gnn-training",
        config=config,
        tags=["code", "humaneval", "gat"],
    )
    
    print("Creating data loaders...")
    feature_extractor = FeatureExtractor(embedding_dim=384)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=config["batch_size"],
        feature_extractor=feature_extractor,
    )
    
    # Create model
    model = ConfidenceGatedGAT(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        checkpoint_dir=checkpoint_dir,
    )
    
    # Train
    print("\nStarting training...")
    results = trainer.train(use_wandb=True)
    
    print(f"\nTest Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Origin Detection Accuracy: {results['test_metrics']['origin_accuracy']:.4f}")


if __name__ == "__main__":
    main()

