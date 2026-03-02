"""
Training script for Math domain GNN (PRM800K).
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
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints" / "math"
    
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    
    # Check if data exists
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Please run data conversion and splitting first:")
        print("  1. python src/data_processing/prm800k_converter.py")
        print("  2. python src/data_processing/splitter.py")
        return
    
    # Config
    config = {
        "domain": "math",
        "model": "ConfidenceGatedGAT",
        "input_dim": 384 + 5 + 6,  # text (384) + topology (5) + task (6)
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 16,
        "max_epochs": 100,
        "patience": 10,
    }
    
    # Initialize W&B
    init_wandb(
        project_name="chg-framework",
        experiment_name="math-gnn-training",
        config=config,
        tags=["math", "prm800k", "gat"],
    )
    
    print("Creating data loaders...")
    feature_extractor = FeatureExtractor(embedding_dim=384)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=config["batch_size"],
        num_workers=0,
        feature_extractor=feature_extractor,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("Initializing model...")
    model = ConfidenceGatedGAT(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        use_confidence_gating=True,
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
    
    # Save final results
    import json
    results_path = checkpoint_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config,
            "test_metrics": results["test_metrics"],
            "final_train_loss": results["history"]["train_loss"][-1],
            "final_val_loss": results["history"]["val_loss"][-1],
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()


