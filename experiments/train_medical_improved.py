"""
Train Medical GNN on improved synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json

from src.data_processing.graph_dataloader import create_dataloaders
from src.graph_construction.feature_extractor import FeatureExtractor
from models.gnn_architectures.gat_model import ConfidenceGatedGAT
from src.training.trainer import MultiTaskTrainer


def main():
    print("="*70)
    print("TRAINING MEDICAL GNN - IMPROVED DATA")
    print("="*70)
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data/processed/medical_improved_splits"
    checkpoint_dir = Path(__file__).parent.parent / "models/checkpoints/medical_improved_run"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        "domain": "medical",
        "model": "ConfidenceGatedGAT",
        "input_dim": 384 + 5 + 6,  # text + topology + task
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 16,
        "max_epochs": 25,
        "patience": 7,
        "origin_loss_weight": 5.0,  # Higher for medical domain
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    print(f"\n📂 Loading data from {data_dir}...")
    feature_extractor = FeatureExtractor(embedding_dim=384)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=data_dir / "train.jsonl",
        val_path=data_dir / "val.jsonl",
        test_path=data_dir / "test.jsonl",
        batch_size=config["batch_size"],
        num_workers=0,
        feature_extractor=feature_extractor,
        cache_dir=Path(__file__).parent.parent / "data/graphs/medical_improved_cache",
    )
    
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")
    
    # Create model
    print("\n🧠 Initializing ConfidenceGatedGAT...")
    model = ConfidenceGatedGAT(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        use_confidence_gating=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create trainer
    print("\n🚀 Initializing trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        origin_loss_weight=config["origin_loss_weight"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        checkpoint_dir=checkpoint_dir,
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING...")
    print("="*70)
    
    results = trainer.train(use_wandb=False)
    
    print("\n" + "="*70)
    print("✅ MEDICAL MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTest Results:")
    print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"  Origin Detection Accuracy: {results['test_metrics']['origin_accuracy']:.4f}")
    print(f"  Test Loss: {results['test_metrics']['loss']:.4f}")
    
    # Save results
    results_path = checkpoint_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config,
            "test_metrics": results["test_metrics"],
            "final_train_loss": results["history"]["train_loss"][-1],
            "final_val_loss": results["history"]["val_loss"][-1],
            "num_epochs": len(results["history"]["train_loss"]),
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Model saved to {checkpoint_dir}/best_model.pth")
    
    return results


if __name__ == "__main__":
    results = main()
