"""
Training script for Code GNN (HumanEval) - Test Run with 1000 samples.

This script trains on HumanEval data with limited samples for testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from tqdm import tqdm


def create_code_splits():
    """Create test splits for HumanEval with 1000 samples."""
    # First, convert HumanEval if not already done
    from src.data_processing.humaneval_converter import convert_humaneval_file
    
    project_root = Path(__file__).parent.parent
    raw_file = project_root / "data/raw/human-eval/data/HumanEval.jsonl.gz"
    processed_file = project_root / "data/processed/humaneval_full.jsonl"
    
    # Convert if doesn't exist
    if not processed_file.exists():
        print("Converting HumanEval dataset...")
        stats = convert_humaneval_file(
            input_path=raw_file,
            output_path=processed_file,
            augment_to=3000,  # Augment to 3K samples
        )
        print(f"Converted {stats['converted']} samples")
    
    # Create test splits
    test_dir = project_root / "data" / "processed" / "code_test_splits"
    test_dir.mkdir(exist_ok=True)
    
    # Load and subsample
    def subsample_file(input_path, output_path, max_samples):
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i >= max_samples:
                    break
                f_out.write(line)
        return i + 1
    
    # Create test splits (700/150/150 = 1000 total)
    train_count = subsample_file(processed_file, test_dir / "train.jsonl", 700)
    val_count = subsample_file(processed_file, test_dir / "val.jsonl", 150)
    test_count = subsample_file(processed_file, test_dir / "test.jsonl", 150)
    
    print(f"Created code test splits: {train_count} train, {val_count} val, {test_count} test")
    return test_dir


def main():
    print("="*60)
    print("CODE GNN TRAINING (HumanEval - 1000 samples)")
    print("="*60)
    
    # Create test splits
    print("\nPreparing HumanEval data...")
    test_dir = create_code_splits()
    
    # Import after creating splits
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    from src.training.trainer import MultiTaskTrainer
    
    # Paths
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints" / "code_test_run"
    
    # Config for code domain
    config = {
        "domain": "code",
        "model": "ConfidenceGatedGAT",
        "input_dim": 384 + 5 + 6,  # text (384) + topology (5) + task (6)
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 16,
        "max_epochs": 20,
        "patience": 5,
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
    
    # Create trainer
    print("\nInitializing trainer...")
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
    print("\n" + "="*60)
    print("STARTING CODE GNN TRAINING...")
    print("="*60)
    
    try:
        results = trainer.train(use_wandb=False)
        
        print("\n" + "="*60)
        print("✅ CODE GNN TRAINING COMPLETE!")
        print("="*60)
        print(f"\nTest Results:")
        print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"  Origin Detection Accuracy: {results['test_metrics']['origin_accuracy']:.4f}")
        print(f"  Test Loss: {results['test_metrics']['loss']:.4f}")
        
        # Save results
        results_path = checkpoint_dir / "code_training_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "domain": "code",
                "config": config,
                "test_metrics": results["test_metrics"],
                "final_train_loss": results["history"]["train_loss"][-1],
                "final_val_loss": results["history"]["val_loss"][-1],
                "num_epochs": len(results["history"]["train_loss"]),
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        print("\n" + "="*60)
        print("CODE GNN PIPELINE VERIFIED! ✅")
        print("="*60)
        print(f"\nTarget Metrics (for reference):")
        print(f"  Origin Detection: >75%")
        print(f"  Node Classification F1: >70%")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

