"""
Training script for Medical GNN (MedHallu) - Test Run with 1000 samples.

This script trains on MedHallu data with limited samples for testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from tqdm import tqdm


def create_medical_splits():
    """Create test splits for MedHallu with 1000 samples."""
    # First, convert MedHallu if not already done
    from src.data_processing.medhallu_converter import convert_medhallu_dataset
    
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data/raw/medhallu"
    processed_file = project_root / "data/processed/medhallu_full.jsonl"
    
    # Convert if doesn't exist
    if not processed_file.exists():
        print("Converting MedHallu dataset...")
        try:
            stats = convert_medhallu_dataset(
                input_dir=raw_dir,
                output_path=processed_file,
                split="pqa_labeled",
                max_samples=None,  # Convert all
                use_llm=False,
            )
            print(f"Converted {stats['converted']} samples")
        except Exception as e:
            print(f"Error converting MedHallu: {e}")
            print("Using fallback: creating synthetic medical data for testing...")
            # Create some synthetic data if conversion fails
            create_synthetic_medical_data(processed_file)
    
    # Create test splits
    test_dir = project_root / "data" / "processed" / "medical_test_splits"
    test_dir.mkdir(exist_ok=True)
    
    # Load and subsample
    def subsample_file(input_path, output_path, max_samples):
        if not input_path.exists():
            return 0
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
    
    print(f"Created medical test splits: {train_count} train, {val_count} val, {test_count} test")
    return test_dir


def create_synthetic_medical_data(output_path):
    """Create synthetic medical data for testing if real data unavailable."""
    from src.data_processing.unified_schema import (
        ReasoningChain, ReasoningStep, DependencyGraph, Domain, ErrorType
    )
    
    synthetic_samples = []
    for i in range(1000):
        # Create a simple medical reasoning chain
        is_correct = i % 3 != 0  # Make every 3rd sample have an error
        error_step = 2 if not is_correct else -1
        
        steps = []
        for j in range(4):
            is_step_correct = j < error_step if error_step >= 0 else True
            is_origin = j == error_step
            
            steps.append(ReasoningStep(
                step_id=j,
                text=f"Medical reasoning step {j+1}: Analyze symptom and diagnosis",
                is_correct=is_step_correct,
                is_origin=is_origin,
                error_type=ErrorType.FACTUAL if not is_step_correct else None,
                depends_on=[j-1] if j > 0 else [],
            ))
        
        chain = ReasoningChain(
            domain=Domain.MEDICAL,
            query_id=f"medical_synthetic_{i}",
            query=f"What is the diagnosis for patient {i}?",
            ground_truth="Correct medical diagnosis",
            reasoning_steps=steps,
            dependency_graph=DependencyGraph(
                nodes=[0, 1, 2, 3],
                edges=[[0, 1], [1, 2], [2, 3]]
            ),
        )
        synthetic_samples.append(chain)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for chain in synthetic_samples:
            f.write(chain.to_json(pretty=False) + "\n")
    
    print(f"Created {len(synthetic_samples)} synthetic medical samples")


def main():
    print("="*60)
    print("MEDICAL GNN TRAINING (MedHallu - 1000 samples)")
    print("="*60)
    
    # Create test splits
    print("\nPreparing MedHallu data...")
    test_dir = create_medical_splits()
    
    # Import after creating splits
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    from src.training.trainer import MultiTaskTrainer
    
    # Paths
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints" / "medical_test_run"
    
    # Config for medical domain
    config = {
        "domain": "medical",
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
            cache_dir=Path(__file__).parent.parent / "data" / "graphs" / "medical_cache",
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
    print("\nInitializing Medical GNN model...")
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
    print("STARTING MEDICAL GNN TRAINING...")
    print("="*60)
    
    try:
        results = trainer.train(use_wandb=False)
        
        print("\n" + "="*60)
        print("✅ MEDICAL GNN TRAINING COMPLETE!")
        print("="*60)
        print(f"\nTest Results:")
        print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"  Origin Detection Accuracy: {results['test_metrics']['origin_accuracy']:.4f}")
        print(f"  Test Loss: {results['test_metrics']['loss']:.4f}")
        
        # Save results
        results_path = checkpoint_dir / "medical_training_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "domain": "medical",
                "config": config,
                "test_metrics": results["test_metrics"],
                "final_train_loss": results["history"]["train_loss"][-1],
                "final_val_loss": results["history"]["val_loss"][-1],
                "num_epochs": len(results["history"]["train_loss"]),
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        print("\n" + "="*60)
        print("MEDICAL GNN PIPELINE VERIFIED! ✅")
        print("="*60)
        print(f"\nTarget Metrics (for reference):")
        print(f"  Origin Detection: >70%")
        print(f"  Node Classification F1: >65%")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

