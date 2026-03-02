"""
Ablation Studies.

This script runs ablation experiments to analyze the contribution of different
components: graph structure, confidence gating, architecture choices, and features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from src.data_processing.graph_dataloader import create_dataloaders
from src.graph_construction.feature_extractor import FeatureExtractor
from models.gnn_architectures.gat_model import ConfidenceGatedGAT, SimpleGCN
from src.training.trainer import MultiTaskTrainer
from src.evaluation.evaluator import ModelEvaluator


class SequentialBaseline(torch.nn.Module):
    """
    Sequential baseline model (no graph structure).
    Treats reasoning chains as sequences using LSTM.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        
        self.origin_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, edge_index=None, batch=None):
        # Note: This is a simplified version - would need proper batching
        lstm_out, _ = self.lstm(x.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        
        return {
            "node_pred": torch.sigmoid(self.node_classifier(lstm_out)).squeeze(-1),
            "origin_pred": torch.sigmoid(self.origin_classifier(lstm_out)).squeeze(-1),
            "error_type_pred": torch.zeros(lstm_out.size(0), 4),  # Placeholder
            "node_embeddings": lstm_out,
        }


def run_ablation_experiment(
    experiment_name: str,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    config: dict,
    checkpoint_dir: Path,
) -> dict:
    """Run a single ablation experiment."""
    
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"{'='*60}")
    
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
        max_epochs=config.get("max_epochs", 50),  # Reduced for ablations
        patience=config.get("patience", 10),
        checkpoint_dir=checkpoint_dir,
    )
    
    results = trainer.train(use_wandb=False)
    
    return {
        "experiment": experiment_name,
        "test_accuracy": results["test_metrics"]["accuracy"],
        "test_origin_accuracy": results["test_metrics"]["origin_accuracy"],
        "test_loss": results["test_metrics"]["loss"],
        "config": config,
    }


def main():
    # Setup
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "splits"
    results_dir = Path(__file__).parent / "results" / "ablations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / "train_small.jsonl"
    val_path = data_dir / "val_small.jsonl"
    test_path = data_dir / "test_small.jsonl"
    
    if not train_path.exists():
        print("Error: Data not found. Please run conversion and splitting first.")
        return
    
    # Base config
    base_config = {
        "input_dim": 384 + 5 + 6,
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 16,
        "max_epochs": 1,
        "patience": 10,
    }
    
    # Create data loaders (reuse for all experiments)
    print("Loading data...")
    feature_extractor = FeatureExtractor()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path, val_path, test_path,
        batch_size=base_config["batch_size"],
        feature_extractor=feature_extractor,
    )
    
    all_results = []
    
    # Experiment 1: Full Model (Baseline)
    print("\n" + "="*60)
    print("EXPERIMENT 1: Full GAT with Confidence Gating")
    print("="*60)
    
    model1 = ConfidenceGatedGAT(
        input_dim=base_config["input_dim"],
        hidden_dim=base_config["hidden_dim"],
        num_layers=base_config["num_layers"],
        num_heads=base_config["num_heads"],
        dropout=base_config["dropout"],
        use_confidence_gating=True,
    )
    
    result1 = run_ablation_experiment(
        "Full_GAT_with_Confidence",
        model1,
        train_loader,
        val_loader,
        test_loader,
        base_config,
        results_dir / "full_gat",
    )
    all_results.append(result1)
    
    # Experiment 2: GAT without Confidence Gating
    print("\n" + "="*60)
    print("EXPERIMENT 2: GAT without Confidence Gating")
    print("="*60)
    
    model2 = ConfidenceGatedGAT(
        input_dim=base_config["input_dim"],
        hidden_dim=base_config["hidden_dim"],
        num_layers=base_config["num_layers"],
        num_heads=base_config["num_heads"],
        dropout=base_config["dropout"],
        use_confidence_gating=False,
    )
    
    result2 = run_ablation_experiment(
        "GAT_without_Confidence",
        model2,
        train_loader,
        val_loader,
        test_loader,
        base_config,
        results_dir / "gat_no_confidence",
    )
    all_results.append(result2)
    
    # Experiment 3: Simple GCN
    print("\n" + "="*60)
    print("EXPERIMENT 3: Simple GCN (No Attention)")
    print("="*60)
    
    model3 = SimpleGCN(
        input_dim=base_config["input_dim"],
        hidden_dim=base_config["hidden_dim"],
        num_layers=base_config["num_layers"],
        dropout=base_config["dropout"],
    )
    
    result3 = run_ablation_experiment(
        "Simple_GCN",
        model3,
        train_loader,
        val_loader,
        test_loader,
        base_config,
        results_dir / "simple_gcn",
    )
    all_results.append(result3)
    
    # Experiment 4: Sequential Baseline (No Graph Structure)
    print("\n" + "="*60)
    print("EXPERIMENT 4: Sequential LSTM Baseline")
    print("="*60)
    
    model4 = SequentialBaseline(
        input_dim=base_config["input_dim"],
        hidden_dim=base_config["hidden_dim"],
        num_layers=2,
        dropout=base_config["dropout"],
    )
    
    result4 = run_ablation_experiment(
        "Sequential_LSTM",
        model4,
        train_loader,
        val_loader,
        test_loader,
        base_config,
        results_dir / "sequential_lstm",
    )
    all_results.append(result4)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['experiment']}:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Origin Accuracy: {result['test_origin_accuracy']:.4f}")
        print(f"  Test Loss: {result['test_loss']:.4f}")
    
    # Save results
    results_file = results_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Create comparison plot
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    experiments = [r['experiment'] for r in all_results]
    node_accs = [r['test_accuracy'] for r in all_results]
    origin_accs = [r['test_origin_accuracy'] for r in all_results]
    
    ax1.barh(experiments, node_accs)
    ax1.set_xlabel('Node Classification Accuracy')
    ax1.set_title('Node Classification Performance')
    ax1.grid(True, alpha=0.3)
    
    ax2.barh(experiments, origin_accs)
    ax2.set_xlabel('Origin Detection Accuracy')
    ax2.set_title('Origin Detection Performance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "ablation_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {results_dir / 'ablation_comparison.png'}")


if __name__ == "__main__":
    main()


