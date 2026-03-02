"""
Ensemble Model for Multi-Domain Hallucination Detection.

Combines predictions from domain-specific GNNs using:
1. Weighted voting based on confidence
2. Domain-aware routing
3. Ensemble averaging
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class EnsembleGNN(nn.Module):
    """
    Ensemble of domain-specific GNNs.
    
    Combines predictions using:
    - Weighted averaging based on confidence
    - Domain-aware routing (if domain is known)
    - Max confidence selection
    """
    
    def __init__(self, models: Dict[str, nn.Module], strategy: str = "weighted_average"):
        """
        Initialize ensemble.
        
        Args:
            models: Dictionary of {domain_name: model}
            strategy: Ensemble strategy ('weighted_average', 'max_confidence', 'domain_routing')
        """
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.strategy = strategy
        self.domain_names = list(models.keys())
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch vector
            domain: Optional domain hint for routing
            
        Returns:
            Dictionary with ensemble predictions
        """
        # Get predictions from all models
        all_outputs = {}
        for domain_name, model in self.models.items():
            with torch.no_grad():
                outputs = model(x, edge_index, confidence=None, batch=batch)
                all_outputs[domain_name] = outputs
        
        # Ensemble strategy
        if self.strategy == "domain_routing" and domain in self.models:
            # Use domain-specific model if domain is known
            return all_outputs[domain]
        
        elif self.strategy == "weighted_average":
            # Weighted average based on confidence
            node_preds = []
            origin_preds = []
            weights = []
            
            for domain_name, outputs in all_outputs.items():
                # Use prediction confidence as weight
                node_conf = torch.abs(outputs['node_pred'] - 0.5) * 2  # Convert to [0, 1]
                origin_conf = torch.abs(outputs['origin_pred'] - 0.5) * 2
                
                weight = (node_conf + origin_conf) / 2
                weights.append(weight)
                node_preds.append(outputs['node_pred'] * weight)
                origin_preds.append(outputs['origin_pred'] * weight)
            
            # Normalize weights
            total_weight = torch.stack(weights).sum(dim=0)
            total_weight = torch.clamp(total_weight, min=1e-8)  # Avoid division by zero
            
            ensemble_node_pred = torch.stack(node_preds).sum(dim=0) / total_weight
            ensemble_origin_pred = torch.stack(origin_preds).sum(dim=0) / total_weight
            
            return {
                'node_pred': ensemble_node_pred,
                'origin_pred': ensemble_origin_pred,
                'individual_predictions': all_outputs,
            }
        
        elif self.strategy == "max_confidence":
            # Select prediction with highest confidence
            node_preds = []
            origin_preds = []
            confidences = []
            
            for outputs in all_outputs.values():
                node_preds.append(outputs['node_pred'])
                origin_preds.append(outputs['origin_pred'])
                
                # Compute confidence as distance from 0.5
                conf = (torch.abs(outputs['node_pred'] - 0.5) + 
                       torch.abs(outputs['origin_pred'] - 0.5)) / 2
                confidences.append(conf)
            
            # Stack and find max confidence indices
            node_preds = torch.stack(node_preds)
            origin_preds = torch.stack(origin_preds)
            confidences = torch.stack(confidences)
            
            max_conf_idx = confidences.argmax(dim=0)
            
            # Gather predictions from most confident model
            ensemble_node_pred = node_preds.gather(0, max_conf_idx.unsqueeze(0)).squeeze(0)
            ensemble_origin_pred = origin_preds.gather(0, max_conf_idx.unsqueeze(0)).squeeze(0)
            
            return {
                'node_pred': ensemble_node_pred,
                'origin_pred': ensemble_origin_pred,
                'selected_models': max_conf_idx,
                'individual_predictions': all_outputs,
            }
        
        else:
            # Simple averaging
            node_preds = torch.stack([out['node_pred'] for out in all_outputs.values()])
            origin_preds = torch.stack([out['origin_pred'] for out in all_outputs.values()])
            
            return {
                'node_pred': node_preds.mean(dim=0),
                'origin_pred': origin_preds.mean(dim=0),
                'individual_predictions': all_outputs,
            }


def evaluate_ensemble(ensemble, test_loader, device="cpu"):
    """Evaluate ensemble model."""
    ensemble.eval()
    ensemble.to(device)
    
    all_node_preds = []
    all_node_labels = []
    all_origin_preds = []
    all_origin_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            outputs = ensemble(batch.x, batch.edge_index, batch=batch.batch)
            
            node_preds = (outputs['node_pred'] > 0.5).long()
            origin_preds = (outputs['origin_pred'] > 0.5).long()
            
            all_node_preds.extend(node_preds.cpu().numpy())
            all_node_labels.extend(batch.y.cpu().numpy())
            all_origin_preds.extend(origin_preds.cpu().numpy())
            all_origin_labels.extend(batch.y_origin.cpu().numpy())
    
    # Convert to numpy
    all_node_preds = np.array(all_node_preds).flatten()
    all_node_labels = np.array(all_node_labels).flatten()
    all_origin_preds = np.array(all_origin_preds).flatten()
    all_origin_labels = np.array(all_origin_labels).flatten()
    
    # Compute metrics
    node_acc = accuracy_score(all_node_labels, all_node_preds)
    origin_acc = accuracy_score(all_origin_labels, all_origin_preds)
    
    node_p, node_r, node_f1, _ = precision_recall_fscore_support(
        all_node_labels, all_node_preds, average='binary', zero_division=0
    )
    
    origin_p, origin_r, origin_f1, _ = precision_recall_fscore_support(
        all_origin_labels, all_origin_preds, average='binary', zero_division=0
    )
    
    return {
        "node_accuracy": float(node_acc),
        "node_f1": float(node_f1),
        "origin_accuracy": float(origin_acc),
        "origin_f1": float(origin_f1),
    }


def main():
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    
    print("="*80)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Load individual models
    print("\nLoading domain-specific models...")
    models = {}
    
    model_configs = {
        "Math": project_root / "models/checkpoints/test_run/best_model.pth",
        "Code": project_root / "models/checkpoints/code_improved_run/best_model.pth",
    }
    
    for domain, checkpoint_path in model_configs.items():
        if not checkpoint_path.exists():
            print(f"  ✗ {domain}: Checkpoint not found")
            continue
        
        model = ConfidenceGatedGAT(
            input_dim=384 + 5 + 6,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            use_confidence_gating=True,
        )
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            models[domain] = model
            print(f"  ✓ {domain}")
        except Exception as e:
            print(f"  ✗ {domain}: {e}")
    
    if len(models) < 2:
        print("\n❌ Need at least 2 models for ensemble. Exiting.")
        return
    
    # Create ensemble models with different strategies
    print("\nCreating ensemble models...")
    ensembles = {
        "Weighted Average": EnsembleGNN(models, strategy="weighted_average"),
        "Max Confidence": EnsembleGNN(models, strategy="max_confidence"),
        "Simple Average": EnsembleGNN(models, strategy="average"),
    }
    
    for name in ensembles.keys():
        print(f"  ✓ {name}")
    
    # Load test data from all domains
    print("\nLoading test data...")
    feature_extractor = FeatureExtractor(embedding_dim=384)
    
    test_datasets = {
        "Math": {
            "path": project_root / "data/processed/test_splits",
            "cache": project_root / "data/graphs/test_cache",
        },
        "Code": {
            "path": project_root / "data/processed/code_test_splits",
            "cache": project_root / "data/graphs/code_cache",
        },
    }
    
    test_loaders = {}
    for domain, config in test_datasets.items():
        try:
            _, _, test_loader = create_dataloaders(
                train_path=config["path"] / "train.jsonl",
                val_path=config["path"] / "val.jsonl",
                test_path=config["path"] / "test.jsonl",
                batch_size=16,
                num_workers=0,
                feature_extractor=feature_extractor,
                cache_dir=config["cache"],
            )
            test_loaders[domain] = test_loader
            print(f"  ✓ {domain}")
        except Exception as e:
            print(f"  ✗ {domain}: {e}")
    
    # Evaluate each ensemble strategy
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION RESULTS")
    print("="*80)
    
    results = {}
    
    for ensemble_name, ensemble in ensembles.items():
        print(f"\n{ensemble_name} Ensemble:")
        results[ensemble_name] = {}
        
        for domain_name, test_loader in test_loaders.items():
            print(f"  Evaluating on {domain_name} data...")
            metrics = evaluate_ensemble(ensemble, test_loader)
            results[ensemble_name][domain_name] = metrics
            
            print(f"    Node F1: {metrics['node_f1']*100:.2f}%")
            print(f"    Origin F1: {metrics['origin_f1']*100:.2f}%")
    
    # Compare with individual models
    print("\n" + "="*80)
    print("COMPARISON: ENSEMBLE VS INDIVIDUAL MODELS")
    print("="*80)
    
    print(f"\n{'Strategy':<20} {'Test Domain':<15} {'Node F1':<12} {'Origin F1':<12}")
    print("-" * 80)
    
    # Print individual model results first
    for model_name in models.keys():
        for data_name in test_loaders.keys():
            if model_name == data_name:
                # Load individual model results from previous evaluation
                result_file = project_root / "models/checkpoints" / f"{model_name.lower()}_improved_run" / f"{model_name.lower()}-improved_detailed_metrics.json"
                if model_name == "Math":
                    result_file = project_root / "models/checkpoints/test_run/math_detailed_metrics.json"
                
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                        node_f1 = data["node_classification"]["f1"] * 100
                        origin_f1 = data["origin_detection"]["f1"] * 100
                        print(f"{f'{model_name} (single)':<20} {data_name:<15} {node_f1:>10.2f}%  {origin_f1:>10.2f}%")
    
    print()
    
    # Print ensemble results
    for ensemble_name, domain_results in results.items():
        for domain_name, metrics in domain_results.items():
            node_f1 = metrics['node_f1'] * 100
            origin_f1 = metrics['origin_f1'] * 100
            print(f"{ensemble_name:<20} {domain_name:<15} {node_f1:>10.2f}%  {origin_f1:>10.2f}%")
    
    # Save results
    output_path = project_root / "experiments" / "ensemble_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION COMPLETE ✅")
    print("="*80)
    print("\nKey Findings:")
    print("  - Weighted Average: Balances predictions based on confidence")
    print("  - Max Confidence: Selects most confident model per prediction")
    print("  - Simple Average: Equal weight to all models")
    print("\nBest for production: Strategy with highest avg F1 across domains")


if __name__ == "__main__":
    main()

