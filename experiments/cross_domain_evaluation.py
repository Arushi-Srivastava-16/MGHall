"""
Cross-Domain Evaluation Script.

Tests each domain-specific GNN on all other domains to measure:
- Transfer learning capability
- Domain generalization
- Model robustness
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


def evaluate_cross_domain(model, test_loader, device="cpu"):
    """Evaluate model on a different domain's data."""
    model.eval()
    model.to(device)
    
    all_node_preds = []
    all_node_labels = []
    all_origin_preds = []
    all_origin_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            
            outputs = model(batch.x, batch.edge_index, 
                          confidence=getattr(batch, 'confidence', None), 
                          batch=batch.batch)
            
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
        "node_precision": float(node_p),
        "node_recall": float(node_r),
        "node_f1": float(node_f1),
        "origin_accuracy": float(origin_acc),
        "origin_precision": float(origin_p),
        "origin_recall": float(origin_r),
        "origin_f1": float(origin_f1),
    }


def main():
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    
    print("="*80)
    print("CROSS-DOMAIN EVALUATION")
    print("="*80)
    print("\nTesting each model on all domains to measure transfer learning capability.")
    
    project_root = Path(__file__).parent.parent
    
    # Define models and data
    models_config = {
        "Math": project_root / "models/checkpoints/test_run/best_model.pth",
        "Code": project_root / "models/checkpoints/code_improved_run/best_model.pth",
    }
    
    data_config = {
        "Math": {
            "path": project_root / "data/processed/test_splits",
            "cache": project_root / "data/graphs/test_cache",
        },
        "Code": {
            "path": project_root / "data/processed/code_test_splits",
            "cache": project_root / "data/graphs/code_cache",
        },
    }
    
    feature_extractor = FeatureExtractor(embedding_dim=384)
    
    # Load all test loaders
    print("\nLoading test data for all domains...")
    test_loaders = {}
    for domain, config in data_config.items():
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
            print(f"  ✓ {domain}: {len(test_loader)} batches")
        except Exception as e:
            print(f"  ✗ {domain}: {e}")
    
    # Load all models
    print("\nLoading models...")
    models = {}
    for domain, checkpoint_path in models_config.items():
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
            models[domain] = model
            print(f"  ✓ {domain}")
        except Exception as e:
            print(f"  ✗ {domain}: {e}")
    
    # Cross-domain evaluation matrix
    print("\n" + "="*80)
    print("CROSS-DOMAIN EVALUATION MATRIX")
    print("="*80)
    print("\nRows = Models, Columns = Test Data")
    
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        for data_name, test_loader in test_loaders.items():
            print(f"\nEvaluating {model_name} model on {data_name} data...")
            metrics = evaluate_cross_domain(model, test_loader)
            results[model_name][data_name] = metrics
            
            print(f"  Node F1: {metrics['node_f1']*100:.2f}%")
            print(f"  Origin F1: {metrics['origin_f1']*100:.2f}%")
    
    # Print results table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\n📊 NODE CLASSIFICATION F1 SCORES:")
    print(f"{'Model':<15}", end="")
    for data_name in test_loaders.keys():
        print(f"{data_name:<15}", end="")
    print()
    print("-" * 80)
    
    for model_name in models.keys():
        print(f"{model_name:<15}", end="")
        for data_name in test_loaders.keys():
            if data_name in results[model_name]:
                f1 = results[model_name][data_name]['node_f1'] * 100
                # Highlight in-domain performance
                if model_name == data_name:
                    print(f"[{f1:>5.2f}%]      ", end="")
                else:
                    print(f" {f1:>5.2f}%       ", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()
    
    print("\n🔍 ORIGIN DETECTION F1 SCORES:")
    print(f"{'Model':<15}", end="")
    for data_name in test_loaders.keys():
        print(f"{data_name:<15}", end="")
    print()
    print("-" * 80)
    
    for model_name in models.keys():
        print(f"{model_name:<15}", end="")
        for data_name in test_loaders.keys():
            if data_name in results[model_name]:
                f1 = results[model_name][data_name]['origin_f1'] * 100
                if model_name == data_name:
                    print(f"[{f1:>5.2f}%]      ", end="")
                else:
                    print(f" {f1:>5.2f}%       ", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()
    
    # Compute transfer learning metrics
    print("\n" + "="*80)
    print("TRANSFER LEARNING ANALYSIS")
    print("="*80)
    
    for model_name in models.keys():
        print(f"\n{model_name} Model:")
        
        in_domain_f1 = results[model_name].get(model_name, {}).get('node_f1', 0) * 100
        
        cross_domain_f1s = []
        for data_name in test_loaders.keys():
            if data_name != model_name and data_name in results[model_name]:
                cross_domain_f1s.append(results[model_name][data_name]['node_f1'] * 100)
        
        if cross_domain_f1s:
            avg_cross_f1 = np.mean(cross_domain_f1s)
            transfer_gap = in_domain_f1 - avg_cross_f1
            
            print(f"  In-domain F1: {in_domain_f1:.2f}%")
            print(f"  Avg cross-domain F1: {avg_cross_f1:.2f}%")
            print(f"  Transfer gap: {transfer_gap:.2f}%")
            
            if transfer_gap < 10:
                print(f"  Assessment: ✅ Good generalization")
            elif transfer_gap < 20:
                print(f"  Assessment: ⚠️  Moderate domain-specific")
            else:
                print(f"  Assessment: ❌ Highly domain-specific")
    
    # Save results
    output_path = project_root / "experiments" / "cross_domain_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*80)
    print("CROSS-DOMAIN EVALUATION COMPLETE ✅")
    print("="*80)
    print("\nKey Insights:")
    print("  - [XX.XX%] = In-domain performance (model trained on this data)")
    print("  -  XX.XX%  = Cross-domain performance (transfer learning)")
    print("  - Transfer gap < 10% indicates good generalization")


if __name__ == "__main__":
    main()

