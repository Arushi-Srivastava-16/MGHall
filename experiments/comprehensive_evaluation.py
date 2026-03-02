"""
Comprehensive evaluation script for all trained GNN models.

Computes detailed metrics: F1, Precision, Recall, Confusion Matrices, Per-class analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm


def evaluate_model_detailed(model, test_loader, device="cpu", origin_threshold=0.5):
    """
    Evaluate model with detailed metrics.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        origin_threshold: Threshold for origin detection (lower for imbalanced classes)
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_origin_preds = []
    all_origin_labels = []
    all_origin_probs = []  # Store probabilities for threshold analysis
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            
            # Forward pass - model returns dict with 'node_pred' and 'origin_pred' (already sigmoided)
            outputs = model(batch.x, batch.edge_index, confidence=getattr(batch, 'confidence', None), batch=batch.batch)
            
            # Get predictions (already sigmoided, so just threshold)
            node_preds = (outputs['node_pred'] > 0.5).long()
            origin_preds = (outputs['origin_pred'] > origin_threshold).long()  # Use adaptive threshold
            all_origin_probs.extend(outputs['origin_pred'].cpu().numpy())
            
            # Collect predictions and labels
            all_preds.extend(node_preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_origin_preds.extend(origin_preds.cpu().numpy())
            # Use y_origin attribute (as set in crg_builder.py)
            all_origin_labels.extend(batch.y_origin.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_origin_preds = np.array(all_origin_preds).flatten()
    all_origin_labels = np.array(all_origin_labels).flatten()
    
    # Compute metrics
    metrics = {}
    
    # Node classification metrics
    metrics["node_classification"] = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    # Precision, Recall, F1 for node classification
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    metrics["node_classification"]["precision"] = float(precision) if precision is not None else 0.0
    metrics["node_classification"]["recall"] = float(recall) if recall is not None else 0.0
    metrics["node_classification"]["f1"] = float(f1) if f1 is not None else 0.0
    metrics["node_classification"]["support"] = int(support) if support is not None else len(all_labels)
    
    # Per-class metrics for node classification
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    metrics["node_classification"]["per_class"] = {
        "correct": {
            "precision": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
            "recall": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
            "f1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "support": int(support_per_class[1]) if len(support_per_class) > 1 else 0,
        },
        "incorrect": {
            "precision": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
            "recall": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
            "f1": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "support": int(support_per_class[0]) if len(support_per_class) > 0 else 0,
        },
    }
    
    # Origin detection metrics
    metrics["origin_detection"] = {
        "accuracy": accuracy_score(all_origin_labels, all_origin_preds),
        "confusion_matrix": confusion_matrix(all_origin_labels, all_origin_preds).tolist(),
    }
    
    # Precision, Recall, F1 for origin detection
    precision, recall, f1, support = precision_recall_fscore_support(
        all_origin_labels, all_origin_preds, average='binary', zero_division=0
    )
    metrics["origin_detection"]["precision"] = float(precision) if precision is not None else 0.0
    metrics["origin_detection"]["recall"] = float(recall) if recall is not None else 0.0
    metrics["origin_detection"]["f1"] = float(f1) if f1 is not None else 0.0
    metrics["origin_detection"]["support"] = int(support) if support is not None else len(all_origin_labels)
    
    # Per-class metrics for origin detection
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_origin_labels, all_origin_preds, average=None, zero_division=0)
    
    metrics["origin_detection"]["per_class"] = {
        "is_origin": {
            "precision": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
            "recall": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
            "f1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "support": int(support_per_class[1]) if len(support_per_class) > 1 else 0,
        },
        "not_origin": {
            "precision": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
            "recall": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
            "f1": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "support": int(support_per_class[0]) if len(support_per_class) > 0 else 0,
        },
    }
    
    # Class distribution
    metrics["data_statistics"] = {
        "total_nodes": len(all_labels),
        "correct_nodes": int(np.sum(all_labels)),
        "incorrect_nodes": int(np.sum(1 - all_labels)),
        "origin_nodes": int(np.sum(all_origin_labels)),
        "non_origin_nodes": int(np.sum(1 - all_origin_labels)),
        "class_balance_node": float(np.mean(all_labels)),
        "class_balance_origin": float(np.mean(all_origin_labels)),
    }
    
    return metrics


def print_detailed_metrics(domain, metrics):
    """Pretty print detailed metrics."""
    print("\n" + "="*70)
    print(f"DETAILED EVALUATION: {domain.upper()}")
    print("="*70)
    
    # Data statistics
    stats = metrics["data_statistics"]
    print(f"\n📊 DATA STATISTICS:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Correct nodes: {stats['correct_nodes']} ({stats['class_balance_node']*100:.1f}%)")
    print(f"  Incorrect nodes: {stats['incorrect_nodes']} ({(1-stats['class_balance_node'])*100:.1f}%)")
    print(f"  Origin nodes: {stats['origin_nodes']} ({stats['class_balance_origin']*100:.1f}%)")
    print(f"  Non-origin nodes: {stats['non_origin_nodes']} ({(1-stats['class_balance_origin'])*100:.1f}%)")
    
    # Node classification
    node_metrics = metrics["node_classification"]
    print(f"\n🎯 NODE CLASSIFICATION:")
    print(f"  Accuracy:  {node_metrics['accuracy']*100:6.2f}%")
    print(f"  Precision: {node_metrics['precision']*100:6.2f}%")
    print(f"  Recall:    {node_metrics['recall']*100:6.2f}%")
    print(f"  F1 Score:  {node_metrics['f1']*100:6.2f}%")
    
    print(f"\n  Per-Class Metrics:")
    for class_name, class_metrics in node_metrics["per_class"].items():
        print(f"    {class_name.capitalize()}:")
        print(f"      Precision: {class_metrics['precision']*100:6.2f}%")
        print(f"      Recall:    {class_metrics['recall']*100:6.2f}%")
        print(f"      F1 Score:  {class_metrics['f1']*100:6.2f}%")
        print(f"      Support:   {class_metrics['support']}")
    
    print(f"\n  Confusion Matrix:")
    cm = node_metrics["confusion_matrix"]
    print(f"                 Predicted")
    print(f"               Incorrect  Correct")
    # Handle different matrix shapes (1D or 2D)
    if len(cm) == 1:
        # Only one class predicted
        print(f"    Incorrect  {cm[0][0] if isinstance(cm[0], list) else cm[0]:>8}  {cm[0][1] if isinstance(cm[0], list) and len(cm[0]) > 1 else 0:>7}")
        print(f"    Correct    {0:>8}  {0:>7}")
    elif len(cm) == 2:
        print(f"    Incorrect  {cm[0][0]:>8}  {cm[0][1] if len(cm[0]) > 1 else 0:>7}")
        print(f"    Correct    {cm[1][0] if len(cm) > 1 else 0:>8}  {cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0:>7}")
    else:
        print(f"    (Matrix shape: {len(cm)}x{len(cm[0]) if cm else 0})")
    
    # Origin detection
    origin_metrics = metrics["origin_detection"]
    print(f"\n🔍 ORIGIN DETECTION:")
    print(f"  Accuracy:  {origin_metrics['accuracy']*100:6.2f}%")
    print(f"  Precision: {origin_metrics['precision']*100:6.2f}%")
    print(f"  Recall:    {origin_metrics['recall']*100:6.2f}%")
    print(f"  F1 Score:  {origin_metrics['f1']*100:6.2f}%")
    
    print(f"\n  Per-Class Metrics:")
    for class_name, class_metrics in origin_metrics["per_class"].items():
        print(f"    {class_name.replace('_', ' ').capitalize()}:")
        print(f"      Precision: {class_metrics['precision']*100:6.2f}%")
        print(f"      Recall:    {class_metrics['recall']*100:6.2f}%")
        print(f"      F1 Score:  {class_metrics['f1']*100:6.2f}%")
        print(f"      Support:   {class_metrics['support']}")
    
    print(f"\n  Confusion Matrix:")
    cm = origin_metrics["confusion_matrix"]
    print(f"                 Predicted")
    print(f"            Not Origin  Is Origin")
    # Handle different matrix shapes
    if len(cm) == 1:
        print(f"    Not Orig {cm[0][0] if isinstance(cm[0], list) else cm[0]:>9}  {cm[0][1] if isinstance(cm[0], list) and len(cm[0]) > 1 else 0:>9}")
        print(f"    Is Orig  {0:>9}  {0:>9}")
    elif len(cm) == 2:
        print(f"    Not Orig {cm[0][0]:>9}  {cm[0][1] if len(cm[0]) > 1 else 0:>9}")
        print(f"    Is Orig  {cm[1][0] if len(cm) > 1 else 0:>9}  {cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0:>9}")
    else:
        print(f"    (Matrix shape: {len(cm)}x{len(cm[0]) if cm else 0})")


def main():
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    domains = [
        {
            "name": "Math",
            "data_dir": project_root / "data/processed/test_splits",
            "checkpoint": project_root / "models/checkpoints/test_run/best_model.pth",
            "cache_dir": project_root / "data/graphs/test_cache",
        },
        {
            "name": "Code",
            "data_dir": project_root / "data/processed/code_test_splits",
            "checkpoint": project_root / "models/checkpoints/code_test_run/best_model.pth",
            "cache_dir": project_root / "data/graphs/code_cache",
        },
        {
            "name": "Code-Improved",
            "data_dir": project_root / "data/processed/code_test_splits",
            "checkpoint": project_root / "models/checkpoints/code_improved_run/best_model.pth",
            "cache_dir": project_root / "data/graphs/code_cache",
        },
        {
            "name": "Medical",
            "data_dir": project_root / "data/processed/medical_test_splits",
            "checkpoint": project_root / "models/checkpoints/medical_test_run/best_model.pth",
            "cache_dir": project_root / "data/graphs/medical_cache",
        },
    ]
    
    all_results = {}
    
    for domain_config in domains:
        domain = domain_config["name"]
        print(f"\n{'='*70}")
        print(f"Evaluating {domain} GNN...")
        print(f"{'='*70}")
        
        # Check if checkpoint exists
        if not domain_config["checkpoint"].exists():
            print(f"⚠️  Checkpoint not found: {domain_config['checkpoint']}")
            continue
        
        # Load data
        print("Loading data...")
        feature_extractor = FeatureExtractor(embedding_dim=384)
        
        try:
            _, _, test_loader = create_dataloaders(
                train_path=domain_config["data_dir"] / "train.jsonl",
                val_path=domain_config["data_dir"] / "val.jsonl",
                test_path=domain_config["data_dir"] / "test.jsonl",
                batch_size=16,
                num_workers=0,
                feature_extractor=feature_extractor,
                cache_dir=domain_config["cache_dir"],
            )
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            continue
        
        # Load model
        print("Loading model...")
        model = ConfidenceGatedGAT(
            input_dim=384 + 5 + 6,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            use_confidence_gating=True,
        )
        
        try:
            checkpoint = torch.load(domain_config["checkpoint"], map_location="cpu")
            # Handle both formats: dict with 'model_state_dict' or direct state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Assume it's the state_dict directly
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Use lower threshold for Code domain (severe class imbalance)
        if "Code-Improved" in domain:
            origin_threshold = 0.25  # Lower threshold for improved model
        elif domain == "Code":
            origin_threshold = 0.3  # Lower threshold for original model
        else:
            origin_threshold = 0.5  # Standard threshold
        
        # Evaluate
        print(f"Running evaluation (origin threshold: {origin_threshold})...")
        metrics = evaluate_model_detailed(model, test_loader, origin_threshold=origin_threshold)
        
        # Print results
        print_detailed_metrics(domain, metrics)
        
        # Save results
        output_path = domain_config["checkpoint"].parent / f"{domain.lower()}_detailed_metrics.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Detailed metrics saved to {output_path}")
        
        all_results[domain] = metrics
    
    # Summary comparison
    print("\n" + "="*70)
    print("CROSS-DOMAIN COMPARISON")
    print("="*70)
    
    print(f"\n{'Domain':<10} {'Node F1':<10} {'Origin F1':<10} {'Node Acc':<10} {'Origin Acc':<10}")
    print("-" * 70)
    
    for domain, metrics in all_results.items():
        node_f1 = metrics["node_classification"]["f1"] * 100
        origin_f1 = metrics["origin_detection"]["f1"] * 100
        node_acc = metrics["node_classification"]["accuracy"] * 100
        origin_acc = metrics["origin_detection"]["accuracy"] * 100
        
        print(f"{domain:<10} {node_f1:>8.2f}%  {origin_f1:>9.2f}%  {node_acc:>9.2f}%  {origin_acc:>10.2f}%")
    
    # Save summary
    summary_path = project_root / "experiments" / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Full summary saved to {summary_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE ✅")
    print("="*70)


if __name__ == "__main__":
    main()

