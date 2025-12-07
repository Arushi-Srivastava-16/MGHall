"""
Comprehensive validation script to compute all metrics on a small subset.

Computes:
- Accuracy, Precision, Recall, F1
- AUROC, AUPRC
- Confusion Matrix
- Per-class metrics
- Origin detection metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import json


def create_small_test_set(num_samples=100):
    """Create a small test set from existing splits."""
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "splits"
    test_dir = Path(__file__).parent.parent / "data" / "processed" / "validation_test"
    test_dir.mkdir(exist_ok=True)
    
    # Take first 100 samples from test split
    with open(data_dir / "test.jsonl", 'r') as f_in, open(test_dir / "test_100.jsonl", 'w') as f_out:
        for i, line in enumerate(f_in):
            if i >= num_samples:
                break
            f_out.write(line)
    
    print(f"✓ Created test set with {i+1} samples")
    return test_dir / "test_100.jsonl"


def compute_all_metrics(y_true, y_pred, y_scores=None, prefix=""):
    """Compute comprehensive metrics."""
    metrics = {}
    
    # Basic metrics
    metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
    metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics[f'{prefix}true_negatives'] = int(tn)
        metrics[f'{prefix}false_positives'] = int(fp)
        metrics[f'{prefix}false_negatives'] = int(fn)
        metrics[f'{prefix}true_positives'] = int(tp)
        
        # Specificity
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # NPV (Negative Predictive Value)
        metrics[f'{prefix}npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # If we have probability scores
    if y_scores is not None:
        try:
            metrics[f'{prefix}auroc'] = roc_auc_score(y_true, y_scores)
            metrics[f'{prefix}auprc'] = average_precision_score(y_true, y_scores)
        except:
            metrics[f'{prefix}auroc'] = 0.0
            metrics[f'{prefix}auprc'] = 0.0
    
    return metrics


def evaluate_model(model, test_loader, device='cpu'):
    """Run model evaluation and collect predictions."""
    model.eval()
    
    all_node_labels = []
    all_node_preds = []
    all_node_scores = []
    
    all_origin_labels = []
    all_origin_preds = []
    all_origin_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch.x, batch.edge_index)
            
            # Node classification (correctness prediction)
            # outputs['node_pred'] is already sigmoid activated
            node_probs = outputs['node_pred']  # Already in [0, 1]
            node_pred = (node_probs > 0.5).long()  # 0 = incorrect, 1 = correct
            
            all_node_labels.extend(batch.y.cpu().numpy())
            all_node_preds.extend(node_pred.cpu().numpy())
            all_node_scores.extend(node_probs.cpu().numpy())  # Probability of being correct
            
            # Origin detection
            if hasattr(batch, 'y_origin'):
                # outputs['origin_pred'] is already sigmoid activated
                origin_probs = outputs['origin_pred']
                origin_pred = (origin_probs > 0.5).long()
                
                all_origin_labels.extend(batch.y_origin.cpu().numpy())
                all_origin_preds.extend(origin_pred.cpu().numpy())
                all_origin_scores.extend(origin_probs.cpu().numpy())
    
    return {
        'node_labels': np.array(all_node_labels),
        'node_preds': np.array(all_node_preds),
        'node_scores': np.array(all_node_scores),
        'origin_labels': np.array(all_origin_labels),
        'origin_preds': np.array(all_origin_preds),
        'origin_scores': np.array(all_origin_scores),
    }


def print_metrics_table(metrics, title="Metrics"):
    """Pretty print metrics table."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)
    
    # Group metrics
    basic = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv']
    advanced = ['auroc', 'auprc']
    confusion = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
    
    print("\n📊 Basic Classification Metrics:")
    print("-" * 60)
    for key in basic:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key:.<25} {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"  {key:.<25} {value}")
    
    if any(k in metrics for k in advanced):
        print("\n📈 Advanced Metrics:")
        print("-" * 60)
        for key in advanced:
            if key in metrics:
                print(f"  {key.upper():.<25} {metrics[key]:.4f}")
    
    if any(k in metrics for k in confusion):
        print("\n🔢 Confusion Matrix Breakdown:")
        print("-" * 60)
        for key in confusion:
            if key in metrics:
                print(f"  {key.replace('_', ' ').title():.<25} {metrics[key]}")
    
    print('='*60)


def main():
    print("="*60)
    print("COMPREHENSIVE METRICS VALIDATION (100 samples)")
    print("="*60)
    
    # Create small test set
    print("\n1️⃣ Creating test subset...")
    test_path = create_small_test_set(num_samples=100)
    
    # Load the trained model from quick test
    print("\n2️⃣ Loading trained model...")
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints" / "test_run"
    model_path = checkpoint_dir / "best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run quick_test_train.py first to train a model.")
        return
    
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    
    # Model config (same as training)
    config = {
        "input_dim": 384 + 5 + 6,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
    }
    
    model = ConfidenceGatedGAT(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        use_confidence_gating=True,
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', '?')}")
    
    # Create data loader
    print("\n3️⃣ Loading test data...")
    feature_extractor = FeatureExtractor(embedding_dim=384)
    
    from torch_geometric.loader import DataLoader
    from src.data_processing.graph_dataloader import ReasoningChainDataset
    
    test_dataset = ReasoningChainDataset(
        test_path,
        feature_extractor=feature_extractor,
        cache_path=Path(__file__).parent.parent / "data" / "graphs" / "validation_cache" / "test.pkl",
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )
    
    print(f"✓ Loaded {len(test_dataset)} graphs")
    
    # Evaluate
    print("\n4️⃣ Running evaluation...")
    results = evaluate_model(model, test_loader, device='cpu')
    
    # Compute metrics
    print("\n5️⃣ Computing metrics...")
    
    # Node classification metrics
    node_metrics = compute_all_metrics(
        results['node_labels'],
        results['node_preds'],
        results['node_scores'],
        prefix='node_'
    )
    
    # Origin detection metrics
    origin_metrics = compute_all_metrics(
        results['origin_labels'],
        results['origin_preds'],
        results['origin_scores'],
        prefix='origin_'
    )
    
    # Print results
    print_metrics_table(node_metrics, "Node Classification (Correctness Prediction)")
    print_metrics_table(origin_metrics, "Origin Detection (First Error Identification)")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report (Node Classification):")
    print("="*60)
    print(classification_report(
        results['node_labels'],
        results['node_preds'],
        target_names=['Incorrect', 'Correct'],
        digits=4
    ))
    
    print("\n📋 Detailed Classification Report (Origin Detection):")
    print("="*60)
    print(classification_report(
        results['origin_labels'],
        results['origin_preds'],
        target_names=['Not Origin', 'Is Origin'],
        digits=4
    ))
    
    # Save results
    output_path = checkpoint_dir / "validation_metrics.json"
    all_metrics = {
        'node_classification': node_metrics,
        'origin_detection': origin_metrics,
        'test_samples': len(test_dataset),
        'total_nodes': len(results['node_labels']),
    }
    
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test Samples: {len(test_dataset)}")
    print(f"Total Nodes Evaluated: {len(results['node_labels'])}")
    print(f"\nNode Classification:")
    print(f"  • Accuracy: {node_metrics['node_accuracy']:.4f} ({node_metrics['node_accuracy']*100:.2f}%)")
    print(f"  • F1 Score: {node_metrics['node_f1']:.4f}")
    print(f"  • AUROC: {node_metrics.get('node_auroc', 0):.4f}")
    print(f"\nOrigin Detection:")
    print(f"  • Accuracy: {origin_metrics['origin_accuracy']:.4f} ({origin_metrics['origin_accuracy']*100:.2f}%)")
    print(f"  • F1 Score: {origin_metrics['origin_f1']:.4f}")
    print(f"  • AUROC: {origin_metrics.get('origin_auroc', 0):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

