"""
Analysis Report Generator.

This script generates comprehensive analysis reports including:
- Model comparison across domains
- Error analysis
- Attention pattern visualizations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

from src.evaluation.evaluator import ModelEvaluator
from models.gnn_architectures.gat_model import ConfidenceGatedGAT


def generate_error_analysis(
    model: torch.nn.Module,
    data_loader,
    output_dir: Path,
    domain: str,
):
    """
    Analyze error patterns in model predictions.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        output_dir: Directory to save analysis
        domain: Domain name
    """
    model.eval()
    device = next(model.parameters()).device
    
    error_types = {
        'false_positives': [],  # Predicted correct, actually incorrect
        'false_negatives': [],  # Predicted incorrect, actually correct
        'true_positives': [],
        'true_negatives': [],
    }
    
    error_positions = []
    error_depths = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch=batch.batch)
            
            node_pred = (outputs['node_pred'] > 0.5).float()
            node_true = batch.y
            
            # Classify errors
            for i in range(len(node_pred)):
                pred = node_pred[i].item()
                true = node_true[i].item()
                
                if pred == 1 and true == 0:
                    error_types['false_positives'].append(i)
                elif pred == 0 and true == 1:
                    error_types['false_negatives'].append(i)
                elif pred == 1 and true == 1:
                    error_types['true_positives'].append(i)
                else:
                    error_types['true_negatives'].append(i)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error type distribution
    ax = axes[0, 0]
    counts = [len(v) for v in error_types.values()]
    ax.bar(error_types.keys(), counts)
    ax.set_title(f'Error Distribution - {domain}')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Placeholder for other plots
    axes[0, 1].text(0.5, 0.5, 'Error Position Analysis', ha='center', va='center')
    axes[1, 0].text(0.5, 0.5, 'Error Depth Analysis', ha='center', va='center')
    axes[1, 1].text(0.5, 0.5, 'Propagation Patterns', ha='center', va='center')
    
    plt.tight_layout()
    output_path = output_dir / f"error_analysis_{domain}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical report
    report = {
        "domain": domain,
        "error_counts": {k: len(v) for k, v in error_types.items()},
        "total_samples": sum(len(v) for v in error_types.values()),
    }
    
    report_path = output_dir / f"error_report_{domain}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)


def visualize_attention_patterns(
    model: torch.nn.Module,
    sample_batch,
    output_dir: Path,
):
    """
    Visualize attention patterns from GAT model.
    
    Args:
        model: Trained GAT model
        sample_batch: Sample batch for visualization
        output_dir: Directory to save visualizations
    """
    if not isinstance(model, ConfidenceGatedGAT):
        print("Attention visualization only available for GAT models")
        return
    
    # This would require modifications to the GAT model to return attention weights
    # Placeholder visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, 'Attention Pattern Visualization\n(Requires attention weight extraction)',
           ha='center', va='center', fontsize=14)
    
    output_path = output_dir / "attention_patterns.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comprehensive_report(
    models: Dict[str, torch.nn.Module],
    data_loaders: Dict[str, any],
    output_dir: Path,
):
    """
    Generate comprehensive analysis report.
    
    Args:
        models: Dictionary of trained models
        data_loaders: Dictionary of data loaders per domain
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}...")
        evaluator = ModelEvaluator(model)
        
        model_results = {}
        for domain, loader in data_loaders.items():
            print(f"  Evaluating on {domain}...")
            metrics = evaluator.evaluate(loader, domain=domain)
            model_results[domain] = metrics
            
            # Generate confusion matrices
            evaluator.generate_confusion_matrix_plot(
                metrics,
                output_dir / f"cm_{model_name}_{domain}_node.png",
                task="node_classification",
            )
            
            # Error analysis
            generate_error_analysis(model, loader, output_dir, f"{model_name}_{domain}")
        
        all_results[model_name] = model_results
    
    # Generate cross-model comparison
    comparison_data = []
    for model_name, domain_results in all_results.items():
        for domain, metrics in domain_results.items():
            comparison_data.append({
                "model": model_name,
                "domain": domain,
                "node_f1": metrics["node_classification"]["f1"],
                "origin_f1": metrics["origin_detection"]["f1"],
            })
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models_list = list(models.keys())
    domains_list = list(data_loaders.keys())
    
    # Node F1 heatmap
    node_f1_matrix = np.zeros((len(models_list), len(domains_list)))
    for i, model in enumerate(models_list):
        for j, domain in enumerate(domains_list):
            data = [d for d in comparison_data if d["model"] == model and d["domain"] == domain]
            node_f1_matrix[i, j] = data[0]["node_f1"] if data else 0
    
    sns.heatmap(node_f1_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
               xticklabels=domains_list, yticklabels=models_list, ax=ax1)
    ax1.set_title('Node Classification F1 Scores')
    
    # Origin F1 heatmap
    origin_f1_matrix = np.zeros((len(models_list), len(domains_list)))
    for i, model in enumerate(models_list):
        for j, domain in enumerate(domains_list):
            data = [d for d in comparison_data if d["model"] == model and d["domain"] == domain]
            origin_f1_matrix[i, j] = data[0]["origin_f1"] if data else 0
    
    sns.heatmap(origin_f1_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
               xticklabels=domains_list, yticklabels=models_list, ax=ax2)
    ax2.set_title('Origin Detection F1 Scores')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive JSON report
    report_path = output_dir / "comprehensive_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "models": list(models.keys()),
            "domains": list(data_loaders.keys()),
            "results": all_results,
            "comparison_data": comparison_data,
        }, f, indent=2)
    
    print(f"\nComprehensive report saved to {output_dir}")
    print(f"  - Model comparison heatmap: model_comparison_heatmap.png")
    print(f"  - JSON report: comprehensive_report.json")
    print(f"  - Individual confusion matrices and error analyses")


if __name__ == "__main__":
    print("Analysis report generator loaded")
    print("Use generate_comprehensive_report() to create full analysis")

