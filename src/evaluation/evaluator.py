"""
Evaluation Framework.

This module provides comprehensive evaluation metrics for GNN models including
per-domain performance, cross-domain transfer, and detailed error analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained GNN model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        domain: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Evaluate model on a data loader.
        
        Args:
            data_loader: PyTorch Geometric DataLoader
            domain: Domain name for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_node_preds = []
        all_node_labels = []
        all_origin_preds = []
        all_origin_labels = []
        all_node_probs = []
        all_origin_probs = []
        
        for batch in data_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Collect predictions and labels
            node_pred = (outputs['node_pred'] > 0.5).float().cpu().numpy()
            origin_pred = (outputs['origin_pred'] > 0.5).float().cpu().numpy()
            
            all_node_preds.extend(node_pred)
            all_node_labels.extend(batch.y.cpu().numpy())
            all_origin_preds.extend(origin_pred)
            all_origin_labels.extend(batch.y_origin.cpu().numpy())
            all_node_probs.extend(outputs['node_pred'].cpu().numpy())
            all_origin_probs.extend(outputs['origin_pred'].cpu().numpy())
        
        # Convert to numpy arrays
        all_node_preds = np.array(all_node_preds)
        all_node_labels = np.array(all_node_labels)
        all_origin_preds = np.array(all_origin_preds)
        all_origin_labels = np.array(all_origin_labels)
        all_node_probs = np.array(all_node_probs)
        all_origin_probs = np.array(all_origin_probs)
        
        # Compute metrics
        metrics = self._compute_metrics(
            node_preds=all_node_preds,
            node_labels=all_node_labels,
            node_probs=all_node_probs,
            origin_preds=all_origin_preds,
            origin_labels=all_origin_labels,
            origin_probs=all_origin_probs,
            domain=domain,
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        node_preds: np.ndarray,
        node_labels: np.ndarray,
        node_probs: np.ndarray,
        origin_preds: np.ndarray,
        origin_labels: np.ndarray,
        origin_probs: np.ndarray,
        domain: str,
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics."""
        
        # Node classification metrics
        node_acc = accuracy_score(node_labels, node_preds)
        node_prec, node_rec, node_f1, _ = precision_recall_fscore_support(
            node_labels,
            node_preds,
            average='binary',
            zero_division=0,
        )
        
        try:
            node_auc = roc_auc_score(node_labels, node_probs)
        except:
            node_auc = 0.0
        
        # Origin detection metrics
        origin_acc = accuracy_score(origin_labels, origin_preds)
        origin_prec, origin_rec, origin_f1, _ = precision_recall_fscore_support(
            origin_labels,
            origin_preds,
            average='binary',
            zero_division=0,
        )
        
        try:
            origin_auc = roc_auc_score(origin_labels, origin_probs)
        except:
            origin_auc = 0.0
        
        # Confusion matrices
        node_cm = confusion_matrix(node_labels, node_preds)
        origin_cm = confusion_matrix(origin_labels, origin_preds)
        
        metrics = {
            "domain": domain,
            "node_classification": {
                "accuracy": float(node_acc),
                "precision": float(node_prec),
                "recall": float(node_rec),
                "f1": float(node_f1),
                "auc": float(node_auc),
                "confusion_matrix": node_cm.tolist(),
            },
            "origin_detection": {
                "accuracy": float(origin_acc),
                "precision": float(origin_prec),
                "recall": float(origin_rec),
                "f1": float(origin_f1),
                "auc": float(origin_auc),
                "confusion_matrix": origin_cm.tolist(),
            },
        }
        
        return metrics
    
    def cross_domain_evaluation(
        self,
        loaders: Dict[str, Any],
        trained_on: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model across multiple domains.
        
        Args:
            loaders: Dictionary mapping domain names to data loaders
            trained_on: Domain the model was trained on
            
        Returns:
            Dictionary with metrics for each domain
        """
        results = {}
        
        for domain, loader in loaders.items():
            print(f"\nEvaluating on {domain} domain...")
            metrics = self.evaluate(loader, domain=domain)
            metrics["trained_on"] = trained_on
            metrics["is_in_domain"] = (domain == trained_on)
            results[domain] = metrics
        
        return results
    
    def generate_confusion_matrix_plot(
        self,
        metrics: Dict[str, Any],
        output_path: Path,
        task: str = "node_classification",
    ) -> None:
        """Generate confusion matrix visualization."""
        cm = np.array(metrics[task]["confusion_matrix"])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Incorrect', 'Correct'],
            yticklabels=['Incorrect', 'Correct'],
        )
        plt.title(f'Confusion Matrix - {task} ({metrics["domain"]})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(
        self,
        all_results: Dict[str, Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """
        Generate comparative analysis across domains.
        
        Args:
            all_results: Dictionary mapping model names to their cross-domain results
            output_path: Path to save report
        """
        # Extract metrics for comparison
        comparison = []
        
        for model_name, domain_results in all_results.items():
            for domain, metrics in domain_results.items():
                comparison.append({
                    "model": model_name,
                    "domain": domain,
                    "trained_on": metrics["trained_on"],
                    "in_domain": metrics["is_in_domain"],
                    "node_f1": metrics["node_classification"]["f1"],
                    "node_acc": metrics["node_classification"]["accuracy"],
                    "origin_f1": metrics["origin_detection"]["f1"],
                    "origin_acc": metrics["origin_detection"]["accuracy"],
                })
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(set(c["model"] for c in comparison))
        domains = list(set(c["domain"] for c in comparison))
        
        # Node F1 comparison
        ax = axes[0, 0]
        for model in models:
            model_data = [c for c in comparison if c["model"] == model]
            values = [c["node_f1"] for c in model_data]
            ax.plot(domains, values, marker='o', label=model)
        ax.set_title('Node Classification F1 by Domain')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Origin F1 comparison
        ax = axes[0, 1]
        for model in models:
            model_data = [c for c in comparison if c["model"] == model]
            values = [c["origin_f1"] for c in model_data]
            ax.plot(domains, values, marker='o', label=model)
        ax.set_title('Origin Detection F1 by Domain')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # In-domain vs Out-of-domain
        ax = axes[1, 0]
        in_domain = [c for c in comparison if c["in_domain"]]
        out_domain = [c for c in comparison if not c["in_domain"]]
        
        ax.bar(['In-Domain', 'Out-of-Domain'], [
            np.mean([c["node_f1"] for c in in_domain]),
            np.mean([c["node_f1"] for c in out_domain]),
        ])
        ax.set_title('In-Domain vs Out-of-Domain Performance')
        ax.set_ylabel('Average Node F1')
        ax.grid(True, alpha=0.3)
        
        # Transfer learning heatmap
        ax = axes[1, 1]
        transfer_matrix = np.zeros((len(domains), len(domains)))
        for i, train_domain in enumerate(domains):
            for j, test_domain in enumerate(domains):
                scores = [c["node_f1"] for c in comparison 
                         if c["trained_on"] == train_domain and c["domain"] == test_domain]
                transfer_matrix[i, j] = np.mean(scores) if scores else 0
        
        sns.heatmap(transfer_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=domains, yticklabels=domains, ax=ax)
        ax.set_title('Transfer Learning Matrix (Node F1)')
        ax.set_xlabel('Test Domain')
        ax.set_ylabel('Train Domain')
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save JSON report
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    print("Evaluator module loaded successfully")


