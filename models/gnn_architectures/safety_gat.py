"""
Safety-Aware GAT for Harmful Query Detection.

Extends the GAT architecture to predict safety scores alongside correctness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Optional

from models.gnn_architectures.gat_model import ConfidenceGatedGAT


class SafetyGAT(ConfidenceGatedGAT):
    """
    GAT with Safety Classification Head.
    
    Predicts:
    1. Node Correctness (Base Task)
    2. Origin Detection (Base Task)
    3. Query Safety (New Task: 0=Harmful, 1=Safe)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Final embedding dim
        final_dim = hidden_dim * num_heads
        
        # Safety Classifier
        # Takes global graph pooling (max or mean) as input
        self.safety_classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> dict:
        """
        Forward pass including safety prediction.
        """
        # Run base GAT
        base_outputs = super().forward(
            x, edge_index, confidence, batch, 
            return_attention_weights=return_attention_weights
        )
        
        # Get node embeddings
        h = base_outputs['node_embeddings'] # (N, final_dim)
        
        # Global pooling for graph-level safety prediction
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            
        from torch_geometric.nn import global_mean_pool
        graph_embedding = global_mean_pool(h, batch) # (Batch, final_dim)
        
        # Predict Safety
        safety_pred = self.safety_classifier(graph_embedding) # (Batch, 1)
        
        outputs = base_outputs
        outputs['safety_pred'] = safety_pred.squeeze(-1)
        
        return outputs


if __name__ == "__main__":
    print("Testing SafetyGAT...")
    model = SafetyGAT(input_dim=395)
    
    # Test input
    N = 10
    x = torch.randn(N, 395)
    edge_index = torch.randint(0, N, (2, 20))
    batch = torch.zeros(N, dtype=torch.long) # Single graph
    
    out = model(x, edge_index, batch=batch)
    print(f"Safety Pred Shape: {out['safety_pred'].shape}")
    print(f"Safety Score: {out['safety_pred'].item():.4f}")
    print("✓ SafetyGAT Test Passed")
