"""
Graph Attention Network (GAT) with Confidence Gating.

This module implements the base GAT architecture for hallucination detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Optional


class ConfidenceGatedGAT(nn.Module):
    """
    Graph Attention Network with confidence gating.
    
    This model uses GAT layers with attention weights multiplied by
    normalized confidence scores to prevent low-confidence nodes from
    contaminating their neighbors.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_confidence_gating: bool = True,
    ):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension for GAT layers
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_confidence_gating: Whether to use confidence gating
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_confidence_gating = use_confidence_gating
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
        
        # Confidence gating (learnable)
        if use_confidence_gating:
            self.confidence_proj = nn.Linear(hidden_dim * num_heads, 1)
        
        # Output layers for multi-task learning
        # Task 1: Node classification (correctness)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Task 2: Origin detection
        self.origin_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Task 3: Error type classification (optional)
        self.error_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # 0: none, 1: factual, 2: logical, 3: syntax
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
        Forward pass.
        
        Args:
            x: Node features (N, input_dim)
            edge_index: Edge connectivity (2, E)
            confidence: Optional confidence scores (N,)
            batch: Optional batch vector for graph-level tasks
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary with predictions for each task
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        attention_weights_list = []
        
        # GAT layers with confidence gating
        for i, gat_layer in enumerate(self.gat_layers):
            # Forward through GAT
            # Standard GATConv returns (x, (edge_index, alpha)) if return_attention_weights=True
            if return_attention_weights:
                x_new, (edge_idx, alpha) = gat_layer(x, edge_index, return_attention_weights=True)
                attention_weights_list.append((edge_idx, alpha))
            else:
                x_new = gat_layer(x, edge_index)
            
            # Apply confidence gating if enabled
            if self.use_confidence_gating and confidence is not None:
                # Compute confidence weights based on the new representation
                conf_weights = torch.sigmoid(self.confidence_proj(x_new))
                conf_weights = conf_weights.squeeze(-1)
                
                # Normalize confidence
                if confidence is not None:
                    conf_norm = torch.sigmoid(confidence)
                    conf_weights = conf_weights * conf_norm
                
                # Apply gating (element-wise multiplication)
                x_new = x_new * conf_weights.unsqueeze(-1)
            
            # Residual connection (if dimensions match)
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
            
            # Activation and dropout
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Multi-task predictions
        node_pred = torch.sigmoid(self.node_classifier(x))  # Binary classification
        origin_pred = torch.sigmoid(self.origin_classifier(x))  # Binary classification
        error_type_pred = self.error_type_classifier(x)  # Multi-class
        
        outputs = {
            "node_pred": node_pred.squeeze(-1),  # (N,)
            "origin_pred": origin_pred.squeeze(-1),  # (N,)
            "error_type_pred": error_type_pred,  # (N, 4)
            "node_embeddings": x,  # (N, hidden_dim * num_heads)
        }
        
        if return_attention_weights:
            outputs["attention_weights"] = attention_weights_list
        
        return outputs


class SimpleGCN(nn.Module):
    """
    Simple Graph Convolutional Network for ablation studies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = dropout
        
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        self.origin_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> dict:
        x = self.input_proj(x)
        x = F.relu(x)
        
        for gcn_layer in self.gcn_layers:
            x_new = gcn_layer(x, edge_index)
            x = x + x_new  # Residual
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return {
            "node_pred": torch.sigmoid(self.node_classifier(x)).squeeze(-1),
            "origin_pred": torch.sigmoid(self.origin_classifier(x)).squeeze(-1),
            "error_type_pred": torch.zeros(x.size(0), 4).to(x.device),  # Placeholder
            "node_embeddings": x,
        }


if __name__ == "__main__":
    # Test model
    model = ConfidenceGatedGAT(
        input_dim=384,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
    )
    
    # Dummy input
    num_nodes = 10
    num_edges = 15
    x = torch.randn(num_nodes, 384)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    confidence = torch.rand(num_nodes)
    
    outputs = model(x, edge_index, confidence=confidence)
    
    print(f"Node predictions shape: {outputs['node_pred'].shape}")
    print(f"Origin predictions shape: {outputs['origin_pred'].shape}")
    print(f"Error type predictions shape: {outputs['error_type_pred'].shape}")

