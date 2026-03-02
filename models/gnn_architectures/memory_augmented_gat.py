"""
Memory-Augmented GAT for Multi-Turn Hallucination Detection.

Extends ConfidenceGatedGAT with memory attention mechanism to attend
to representations from previous conversation turns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Optional

from models.gnn_architectures.gat_model import ConfidenceGatedGAT


class MemoryAugmentedGAT(ConfidenceGatedGAT):
    """
    GAT with Memory Attention for Cross-Turn Hallucination Detection.
    
    Extends the base ConfidenceGatedGAT with:
    - Multi-head attention to previous turn representations
    - Memory projection layer to combine current + memory features
    - Cross-turn consistency prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_confidence_gating: bool = True,
        memory_dim: int = 64,
        num_memory_heads: int = 4,
    ):
        """
        Initialize Memory-Augmented GAT.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GAT layers
            num_layers: Number of GAT layers
            num_heads: Number of attention heads for GAT
            dropout: Dropout rate
            use_confidence_gating: Whether to use confidence gating
            memory_dim: Dimension for memory features
            num_memory_heads: Number of attention heads for memory
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_confidence_gating=use_confidence_gating,
        )
        
        # Memory attention mechanism
        self.memory_dim = memory_dim
        self.num_memory_heads = num_memory_heads
        
        # Final dimension after GAT layers
        final_dim = hidden_dim * num_heads
        
        # Multi-head attention to previous turns
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=final_dim,
            num_heads=num_memory_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Project memory context
        self.memory_projection = nn.Linear(memory_dim, final_dim)
        
        # Combine current + memory representations
        self.memory_combiner = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim),
        )
        
        # Additional task: cross-turn consistency prediction
        self.consistency_head = nn.Sequential(
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
        memory_context: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with optional memory context.
        
        Args:
            x: Node features (N, input_dim)
            edge_index: Edge connectivity (2, E)
            confidence: Optional confidence scores (N,)
            batch: Optional batch vector for batching
            memory_context: Optional memory from previous turns (M, memory_dim)
                           where M is number of nodes from previous turns
        
        Returns:
            Dictionary with predictions including:
            - node_pred: Correctness prediction
            - origin_pred: Origin detection
            - error_type_pred: Error type classification
            - consistency_pred: Cross-turn consistency (NEW)
            - node_embeddings: Node representations
        """
        # Standard GAT forward pass
        base_outputs = super().forward(x, edge_index, confidence, batch)
        
        # Get node embeddings after GAT layers
        h = base_outputs['node_embeddings']  # (N, final_dim)
        
        # If memory available, attend to past turns
        if memory_context is not None and memory_context.size(0) > 0:
            # Project memory context to same dim as current embeddings
            memory_proj = self.memory_projection(memory_context)  # (M, final_dim)
            
            # Add batch dimension for attention
            # Query: current turn nodes
            query = h.unsqueeze(0)  # (1, N, final_dim)
            
            # Key/Value: previous turn nodes
            key = memory_proj.unsqueeze(0)  # (1, M, final_dim)
            value = memory_proj.unsqueeze(0)  # (1, M, final_dim)
            
            # Multi-head attention
            attn_out, attn_weights = self.memory_attention(
                query=query,
                key=key,
                value=value,
            )
            attn_out = attn_out.squeeze(0)  # (N, final_dim)
            
            # Combine current + memory representations
            combined = torch.cat([h, attn_out], dim=-1)  # (N, 2*final_dim)
            h_memory = self.memory_combiner(combined)  # (N, final_dim)
            
            # Predict cross-turn consistency
            consistency_pred = self.consistency_head(h_memory)  # (N, 1)
            consistency_pred = consistency_pred.squeeze(-1)  # (N,)
            
            # Update embeddings for downstream tasks
            h = h_memory
        else:
            # No memory context, default to high consistency
            consistency_pred = torch.ones(h.size(0), device=h.device)
            attn_weights = None
        
        # Re-compute task predictions with memory-aware embeddings
        node_pred = torch.sigmoid(self.node_classifier(h))
        origin_pred = torch.sigmoid(self.origin_classifier(h))
        error_type_pred = self.error_type_classifier(h)
        
        return {
            'node_pred': node_pred.squeeze(-1),
            'origin_pred': origin_pred.squeeze(-1),
            'error_type_pred': error_type_pred,
            'consistency_pred': consistency_pred,  # NEW
            'node_embeddings': h,
            'attention_weights': attn_weights,  # For visualization
        }
    
    def get_memory_context(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract memory context from current turn embeddings.
        
        Args:
            node_embeddings: Current turn node embeddings (N, final_dim)
            
        Returns:
            Memory context tensor (N, memory_dim)
        """
        # For simplicity, use a projection to reduce dimensionality
        # In practice, you could use a more sophisticated memory encoding
        
        if not hasattr(self, 'memory_encoder'):
            # Create memory encoder on first call
            final_dim = node_embeddings.size(-1)
            self.memory_encoder = nn.Linear(final_dim, self.memory_dim).to(node_embeddings.device)
        
        return self.memory_encoder(node_embeddings)


# Test the model
if __name__ == "__main__":
    print("Testing MemoryAugmentedGAT...")
    
    # Create model
    model = MemoryAugmentedGAT(
        input_dim=395,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        memory_dim=64,
        num_memory_heads=2,
    )
    
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test without memory
    num_nodes = 10
    num_edges = 15
    x = torch.randn(num_nodes, 395)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    confidence = torch.rand(num_nodes)
    
    print("\n1. Forward pass WITHOUT memory:")
    outputs = model(x, edge_index, confidence=confidence)
    print(f"  Node predictions: {outputs['node_pred'].shape}")
    print(f"  Consistency predictions: {outputs['consistency_pred'].shape}")
    print(f"  Consistency scores: {outputs['consistency_pred'][:5].tolist()}")
    
    # Test WITH memory from previous turn
    print("\n2. Forward pass WITH memory:")
    
    # Simulate previous turn embeddings
    prev_embeddings = outputs['node_embeddings'].detach()
    memory_context = model.get_memory_context(prev_embeddings)
    
    # Forward pass with memory
    outputs_with_memory = model(
        x, edge_index,
        confidence=confidence,
        memory_context=memory_context
    )
    
    print(f"  Node predictions: {outputs_with_memory['node_pred'].shape}")
    print(f"  Consistency predictions: {outputs_with_memory['consistency_pred'].shape}")
    print(f"  Consistency scores: {outputs_with_memory['consistency_pred'][:5].tolist()}")
    print(f"  Attention weights shape: {outputs_with_memory['attention_weights'].shape if outputs_with_memory['attention_weights'] is not None else 'None'}")
    
    print("\n✓ MemoryAugmentedGAT test passed!")
