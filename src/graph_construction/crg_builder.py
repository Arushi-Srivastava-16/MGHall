"""
CRG (Causal Reasoning Graph) Builder.

This module converts unified format reasoning chains to PyTorch Geometric Data objects.
"""

import torch
from torch_geometric.data import Data
from typing import List, Optional

from ..data_processing.unified_schema import ReasoningChain, Domain


def build_crg(
    chain: ReasoningChain,
    node_features: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
) -> Data:
    """
    Build a PyTorch Geometric Data object from a reasoning chain.
    
    Args:
        chain: ReasoningChain in unified format
        node_features: Optional pre-computed node features (N × D)
        edge_features: Optional edge features (E × D_edge)
        
    Returns:
        PyTorch Geometric Data object
    """
    # Build edge index from dependency graph
    edges_list = []
    for edge in chain.dependency_graph.edges:
        if len(edge) == 2:
            edges_list.append([edge[0], edge[1]])
    
    if not edges_list:
        # If no edges, create a simple sequential chain
        num_nodes = len(chain.reasoning_steps)
        edges_list = [[i, i + 1] for i in range(num_nodes - 1)]
    
    # Convert to tensor
    if len(edges_list) > 0:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Node labels
    is_correct = torch.tensor(
        [step.is_correct for step in chain.reasoning_steps],
        dtype=torch.float,
    )
    is_origin = torch.tensor(
        [step.is_origin for step in chain.reasoning_steps],
        dtype=torch.float,
    )
    
    # Error type encoding (0: none, 1: factual, 2: logical, 3: syntax)
    error_types = []
    for step in chain.reasoning_steps:
        if step.error_type is None:
            error_types.append(0)
        elif step.error_type.value == "factual":
            error_types.append(1)
        elif step.error_type.value == "logical":
            error_types.append(2)
        elif step.error_type.value == "syntax":
            error_types.append(3)
        else:
            error_types.append(0)
    
    error_type = torch.tensor(error_types, dtype=torch.long)
    
    # Use provided features or create placeholder
    if node_features is None:
        # Placeholder: will be replaced by feature extractor
        num_nodes = len(chain.reasoning_steps)
        node_features = torch.zeros(num_nodes, 1)
    
    # Get number of edges
    num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0
    
    # Edge features (default to ones if not provided)
    if edge_features is None:
        edge_features = torch.ones(num_edges, 1) if num_edges > 0 else torch.empty((0, 1))
    
    # Domain encoding
    domain_map = {"math": 0, "code": 1, "medical": 2}
    domain = chain.domain.value if isinstance(chain.domain, Domain) else chain.domain
    domain_id = domain_map.get(domain, 0)
    
    # Create PyG Data object
    data = Data(
        x=node_features,  # Node features
        edge_index=edge_index,  # Edge connectivity
        edge_attr=edge_features,  # Edge features
        y=is_correct,  # Node labels (correctness)
        y_origin=is_origin,  # Origin labels
        y_error_type=error_type,  # Error type labels
        domain=torch.tensor([domain_id], dtype=torch.long),  # Domain ID
        num_nodes=len(chain.reasoning_steps),  # Number of nodes
    )
    
    return data


def build_crg_batch(data_list: List[Data]) -> Data:
    """
    Build a batch from multiple Data objects.
    
    This is a simple wrapper - PyG's DataLoader handles batching automatically.
    
    Args:
        data_list: List of Data objects
        
    Returns:
        Batched Data object
    """
    from torch_geometric.data import Batch
    
    return Batch.from_data_list(data_list)


if __name__ == "__main__":
    # Example usage
    from ..data_processing.unified_schema import ReasoningStep, DependencyGraph
    
    # Create a simple test chain
    steps = [
        ReasoningStep(step_id=0, text="Step 1", is_correct=True, depends_on=[]),
        ReasoningStep(step_id=1, text="Step 2", is_correct=False, is_origin=True, depends_on=[0]),
        ReasoningStep(step_id=2, text="Step 3", is_correct=False, depends_on=[1]),
    ]
    
    chain = ReasoningChain(
        domain=Domain.MATH,
        query_id="test_001",
        query="Test problem",
        ground_truth="Correct answer",
        reasoning_steps=steps,
        dependency_graph=DependencyGraph(nodes=[0, 1, 2], edges=[[0, 1], [1, 2]]),
    )
    
    # Build CRG
    data = build_crg(chain)
    
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Labels: {data.y.tolist()}")

