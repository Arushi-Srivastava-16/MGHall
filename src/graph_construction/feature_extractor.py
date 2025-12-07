"""
Node Feature Extractor.

This module extracts features for graph nodes including text embeddings,
graph topology, and task features.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import networkx as nx

from ..data_processing.unified_schema import ReasoningChain, Domain


class FeatureExtractor:
    """Extract features for reasoning chain steps."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ):
        """
        Initialize feature extractor.
        
        Args:
            embedding_model: Sentence transformer model name
            embedding_dim: Dimension of text embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = embedding_dim
    
    def extract_text_features(self, step_texts: List[str]) -> torch.Tensor:
        """
        Extract text embeddings for steps.
        
        Args:
            step_texts: List of step text strings
            
        Returns:
            Tensor of shape (N, embedding_dim) where N is number of steps
        """
        embeddings = self.embedding_model.encode(step_texts, convert_to_tensor=True)
        return embeddings
    
    def extract_topology_features(
        self,
        chain: ReasoningChain,
    ) -> torch.Tensor:
        """
        Extract graph topology features for each node.
        
        Features include:
        - Node degree (in-degree, out-degree)
        - Clustering coefficient
        - Path length from root
        - Centrality measures
        
        Args:
            chain: ReasoningChain to extract features from
            
        Returns:
            Tensor of shape (N, num_topology_features)
        """
        # Build NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(chain.dependency_graph.nodes)
        G.add_edges_from(chain.dependency_graph.edges)
        
        num_nodes = len(chain.reasoning_steps)
        features = []
        
        for node_id in range(num_nodes):
            node_features = []
            
            # In-degree and out-degree
            in_degree = G.in_degree(node_id)
            out_degree = G.out_degree(node_id)
            node_features.extend([in_degree, out_degree])
            
            # Clustering coefficient (for undirected version)
            G_undirected = G.to_undirected()
            clustering = nx.clustering(G_undirected, node_id)
            node_features.append(clustering)
            
            # Path length from root (node 0)
            try:
                if nx.has_path(G, 0, node_id):
                    path_length = nx.shortest_path_length(G, 0, node_id)
                else:
                    path_length = -1  # Unreachable
            except:
                path_length = -1
            node_features.append(path_length)
            
            # Betweenness centrality (normalized)
            try:
                betweenness = nx.betweenness_centrality(G)[node_id]
            except:
                betweenness = 0.0
            node_features.append(betweenness)
            
            features.append(node_features)
        
        # Pad if needed
        max_features = max(len(f) for f in features) if features else 0
        for f in features:
            while len(f) < max_features:
                f.append(0.0)
        
        return torch.tensor(features, dtype=torch.float)
    
    def extract_task_features(
        self,
        chain: ReasoningChain,
    ) -> torch.Tensor:
        """
        Extract task-level features for each node.
        
        Features include:
        - Domain encoding
        - Reasoning depth (max path length)
        - Step position (normalized)
        - Factual density (placeholder for future)
        
        Args:
            chain: ReasoningChain to extract features from
            
        Returns:
            Tensor of shape (N, num_task_features)
        """
        num_nodes = len(chain.reasoning_steps)
        
        # Domain encoding
        domain_map = {"math": 0, "code": 1, "medical": 2}
        domain = chain.domain.value if isinstance(chain.domain, Domain) else chain.domain
        domain_id = domain_map.get(domain, 0)
        
        # Reasoning depth (max path length in graph)
        G = nx.DiGraph()
        G.add_nodes_from(chain.dependency_graph.nodes)
        G.add_edges_from(chain.dependency_graph.edges)
        
        max_depth = 0
        if G.number_of_nodes() > 0:
            try:
                # Find longest path
                longest_path = nx.dag_longest_path(G)
                max_depth = len(longest_path) - 1 if longest_path else 0
            except:
                max_depth = num_nodes - 1
        
        features = []
        for i, step in enumerate(chain.reasoning_steps):
            step_features = []
            
            # Domain (one-hot encoding)
            domain_onehot = [0, 0, 0]
            domain_onehot[domain_id] = 1
            step_features.extend(domain_onehot)
            
            # Reasoning depth
            step_features.append(max_depth)
            
            # Step position (normalized)
            step_position = i / max(num_nodes - 1, 1)
            step_features.append(step_position)
            
            # Factual density (placeholder - would require external knowledge)
            factual_density = 0.5  # Placeholder
            step_features.append(factual_density)
            
            features.append(step_features)
        
        return torch.tensor(features, dtype=torch.float)
    
    def extract_all_features(
        self,
        chain: ReasoningChain,
        include_text: bool = True,
        include_topology: bool = True,
        include_task: bool = True,
    ) -> torch.Tensor:
        """
        Extract all features for a reasoning chain.
        
        Args:
            chain: ReasoningChain to extract features from
            include_text: Whether to include text embeddings
            include_topology: Whether to include topology features
            include_task: Whether to include task features
            
        Returns:
            Tensor of shape (N, total_feature_dim)
        """
        feature_parts = []
        
        if include_text:
            step_texts = [step.text for step in chain.reasoning_steps]
            text_features = self.extract_text_features(step_texts)
            feature_parts.append(text_features)
        
        if include_topology:
            topology_features = self.extract_topology_features(chain)
            feature_parts.append(topology_features)
        
        if include_task:
            task_features = self.extract_task_features(chain)
            feature_parts.append(task_features)
        
        if not feature_parts:
            # Return placeholder if no features requested
            num_nodes = len(chain.reasoning_steps)
            return torch.zeros(num_nodes, 1)
        
        # Move all features to CPU before concatenating (to avoid device mismatch)
        feature_parts = [f.cpu() for f in feature_parts]
        
        # Concatenate all features
        all_features = torch.cat(feature_parts, dim=1)
        
        return all_features


if __name__ == "__main__":
    # Example usage
    from ..data_processing.unified_schema import ReasoningStep, DependencyGraph, Domain
    
    steps = [
        ReasoningStep(step_id=0, text="First step", is_correct=True, depends_on=[]),
        ReasoningStep(step_id=1, text="Second step", is_correct=True, depends_on=[0]),
        ReasoningStep(step_id=2, text="Third step", is_correct=False, depends_on=[1]),
    ]
    
    chain = ReasoningChain(
        domain=Domain.MATH,
        query_id="test",
        query="Test problem",
        ground_truth="Answer",
        reasoning_steps=steps,
        dependency_graph=DependencyGraph(nodes=[0, 1, 2], edges=[[0, 1], [1, 2]]),
    )
    
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(chain)
    
    print(f"Feature shape: {features.shape}")
    print(f"Features:\n{features}")

