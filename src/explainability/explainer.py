"""
Explanation Generator for Safety and Correctness.

Generates human-readable explanations based on model predictions and attention weights.
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Any

class ExplanationGenerator:
    """Generates explanations for model decisions."""
    
    def __init__(self):
        # Safety Templates
        self.safety_templates = {
            "Illegal Acts": "This query appears to request assistance with illegal activities.",
            "Violence": "This content violates our safety policy regarding violence and physical harm.",
            "Hate Speech": "This query contains hate speech or harassment.",
            "Malicious Code": "This request involves generating malicious code or cyberattacks.",
            "Self-Harm": "This content involves self-harm or suicide.",
            "Unknown": "This query was flagged as unsafe by our safety filter."
        }
    
    def explain_safety_decision(self, 
                              query: str, 
                              is_safe_prob: float, 
                              attention_weights: List[tuple] = None,
                              tokens: List[str] = None) -> Dict[str, Any]:
        """
        Generate explanation for safety decision.
        
        Args:
            query: Original query text
            is_safe_prob: Probability of being safe (0.0 = Harmful, 1.0 = Safe)
            attention_weights: List of (edge_index, alpha) from GAT
            tokens: List of tokens corresponding to graph nodes
            
        Returns:
            Dict with explanation text and metadata
        """
        # Conservative threshold: > 0.6 to be considered safe
        # 0.5 is ambiguous and potentially unsafe
        is_safe = is_safe_prob > 0.6
        status = "SAFE" if is_safe else "UNSAFE"
        
        explanation = {
            "status": status,
            "confidence": f"{abs(is_safe_prob - 0.5) * 2:.2%}", # Scale 0.5-1.0 to 0-100%
            "reason": ""
        }
        
        if is_safe:
            explanation["reason"] = "This query complies with safety guidelines."
        else:
            # TODO: Integrate with category classifier for specific template
            # For now, generic
            explanation["reason"] = "This query was flagged as potentially harmful."
            
            # Attention analysis
            if attention_weights and tokens:
                # Simple heuristic: Aggregate attention per node (token)
                # Last layer attention usually most relevant
                last_layer_att = attention_weights[-1][1] # alpha
                edge_index = attention_weights[-1][0]
                
                # Sum attention incoming to each node
                # Note: GAT attention is edge-based. source -> target.
                # We want to know which SOURCE nodes (words) were most attended TO.
                # Actually, in GAT, 'alpha' is attention *score*.
                
                # Let's just find the max attention edges
                # Filter for self-loops or meaningful edges depends on graph construction
                pass 
                
        return explanation

    def generate_attention_heatmap(self, 
                                 tokens: List[str], 
                                 attention_weights: List[tuple]) -> Dict[str, float]:
        """
        Generate a token -> attention_score map for visualization.
        """
        if not attention_weights:
            return {}
            
        # Use last layer
        edge_index, alpha = attention_weights[-1]
        
        # alpha shape: (num_edges, num_heads)
        # Average heads
        att_scores = alpha.mean(dim=1) # (num_edges,)
        
        # Map back to nodes.
        # In our graph (CRG), nodes are steps.
        # If we had word-level graph, this would map to words.
        # Here we map to "Reasoning Steps".
        
        node_scores = torch.zeros(len(tokens))
        node_counts = torch.zeros(len(tokens))
        
        # Aggregate incoming attention to each node? 
        # Or outgoing?
        # High incoming attention -> This node is important context.
        
        src, dst = edge_index
        
        # Sum attention *targeting* a node
        for i in range(len(src)):
            target_node = dst[i]
            score = att_scores[i]
            if target_node < len(node_scores):
                node_scores[target_node] += score
                node_counts[target_node] += 1
                
        # Normalize
        # node_scores = node_scores / (node_counts + 1e-6)
        
        # Return map
        heatmap = {}
        for i, token in enumerate(tokens):
            heatmap[f"Step {i}: {token[:20]}..."] = float(node_scores[i])
            
        return heatmap

if __name__ == "__main__":
    explainer = ExplanationGenerator()
    print("Explainer initialized.")
