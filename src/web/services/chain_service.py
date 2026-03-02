"""
Chain Service.

Handles chain loading, processing, and graph data preparation.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data_processing.unified_schema import ReasoningChain


class ChainService:
    """Service for chain operations."""
    
    def __init__(self):
        """Initialize chain service."""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.chain_dirs = [
            self.project_root / "data/processed/splits",
            self.project_root / "data/processed/code_test_splits",
            self.project_root / "data/processed/medical_test_splits",
            self.project_root / "data/multi_model/generated_chains",
        ]
    
    def list_chains(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List available chains.
        
        Args:
            domain: Optional domain filter
            limit: Maximum number of chains to return
            
        Returns:
            List of chain summaries
        """
        chains = []
        
        for chain_dir in self.chain_dirs:
            if not chain_dir.exists():
                continue
            
            for jsonl_file in chain_dir.glob("*.jsonl"):
                if domain and domain not in jsonl_file.name:
                    continue
                
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if len(chains) >= limit:
                            return chains
                        
                        chain_dict = json.loads(line)
                        chain = ReasoningChain.from_dict(chain_dict)
                        
                        error_count = sum(1 for step in chain.reasoning_steps if not step.is_correct)
                        
                        chains.append({
                            "chain_id": chain.query_id,
                            "domain": chain.domain.value,
                            "query": chain.query[:100] + "..." if len(chain.query) > 100 else chain.query,
                            "num_steps": len(chain.reasoning_steps),
                            "has_errors": error_count > 0,
                            "error_count": error_count,
                        })
        
        return chains
    
    def get_chain(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chain by ID.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Chain details or None
        """
        for chain_dir in self.chain_dirs:
            if not chain_dir.exists():
                continue
            
            for jsonl_file in chain_dir.glob("*.jsonl"):
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        chain_dict = json.loads(line)
                        chain = ReasoningChain.from_dict(chain_dict)
                        
                        if chain.query_id == chain_id:
                            return {
                                "chain_id": chain.query_id,
                                "domain": chain.domain.value,
                                "query": chain.query,
                                "ground_truth": chain.ground_truth,
                                "steps": [
                                    {
                                        "step_id": step.step_id,
                                        "text": step.text,
                                        "is_correct": step.is_correct,
                                        "is_origin": step.is_origin,
                                        "error_type": step.error_type.value if step.error_type else None,
                                        "depends_on": step.depends_on,
                                    }
                                    for step in chain.reasoning_steps
                                ],
                                "dependency_graph": {
                                    "nodes": chain.dependency_graph.nodes,
                                    "edges": chain.dependency_graph.edges,
                                },
                            }
        
        return None
    
    def get_graph_data(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """
        Get graph visualization data for a chain.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Graph data (nodes and edges) or None
        """
        chain_data = self.get_chain(chain_id)
        if not chain_data:
            return None
        
        # Build nodes
        nodes = []
        for step in chain_data["steps"]:
            node = {
                "id": step["step_id"],
                "label": f"Step {step['step_id']}",
                "title": step["text"][:100] + "..." if len(step["text"]) > 100 else step["text"],
                "color": {
                    "background": "#90EE90" if step["is_correct"] else "#FF6B6B",
                    "border": "#4CAF50" if step["is_correct"] else "#D32F2F",
                },
                "shape": "box",
            }
            if step["is_origin"]:
                node["borderWidth"] = 3
            nodes.append(node)
        
        # Build edges
        edges = []
        for edge in chain_data["dependency_graph"]["edges"]:
            edges.append({
                "from": edge[0],
                "to": edge[1],
                "arrows": "to",
            })
        
        return {"nodes": nodes, "edges": edges}

