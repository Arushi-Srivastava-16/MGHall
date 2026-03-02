"""
Streaming Inference Pipeline.

Real-time analysis of reasoning chains as steps are generated incrementally.
"""

import torch
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import ReasoningChain, ReasoningStep, DependencyGraph, Domain
from src.graph_construction.feature_extractor import FeatureExtractor
from src.graph_construction.crg_builder import build_crg
from src.proactive.vulnerability_predictor import VulnerabilityPredictor


@dataclass
class StreamState:
    """State of a streaming reasoning chain."""
    chain_id: str
    domain: Domain
    query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    step_count: int = 0
    cached_features: Optional[torch.Tensor] = None
    vulnerability_history: List[Dict[str, Any]] = field(default_factory=list)


class StreamingAnalyzer:
    """
    Real-time streaming analysis of reasoning chains.
    
    Maintains state for multiple concurrent chains and updates predictions
    incrementally as new steps arrive.
    """
    
    def __init__(
        self,
        vulnerability_predictor: VulnerabilityPredictor,
        feature_extractor: Optional[FeatureExtractor] = None,
        max_concurrent_streams: int = 100,
    ):
        """
        Initialize streaming analyzer.
        
        Args:
            vulnerability_predictor: Vulnerability predictor instance
            feature_extractor: Feature extractor for new steps
            max_concurrent_streams: Maximum number of concurrent streams
        """
        self.vulnerability_predictor = vulnerability_predictor
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.max_concurrent_streams = max_concurrent_streams
        
        # Active streams
        self.active_streams: Dict[str, StreamState] = {}
    
    def start_stream(
        self,
        chain_id: str,
        domain: Domain,
        query: str,
    ) -> str:
        """
        Start a new reasoning chain stream.
        
        Args:
            chain_id: Unique identifier for the chain
            domain: Domain of the reasoning task
            query: Query/problem statement
            
        Returns:
            Chain ID
        """
        if len(self.active_streams) >= self.max_concurrent_streams:
            # Remove oldest stream
            oldest_id = list(self.active_streams.keys())[0]
            del self.active_streams[oldest_id]
        
        state = StreamState(
            chain_id=chain_id,
            domain=domain,
            query=query,
        )
        
        self.active_streams[chain_id] = state
        return chain_id
    
    def add_step(
        self,
        chain_id: str,
        step_text: str,
        step_id: Optional[int] = None,
        depends_on: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Add a new step to a streaming chain and analyze vulnerability.
        
        Args:
            chain_id: Chain identifier
            step_text: Text of the new step
            step_id: Optional step ID (auto-assigned if not provided)
            depends_on: Optional list of dependency step IDs
            
        Returns:
            Vulnerability analysis for the updated chain
        """
        if chain_id not in self.active_streams:
            raise ValueError(f"Chain {chain_id} not found in active streams")
        
        state = self.active_streams[chain_id]
        
        # Create new step
        if step_id is None:
            step_id = state.step_count
        
        if depends_on is None:
            # Default to depending on previous step
            depends_on = [step_id - 1] if step_id > 0 else []
        
        new_step = ReasoningStep(
            step_id=step_id,
            text=step_text,
            is_correct=True,  # Unknown at this point
            is_origin=False,
            error_type=None,
            depends_on=depends_on,
        )
        
        state.steps.append(new_step)
        state.step_count += 1
        
        # Build partial chain
        partial_chain = self._build_partial_chain(state)
        
        # Predict vulnerability
        vulnerability_result = self.vulnerability_predictor.predict_vulnerability(partial_chain)
        
        # Store in history
        state.vulnerability_history.append({
            "step_count": state.step_count,
            "result": vulnerability_result,
        })
        
        return vulnerability_result
    
    def _build_partial_chain(self, state: StreamState) -> ReasoningChain:
        """
        Build a ReasoningChain from current stream state.
        
        Args:
            state: Stream state
            
        Returns:
            Partial reasoning chain
        """
        # Build dependency graph
        nodes = list(range(len(state.steps)))
        edges = []
        for step in state.steps:
            for dep_id in step.depends_on:
                if dep_id < len(state.steps):
                    edges.append([dep_id, step.step_id])
        
        dependency_graph = DependencyGraph(nodes=nodes, edges=edges)
        
        chain = ReasoningChain(
            domain=state.domain,
            query_id=state.chain_id,
            query=state.query,
            ground_truth="",  # Unknown
            reasoning_steps=state.steps,
            dependency_graph=dependency_graph,
        )
        
        return chain
    
    def get_stream_status(self, chain_id: str) -> Dict[str, Any]:
        """
        Get current status of a stream.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Status dictionary
        """
        if chain_id not in self.active_streams:
            return {"status": "not_found"}
        
        state = self.active_streams[chain_id]
        
        # Get latest vulnerability result
        latest_result = None
        if state.vulnerability_history:
            latest_result = state.vulnerability_history[-1]["result"]
        
        return {
            "status": "active",
            "chain_id": chain_id,
            "domain": state.domain.value,
            "step_count": state.step_count,
            "latest_vulnerability": latest_result,
            "history_length": len(state.vulnerability_history),
        }
    
    def end_stream(self, chain_id: str) -> Dict[str, Any]:
        """
        End a stream and return final analysis.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Final analysis results
        """
        if chain_id not in self.active_streams:
            return {"status": "not_found"}
        
        state = self.active_streams[chain_id]
        
        # Build final chain
        final_chain = self._build_partial_chain(state)
        
        # Get final vulnerability prediction
        final_result = self.vulnerability_predictor.predict_vulnerability(final_chain)
        
        # Compile summary
        summary = {
            "chain_id": chain_id,
            "domain": state.domain.value,
            "total_steps": state.step_count,
            "final_vulnerability": final_result,
            "vulnerability_history": state.vulnerability_history,
            "final_chain": final_chain.to_dict(),
        }
        
        # Remove from active streams
        del self.active_streams[chain_id]
        
        return summary
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs."""
        return list(self.active_streams.keys())
    
    def clear_all_streams(self):
        """Clear all active streams."""
        self.active_streams.clear()


if __name__ == "__main__":
    # Example usage
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    from src.proactive.vulnerability_predictor import load_vulnerability_predictor
    
    print("Testing StreamingAnalyzer...")
    
    project_root = Path(__file__).parent.parent.parent
    checkpoint_path = project_root / "models/checkpoints/test_run/best_model.pth"
    
    if checkpoint_path.exists():
        # Load predictor
        predictor = load_vulnerability_predictor(
            checkpoint_path=checkpoint_path,
            model_class=ConfidenceGatedGAT,
            model_kwargs={
                "input_dim": 384 + 5 + 6,
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "dropout": 0.1,
                "use_confidence_gating": True,
            },
        )
        
        # Create streaming analyzer
        analyzer = StreamingAnalyzer(
            vulnerability_predictor=predictor,
            max_concurrent_streams=10,
        )
        
        # Simulate streaming
        print("\nSimulating streaming reasoning chain...")
        chain_id = analyzer.start_stream(
            chain_id="test_chain_1",
            domain=Domain.MATH,
            query="Solve: 2x + 3 = 11",
        )
        print(f"Started stream: {chain_id}")
        
        # Add steps incrementally
        steps = [
            "Subtract 3 from both sides",
            "2x = 8",
            "Divide both sides by 2",
            "x = 4",
        ]
        
        for i, step_text in enumerate(steps):
            print(f"\nAdding step {i+1}: {step_text}")
            result = analyzer.add_step(chain_id, step_text)
            print(f"  Risk Score: {result['overall_risk_score']:.3f}")
            print(f"  Risk Level: {result['overall_risk_level']}")
            print(f"  Vulnerable Steps: {result['vulnerable_steps']}")
        
        # Get status
        status = analyzer.get_stream_status(chain_id)
        print(f"\nStream Status:")
        print(f"  Chain ID: {status['chain_id']}")
        print(f"  Steps: {status['step_count']}")
        print(f"  History Length: {status['history_length']}")
        
        # End stream
        summary = analyzer.end_stream(chain_id)
        print(f"\nStream ended. Total steps: {summary['total_steps']}")
        
        print("\nStreamingAnalyzer test passed!")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first")

