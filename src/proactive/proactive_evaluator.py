"""
Proactive Evaluator.

Evaluates effectiveness of proactive prediction system.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import ReasoningChain
from src.proactive.streaming_inference import StreamingAnalyzer
from src.proactive.interventional_controller import InterventionalController


class ProactiveEvaluator:
    """Evaluate proactive prediction system."""
    
    def __init__(
        self,
        streaming_analyzer: StreamingAnalyzer,
        interventional_controller: InterventionalController,
    ):
        """
        Initialize evaluator.
        
        Args:
            streaming_analyzer: Streaming analyzer instance
            interventional_controller: Interventional controller instance
        """
        self.streaming_analyzer = streaming_analyzer
        self.interventional_controller = interventional_controller
    
    def evaluate_on_chains(
        self,
        chains: List[ReasoningChain],
    ) -> Dict[str, Any]:
        """
        Evaluate proactive system on full chains.
        
        Simulates streaming and measures:
        - Vulnerability detection accuracy
        - Intervention effectiveness
        - Time-to-detection
        
        Args:
            chains: List of full reasoning chains
            
        Returns:
            Evaluation metrics
        """
        metrics = {
            "total_chains": len(chains),
            "chains_with_errors": 0,
            "vulnerability_detected": 0,
            "interventions_triggered": 0,
            "early_detections": 0,  # Detected before error
            "time_to_detection": [],  # Steps before error
            "false_positives": 0,
            "false_negatives": 0,
        }
        
        for chain in chains:
            result = self._evaluate_single_chain(chain)
            
            # Aggregate metrics
            if result["has_error"]:
                metrics["chains_with_errors"] += 1
                
                if result["vulnerability_detected"]:
                    metrics["vulnerability_detected"] += 1
                    
                if result["detected_early"]:
                    metrics["early_detections"] += 1
                    metrics["time_to_detection"].append(result["steps_before_error"])
                
                if not result["vulnerability_detected"]:
                    metrics["false_negatives"] += 1
            else:
                # Correct chain - check for false positives
                if result["interventions_triggered"] > 0:
                    metrics["false_positives"] += 1
            
            metrics["interventions_triggered"] += result["interventions_triggered"]
        
        # Compute aggregate statistics
        if metrics["chains_with_errors"] > 0:
            metrics["detection_rate"] = metrics["vulnerability_detected"] / metrics["chains_with_errors"]
            metrics["early_detection_rate"] = metrics["early_detections"] / metrics["chains_with_errors"]
        else:
            metrics["detection_rate"] = 0
            metrics["early_detection_rate"] = 0
        
        if metrics["time_to_detection"]:
            metrics["avg_time_to_detection"] = np.mean(metrics["time_to_detection"])
        else:
            metrics["avg_time_to_detection"] = 0
        
        # False positive rate
        correct_chains = metrics["total_chains"] - metrics["chains_with_errors"]
        if correct_chains > 0:
            metrics["false_positive_rate"] = metrics["false_positives"] / correct_chains
        else:
            metrics["false_positive_rate"] = 0
        
        return metrics
    
    def _evaluate_single_chain(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Evaluate a single chain."""
        # Start stream
        chain_id = f"eval_{chain.query_id}"
        self.streaming_analyzer.start_stream(
            chain_id=chain_id,
            domain=chain.domain,
            query=chain.query,
        )
        
        # Find first error in chain
        first_error_step = -1
        for i, step in enumerate(chain.reasoning_steps):
            if not step.is_correct:
                first_error_step = i
                break
        
        has_error = first_error_step >= 0
        vulnerability_detected = False
        detected_early = False
        steps_before_error = -1
        interventions_triggered = 0
        
        # Stream steps one by one
        for i, step in enumerate(chain.reasoning_steps):
            # Add step
            vuln_result = self.streaming_analyzer.add_step(
                chain_id=chain_id,
                step_text=step.text,
                step_id=step.step_id,
                depends_on=step.depends_on,
            )
            
            # Check for interventions
            interventions = self.interventional_controller.evaluate_and_intervene(
                chain_id=chain_id,
                vulnerability_result=vuln_result,
            )
            
            interventions_triggered += len(interventions)
            
            # Check if vulnerability detected before error
            if has_error and not vulnerability_detected:
                if vuln_result["overall_risk_level"] in ["MEDIUM", "HIGH"]:
                    vulnerability_detected = True
                    
                    if i < first_error_step:
                        detected_early = True
                        steps_before_error = first_error_step - i
        
        # End stream
        self.streaming_analyzer.end_stream(chain_id)
        
        return {
            "has_error": has_error,
            "first_error_step": first_error_step,
            "vulnerability_detected": vulnerability_detected,
            "detected_early": detected_early,
            "steps_before_error": steps_before_error,
            "interventions_triggered": interventions_triggered,
        }


if __name__ == "__main__":
    print("Testing ProactiveEvaluator...")
    
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    from src.proactive.vulnerability_predictor import load_vulnerability_predictor
    from src.proactive.interventional_controller import CorrectionGenerator
    
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
        
        # Create components
        analyzer = StreamingAnalyzer(vulnerability_predictor=predictor)
        controller = InterventionalController(
            warn_threshold=0.3,
            correct_threshold=0.7,
            correction_generator=CorrectionGenerator(),
        )
        
        # Create evaluator
        evaluator = ProactiveEvaluator(
            streaming_analyzer=analyzer,
            interventional_controller=controller,
        )
        
        # Load test chains
        test_path = project_root / "data/processed/splits/test.jsonl"
        if test_path.exists():
            import json
            chains = []
            with open(test_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Test on 10 chains
                        break
                    chain_dict = json.loads(line)
                    chain = ReasoningChain.from_dict(chain_dict)
                    chains.append(chain)
            
            print(f"\nEvaluating on {len(chains)} test chains...")
            metrics = evaluator.evaluate_on_chains(chains)
            
            print(f"\nProactive System Metrics:")
            print(f"  Total Chains: {metrics['total_chains']}")
            print(f"  Chains with Errors: {metrics['chains_with_errors']}")
            print(f"  Detection Rate: {metrics['detection_rate']*100:.2f}%")
            print(f"  Early Detection Rate: {metrics['early_detection_rate']*100:.2f}%")
            print(f"  Avg Time-to-Detection: {metrics['avg_time_to_detection']:.2f} steps")
            print(f"  False Positive Rate: {metrics['false_positive_rate']*100:.2f}%")
            print(f"  Interventions Triggered: {metrics['interventions_triggered']}")
            
            print("\nProactiveEvaluator test passed!")
        else:
            print(f"Test data not found: {test_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

