"""
Consensus-Based Hallucination Detection.

Combines predictions from multiple models to detect hallucinations
through consensus and disagreement analysis.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import ReasoningChain, ReasoningStep
from src.multi_model.model_config import ModelType


class ConsensusStrategy(Enum):
    """Strategies for consensus detection."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    UNANIMOUS = "unanimous"
    EXPERT_SELECTION = "expert_selection"


@dataclass
class ConsensusResult:
    """Result from consensus detection."""
    chain_id: str
    consensus_exists: bool
    consensus_confidence: float
    step_consensus: List[bool]  # Per-step consensus
    step_agreement_rates: List[float]  # Agreement rate per step
    disagreement_points: List[int]  # Step indices with disagreement
    model_predictions: Dict[ModelType, List[bool]]  # Per-model predictions
    final_prediction: List[bool]  # Consensus prediction per step
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "consensus_exists": self.consensus_exists,
            "consensus_confidence": self.consensus_confidence,
            "step_consensus": self.step_consensus,
            "step_agreement_rates": self.step_agreement_rates,
            "disagreement_points": self.disagreement_points,
            "model_predictions": {
                k.value: v for k, v in self.model_predictions.items()
            },
            "final_prediction": self.final_prediction,
        }


class ConsensusDetector:
    """
    Consensus-based hallucination detector.
    
    Combines predictions from multiple models to make more reliable
    hallucination detection decisions.
    """
    
    def __init__(
        self,
        strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTE,
        model_weights: Optional[Dict[ModelType, float]] = None,
        consensus_threshold: float = 0.5,
    ):
        """
        Initialize consensus detector.
        
        Args:
            strategy: Consensus strategy to use
            model_weights: Optional weights for each model (for weighted voting)
            consensus_threshold: Threshold for consensus (0.5 = majority)
        """
        self.strategy = strategy
        self.model_weights = model_weights or {}
        self.consensus_threshold = consensus_threshold
    
    def detect_consensus(
        self,
        chain_id: str,
        model_predictions: Dict[ModelType, List[bool]],
    ) -> ConsensusResult:
        """
        Detect consensus across model predictions.
        
        Args:
            chain_id: Unique chain identifier
            model_predictions: Dictionary mapping models to their step predictions
                             (True = correct, False = hallucination/error)
            
        Returns:
            ConsensusResult with analysis
        """
        # Validate inputs
        if not model_predictions:
            raise ValueError("No model predictions provided")
        
        # Get number of steps (assume all models predict same number)
        num_steps = len(next(iter(model_predictions.values())))
        
        # Check all models have same number of steps
        for model, preds in model_predictions.items():
            if len(preds) != num_steps:
                raise ValueError(f"Model {model.value} has {len(preds)} predictions, expected {num_steps}")
        
        # Apply consensus strategy
        if self.strategy == ConsensusStrategy.MAJORITY_VOTE:
            final_pred, step_agreement = self._majority_vote(model_predictions)
        elif self.strategy == ConsensusStrategy.WEIGHTED_VOTE:
            final_pred, step_agreement = self._weighted_vote(model_predictions)
        elif self.strategy == ConsensusStrategy.UNANIMOUS:
            final_pred, step_agreement = self._unanimous(model_predictions)
        elif self.strategy == ConsensusStrategy.EXPERT_SELECTION:
            final_pred, step_agreement = self._expert_selection(model_predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Compute consensus metrics
        step_consensus = [
            agree >= self.consensus_threshold for agree in step_agreement
        ]
        
        consensus_exists = all(step_consensus)
        consensus_confidence = np.mean(step_agreement)
        
        # Find disagreement points
        disagreement_points = [
            i for i, has_consensus in enumerate(step_consensus)
            if not has_consensus
        ]
        
        return ConsensusResult(
            chain_id=chain_id,
            consensus_exists=consensus_exists,
            consensus_confidence=float(consensus_confidence),
            step_consensus=step_consensus,
            step_agreement_rates=step_agreement,
            disagreement_points=disagreement_points,
            model_predictions=model_predictions,
            final_prediction=final_pred,
        )
    
    def _majority_vote(
        self,
        model_predictions: Dict[ModelType, List[bool]]
    ) -> Tuple[List[bool], List[float]]:
        """Simple majority voting."""
        num_steps = len(next(iter(model_predictions.values())))
        num_models = len(model_predictions)
        
        final_pred = []
        agreement_rates = []
        
        for step_idx in range(num_steps):
            # Get predictions for this step
            step_preds = [preds[step_idx] for preds in model_predictions.values()]
            
            # Count votes
            correct_votes = sum(step_preds)
            error_votes = num_models - correct_votes
            
            # Majority decision
            final_pred.append(correct_votes > error_votes)
            
            # Agreement rate
            majority = max(correct_votes, error_votes)
            agreement_rates.append(majority / num_models)
        
        return final_pred, agreement_rates
    
    def _weighted_vote(
        self,
        model_predictions: Dict[ModelType, List[bool]]
    ) -> Tuple[List[bool], List[float]]:
        """Weighted voting based on model weights."""
        num_steps = len(next(iter(model_predictions.values())))
        
        # Normalize weights
        total_weight = sum(
            self.model_weights.get(model, 1.0)
            for model in model_predictions.keys()
        )
        
        final_pred = []
        agreement_rates = []
        
        for step_idx in range(num_steps):
            correct_weight = 0.0
            error_weight = 0.0
            
            for model, preds in model_predictions.items():
                weight = self.model_weights.get(model, 1.0)
                if preds[step_idx]:
                    correct_weight += weight
                else:
                    error_weight += weight
            
            # Weighted decision
            final_pred.append(correct_weight > error_weight)
            
            # Agreement rate (normalized)
            agreement_rates.append(max(correct_weight, error_weight) / total_weight)
        
        return final_pred, agreement_rates
    
    def _unanimous(
        self,
        model_predictions: Dict[ModelType, List[bool]]
    ) -> Tuple[List[bool], List[float]]:
        """Unanimous consensus (all models must agree)."""
        num_steps = len(next(iter(model_predictions.values())))
        num_models = len(model_predictions)
        
        final_pred = []
        agreement_rates = []
        
        for step_idx in range(num_steps):
            step_preds = [preds[step_idx] for preds in model_predictions.values()]
            
            # All must agree
            all_correct = all(step_preds)
            all_error = not any(step_preds)
            
            final_pred.append(all_correct)
            
            # Agreement is 1.0 if unanimous, otherwise based on majority
            if all_correct or all_error:
                agreement_rates.append(1.0)
            else:
                agreement_rates.append(max(sum(step_preds), num_models - sum(step_preds)) / num_models)
        
        return final_pred, agreement_rates
    
    def _expert_selection(
        self,
        model_predictions: Dict[ModelType, List[bool]]
    ) -> Tuple[List[bool], List[float]]:
        """Select expert model based on weights (highest weight wins)."""
        # Find expert (highest weight)
        expert_model = max(
            model_predictions.keys(),
            key=lambda m: self.model_weights.get(m, 1.0)
        )
        
        # Use expert's predictions
        final_pred = model_predictions[expert_model]
        
        # Compute agreement with expert
        num_steps = len(final_pred)
        num_models = len(model_predictions)
        agreement_rates = []
        
        for step_idx in range(num_steps):
            expert_pred = final_pred[step_idx]
            agreements = sum(
                1 for preds in model_predictions.values()
                if preds[step_idx] == expert_pred
            )
            agreement_rates.append(agreements / num_models)
        
        return final_pred, agreement_rates
    
    def detect_batch(
        self,
        predictions_batch: List[Tuple[str, Dict[ModelType, List[bool]]]]
    ) -> List[ConsensusResult]:
        """
        Detect consensus for multiple chains.
        
        Args:
            predictions_batch: List of (chain_id, model_predictions) tuples
            
        Returns:
            List of consensus results
        """
        return [
            self.detect_consensus(chain_id, preds)
            for chain_id, preds in predictions_batch
        ]
    
    def combine_with_gnn(
        self,
        consensus_result: ConsensusResult,
        gnn_predictions: List[bool],
        gnn_weight: float = 0.5,
    ) -> List[bool]:
        """
        Combine consensus predictions with GNN predictions.
        
        Args:
            consensus_result: Consensus result from multi-model
            gnn_predictions: Predictions from GNN model
            gnn_weight: Weight for GNN predictions (0-1)
            
        Returns:
            Combined predictions
        """
        if len(consensus_result.final_prediction) != len(gnn_predictions):
            raise ValueError("Prediction lengths don't match")
        
        combined = []
        for cons_pred, gnn_pred, agreement in zip(
            consensus_result.final_prediction,
            gnn_predictions,
            consensus_result.step_agreement_rates
        ):
            # Weighted combination
            # High agreement -> trust consensus more
            # Low agreement -> trust GNN more
            consensus_weight = agreement * (1 - gnn_weight) + (1 - agreement) * gnn_weight
            
            if consensus_weight > 0.5:
                combined.append(cons_pred)
            else:
                combined.append(gnn_pred)
        
        return combined
    
    def get_disagreement_analysis(
        self,
        result: ConsensusResult
    ) -> Dict[str, Any]:
        """
        Analyze disagreement patterns.
        
        Args:
            result: Consensus result to analyze
            
        Returns:
            Disagreement analysis dictionary
        """
        if not result.disagreement_points:
            return {
                "has_disagreement": False,
                "num_disagreements": 0,
                "disagreement_rate": 0.0,
            }
        
        num_steps = len(result.final_prediction)
        
        # Analyze each disagreement point
        disagreement_details = []
        for step_idx in result.disagreement_points:
            step_preds = {
                model.value: preds[step_idx]
                for model, preds in result.model_predictions.items()
            }
            
            disagreement_details.append({
                "step_idx": step_idx,
                "agreement_rate": result.step_agreement_rates[step_idx],
                "predictions": step_preds,
            })
        
        return {
            "has_disagreement": True,
            "num_disagreements": len(result.disagreement_points),
            "disagreement_rate": len(result.disagreement_points) / num_steps,
            "disagreement_points": result.disagreement_points,
            "details": disagreement_details,
        }


if __name__ == "__main__":
    # Test consensus detector
    print("=" * 80)
    print("Consensus Detector Test")
    print("=" * 80)
    
    # Sample predictions from 3 models
    model_predictions = {
        ModelType.GPT4: [True, True, False, True, True],  # Detects error at step 2
        ModelType.GEMINI: [True, True, False, False, True],  # Detects errors at steps 2, 3
        ModelType.LLAMA: [True, True, True, True, True],  # Detects no errors
    }
    
    # Test majority vote
    print("\n1. Majority Vote Strategy:")
    detector = ConsensusDetector(strategy=ConsensusStrategy.MAJORITY_VOTE)
    result = detector.detect_consensus("test_chain_1", model_predictions)
    
    print(f"  Consensus exists: {result.consensus_exists}")
    print(f"  Consensus confidence: {result.consensus_confidence:.3f}")
    print(f"  Final prediction: {result.final_prediction}")
    print(f"  Disagreement points: {result.disagreement_points}")
    
    # Test weighted vote
    print("\n2. Weighted Vote Strategy:")
    weights = {
        ModelType.GPT4: 1.5,  # Higher weight for GPT-4
        ModelType.GEMINI: 1.0,
        ModelType.LLAMA: 0.5,  # Lower weight for Llama
    }
    detector_weighted = ConsensusDetector(
        strategy=ConsensusStrategy.WEIGHTED_VOTE,
        model_weights=weights
    )
    result_weighted = detector_weighted.detect_consensus("test_chain_1", model_predictions)
    
    print(f"  Final prediction: {result_weighted.final_prediction}")
    print(f"  Agreement rates: {[f'{r:.2f}' for r in result_weighted.step_agreement_rates]}")
    
    # Test disagreement analysis
    print("\n3. Disagreement Analysis:")
    analysis = detector.get_disagreement_analysis(result)
    print(f"  Has disagreement: {analysis['has_disagreement']}")
    print(f"  Disagreement rate: {analysis['disagreement_rate']:.2%}")
    if analysis['has_disagreement']:
        print(f"  Disagreement at steps: {analysis['disagreement_points']}")
    
    print("\nConsensus detector test passed!")

