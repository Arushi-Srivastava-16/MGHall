"""
Interventional Control System.

Triggers interventions and suggests corrections at vulnerable steps.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import ReasoningStep
from src.proactive.vulnerability_predictor import RiskLevel


class InterventionType(Enum):
    """Types of interventions."""
    NONE = "none"
    WARNING = "warning"
    CORRECTION = "correction"


@dataclass
class Intervention:
    """An intervention record."""
    chain_id: str
    step_id: int
    timestamp: str
    intervention_type: InterventionType
    risk_level: str
    risk_score: float
    warning_message: Optional[str] = None
    correction_suggestion: Optional[str] = None


class InterventionalController:
    """
    Control system for triggering interventions.
    
    Monitors vulnerability predictions and triggers appropriate
    interventions (warnings or corrections) based on risk levels.
    """
    
    def __init__(
        self,
        warn_threshold: float = 0.3,
        correct_threshold: float = 0.7,
        enable_warnings: bool = True,
        enable_corrections: bool = True,
        correction_generator: Optional['CorrectionGenerator'] = None,
    ):
        """
        Initialize interventional controller.
        
        Args:
            warn_threshold: Risk score threshold for warnings
            correct_threshold: Risk score threshold for corrections
            enable_warnings: Whether to enable warning interventions
            enable_corrections: Whether to enable correction interventions
            correction_generator: Optional correction generator
        """
        self.warn_threshold = warn_threshold
        self.correct_threshold = correct_threshold
        self.enable_warnings = enable_warnings
        self.enable_corrections = enable_corrections
        self.correction_generator = correction_generator
        
        # Intervention history
        self.interventions: List[Intervention] = []
    
    def evaluate_and_intervene(
        self,
        chain_id: str,
        vulnerability_result: Dict[str, Any],
    ) -> List[Intervention]:
        """
        Evaluate vulnerability result and trigger interventions.
        
        Args:
            chain_id: Chain identifier
            vulnerability_result: Result from vulnerability predictor
            
        Returns:
            List of interventions triggered
        """
        interventions = []
        
        overall_risk = vulnerability_result['overall_risk_score']
        step_vulnerabilities = vulnerability_result['step_vulnerabilities']
        step_risk_levels = vulnerability_result['step_risk_levels']
        
        # Check each step
        for step_id, (score, level) in enumerate(zip(step_vulnerabilities, step_risk_levels)):
            intervention = self._decide_intervention(
                chain_id=chain_id,
                step_id=step_id,
                risk_score=score,
                risk_level=level,
            )
            
            if intervention:
                interventions.append(intervention)
                self.interventions.append(intervention)
        
        return interventions
    
    def _decide_intervention(
        self,
        chain_id: str,
        step_id: int,
        risk_score: float,
        risk_level: str,
    ) -> Optional[Intervention]:
        """
        Decide what intervention to trigger.
        
        Args:
            chain_id: Chain identifier
            step_id: Step identifier
            risk_score: Vulnerability score
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            
        Returns:
            Intervention object or None
        """
        # HIGH risk: Correction
        if risk_score >= self.correct_threshold and self.enable_corrections:
            correction = None
            if self.correction_generator:
                correction = self.correction_generator.generate_correction(step_id)
            else:
                correction = f"HIGH RISK: Step {step_id} requires careful review and potential revision"
            
            return Intervention(
                chain_id=chain_id,
                step_id=step_id,
                timestamp=datetime.now().isoformat(),
                intervention_type=InterventionType.CORRECTION,
                risk_level=risk_level,
                risk_score=risk_score,
                correction_suggestion=correction,
            )
        
        # MEDIUM risk: Warning
        elif risk_score >= self.warn_threshold and self.enable_warnings:
            warning = f"CAUTION: Step {step_id} shows moderate vulnerability (risk: {risk_score:.2f})"
            
            return Intervention(
                chain_id=chain_id,
                step_id=step_id,
                timestamp=datetime.now().isoformat(),
                intervention_type=InterventionType.WARNING,
                risk_level=risk_level,
                risk_score=risk_score,
                warning_message=warning,
            )
        
        # LOW risk: No intervention
        return None
    
    def get_intervention_history(
        self,
        chain_id: Optional[str] = None,
    ) -> List[Intervention]:
        """
        Get intervention history.
        
        Args:
            chain_id: Optional filter by chain ID
            
        Returns:
            List of interventions
        """
        if chain_id:
            return [i for i in self.interventions if i.chain_id == chain_id]
        return self.interventions
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get intervention statistics.
        
        Returns:
            Statistics dictionary
        """
        total = len(self.interventions)
        warnings = sum(1 for i in self.interventions if i.intervention_type == InterventionType.WARNING)
        corrections = sum(1 for i in self.interventions if i.intervention_type == InterventionType.CORRECTION)
        
        # Count by risk level
        risk_counts = {
            "LOW": 0,
            "MEDIUM": sum(1 for i in self.interventions if i.risk_level == "MEDIUM"),
            "HIGH": sum(1 for i in self.interventions if i.risk_level == "HIGH"),
        }
        
        return {
            "total_interventions": total,
            "warnings": warnings,
            "corrections": corrections,
            "risk_level_counts": risk_counts,
            "unique_chains": len(set(i.chain_id for i in self.interventions)),
        }


class CorrectionGenerator:
    """
    Generates correction suggestions for vulnerable steps.
    
    This is a simple rule-based generator. Can be extended with LLM integration.
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None,
    ):
        """
        Initialize correction generator.
        
        Args:
            use_llm: Whether to use LLM for correction generation
            llm_api_key: API key for LLM service
        """
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key
    
    def generate_correction(
        self,
        step_id: int,
        step_text: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> str:
        """
        Generate a correction suggestion.
        
        Args:
            step_id: Step identifier
            step_text: Optional step text
            context: Optional context dictionary
            
        Returns:
            Correction suggestion text
        """
        if self.use_llm and self.llm_api_key:
            # TODO: Implement LLM-based correction
            return self._generate_llm_correction(step_id, step_text, context)
        else:
            # Rule-based correction
            return self._generate_rule_based_correction(step_id, step_text)
    
    def _generate_llm_correction(
        self,
        step_id: int,
        step_text: Optional[str],
        context: Optional[Dict],
    ) -> str:
        """Generate correction using LLM (placeholder)."""
        # TODO: Integrate with OpenAI/Anthropic API
        return f"LLM-based correction for step {step_id}: [Requires API integration]"
    
    def _generate_rule_based_correction(
        self,
        step_id: int,
        step_text: Optional[str],
    ) -> str:
        """Generate simple rule-based correction."""
        suggestions = [
            "Double-check calculations and verify with the previous step",
            "Review the logical flow and ensure assumptions are valid",
            "Verify that all dependencies are correctly satisfied",
            "Consider alternative approaches to this reasoning step",
            "Cross-reference with domain knowledge and best practices",
        ]
        
        # Rotate through suggestions
        suggestion = suggestions[step_id % len(suggestions)]
        
        return f"Step {step_id}: {suggestion}"


if __name__ == "__main__":
    print("Testing InterventionalController...")
    
    # Create controller
    controller = InterventionalController(
        warn_threshold=0.3,
        correct_threshold=0.7,
        enable_warnings=True,
        enable_corrections=True,
        correction_generator=CorrectionGenerator(),
    )
    
    # Simulate vulnerability results
    vulnerability_result = {
        "overall_risk_score": 0.8,
        "overall_risk_level": "HIGH",
        "step_vulnerabilities": [0.2, 0.5, 0.8, 0.3],
        "step_risk_levels": ["LOW", "MEDIUM", "HIGH", "MEDIUM"],
        "vulnerable_steps": [1, 2, 3],
        "num_steps": 4,
    }
    
    print("\nEvaluating vulnerability result...")
    interventions = controller.evaluate_and_intervene(
        chain_id="test_chain",
        vulnerability_result=vulnerability_result,
    )
    
    print(f"\nTriggered {len(interventions)} interventions:")
    for intervention in interventions:
        print(f"\n  Step {intervention.step_id}:")
        print(f"    Type: {intervention.intervention_type.value}")
        print(f"    Risk: {intervention.risk_level} ({intervention.risk_score:.2f})")
        if intervention.warning_message:
            print(f"    Warning: {intervention.warning_message}")
        if intervention.correction_suggestion:
            print(f"    Correction: {intervention.correction_suggestion}")
    
    # Get statistics
    stats = controller.get_statistics()
    print(f"\nIntervention Statistics:")
    print(f"  Total: {stats['total_interventions']}")
    print(f"  Warnings: {stats['warnings']}")
    print(f"  Corrections: {stats['corrections']}")
    
    print("\nInterventionalController test passed!")

