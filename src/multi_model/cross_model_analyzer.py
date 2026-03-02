"""
Cross-Model Analysis Tools.

Provides comprehensive analysis and comparison of hallucination patterns
across different LLM models.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import Domain, ReasoningChain
from src.multi_model.model_config import ModelType
from src.multi_model.pattern_database import PatternDatabase, HallucinationType
from src.multi_model.consensus_detector import ConsensusResult


class CrossModelAnalyzer:
    """Analyze and compare hallucination patterns across models."""
    
    def __init__(self, pattern_db: Optional[PatternDatabase] = None):
        """
        Initialize cross-model analyzer.
        
        Args:
            pattern_db: Optional pattern database
        """
        self.pattern_db = pattern_db or PatternDatabase()
        self.analysis_cache = {}
    
    def compute_hallucination_rate(
        self,
        chains_by_model: Dict[ModelType, List[ReasoningChain]],
    ) -> Dict[str, float]:
        """
        Compute hallucination rate for each model.
        
        Args:
            chains_by_model: Dictionary mapping models to their chains
            
        Returns:
            Dictionary mapping model names to hallucination rates
        """
        rates = {}
        
        for model_type, chains in chains_by_model.items():
            total_steps = 0
            incorrect_steps = 0
            
            for chain in chains:
                for step in chain.reasoning_steps:
                    total_steps += 1
                    if not step.is_correct:
                        incorrect_steps += 1
            
            rate = incorrect_steps / total_steps if total_steps > 0 else 0.0
            rates[model_type.value] = rate
        
        return rates
    
    def analyze_agreement_patterns(
        self,
        consensus_results: List[ConsensusResult],
    ) -> Dict[str, Any]:
        """
        Analyze agreement patterns across models.
        
        Args:
            consensus_results: List of consensus results
            
        Returns:
            Analysis dictionary
        """
        if not consensus_results:
            return {"error": "No consensus results provided"}
        
        # Overall statistics
        total_chains = len(consensus_results)
        chains_with_consensus = sum(1 for r in consensus_results if r.consensus_exists)
        
        # Aggregate agreement rates
        all_agreement_rates = []
        for result in consensus_results:
            all_agreement_rates.extend(result.step_agreement_rates)
        
        avg_agreement = np.mean(all_agreement_rates) if all_agreement_rates else 0.0
        
        # Disagreement analysis
        total_disagreements = sum(len(r.disagreement_points) for r in consensus_results)
        total_steps = sum(len(r.final_prediction) for r in consensus_results)
        disagreement_rate = total_disagreements / total_steps if total_steps > 0 else 0.0
        
        return {
            "total_chains": total_chains,
            "chains_with_consensus": chains_with_consensus,
            "consensus_rate": chains_with_consensus / total_chains if total_chains > 0 else 0.0,
            "avg_agreement_rate": float(avg_agreement),
            "total_disagreements": total_disagreements,
            "disagreement_rate": float(disagreement_rate),
            "agreement_distribution": {
                "mean": float(np.mean(all_agreement_rates)),
                "std": float(np.std(all_agreement_rates)),
                "min": float(np.min(all_agreement_rates)),
                "max": float(np.max(all_agreement_rates)),
            }
        }
    
    def compare_domain_performance(
        self,
        chains_by_model: Dict[ModelType, List[ReasoningChain]],
    ) -> Dict[str, Any]:
        """
        Compare model performance across domains.
        
        Args:
            chains_by_model: Dictionary mapping models to their chains
            
        Returns:
            Domain comparison dictionary
        """
        performance = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
        
        for model_type, chains in chains_by_model.items():
            for chain in chains:
                domain = chain.domain.value
                for step in chain.reasoning_steps:
                    performance[model_type.value][domain]["total"] += 1
                    if step.is_correct:
                        performance[model_type.value][domain]["correct"] += 1
        
        # Convert to rates
        domain_comparison = {}
        for model, domains in performance.items():
            domain_comparison[model] = {}
            for domain, stats in domains.items():
                accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                domain_comparison[model][domain] = {
                    "accuracy": accuracy,
                    "total_steps": stats["total"],
                }
        
        return dict(domain_comparison)
    
    def analyze_error_types(
        self,
        chains_by_model: Dict[ModelType, List[ReasoningChain]],
    ) -> Dict[str, Any]:
        """
        Analyze error type distribution across models.
        
        Args:
            chains_by_model: Dictionary mapping models to their chains
            
        Returns:
            Error type analysis
        """
        error_types = defaultdict(lambda: defaultdict(int))
        
        for model_type, chains in chains_by_model.items():
            for chain in chains:
                for step in chain.reasoning_steps:
                    if not step.is_correct and step.error_type:
                        error_types[model_type.value][step.error_type.value] += 1
        
        # Convert to percentages
        error_analysis = {}
        for model, types in error_types.items():
            total = sum(types.values())
            error_analysis[model] = {
                error_type: {
                    "count": count,
                    "percentage": (count / total * 100) if total > 0 else 0.0
                }
                for error_type, count in types.items()
            }
        
        return dict(error_analysis)
    
    def compute_model_similarity(
        self,
        chains_by_model: Dict[ModelType, List[ReasoningChain]],
    ) -> Dict[str, float]:
        """
        Compute similarity between models based on error patterns.
        
        Args:
            chains_by_model: Dictionary mapping models to their chains
            
        Returns:
            Dictionary of model pair similarities
        """
        model_types = list(chains_by_model.keys())
        similarities = {}
        
        for i, model1 in enumerate(model_types):
            for model2 in model_types[i+1:]:
                # Compare error rates per domain
                similarity = self._compute_pair_similarity(
                    chains_by_model[model1],
                    chains_by_model[model2]
                )
                
                pair_key = f"{model1.value}_vs_{model2.value}"
                similarities[pair_key] = similarity
        
        return similarities
    
    def _compute_pair_similarity(
        self,
        chains1: List[ReasoningChain],
        chains2: List[ReasoningChain],
    ) -> float:
        """Compute similarity between two sets of chains."""
        # Group by domain
        domain_rates1 = defaultdict(lambda: {"correct": 0, "total": 0})
        domain_rates2 = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for chain in chains1:
            for step in chain.reasoning_steps:
                domain_rates1[chain.domain.value]["total"] += 1
                if step.is_correct:
                    domain_rates1[chain.domain.value]["correct"] += 1
        
        for chain in chains2:
            for step in chain.reasoning_steps:
                domain_rates2[chain.domain.value]["total"] += 1
                if step.is_correct:
                    domain_rates2[chain.domain.value]["correct"] += 1
        
        # Compute accuracy similarity
        similarities = []
        all_domains = set(domain_rates1.keys()) | set(domain_rates2.keys())
        
        for domain in all_domains:
            acc1 = (domain_rates1[domain]["correct"] / domain_rates1[domain]["total"]
                   if domain_rates1[domain]["total"] > 0 else 0.0)
            acc2 = (domain_rates2[domain]["correct"] / domain_rates2[domain]["total"]
                   if domain_rates2[domain]["total"] > 0 else 0.0)
            
            # Similarity is 1 - absolute difference
            similarity = 1 - abs(acc1 - acc2)
            similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def generate_comparative_report(
        self,
        chains_by_model: Dict[ModelType, List[ReasoningChain]],
        consensus_results: Optional[List[ConsensusResult]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparative analysis report.
        
        Args:
            chains_by_model: Dictionary mapping models to their chains
            consensus_results: Optional consensus results
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            "models_analyzed": [mt.value for mt in chains_by_model.keys()],
            "total_chains": sum(len(chains) for chains in chains_by_model.values()),
        }
        
        # Hallucination rates
        report["hallucination_rates"] = self.compute_hallucination_rate(chains_by_model)
        
        # Domain performance
        report["domain_performance"] = self.compare_domain_performance(chains_by_model)
        
        # Error types
        report["error_type_analysis"] = self.analyze_error_types(chains_by_model)
        
        # Model similarity
        report["model_similarity"] = self.compute_model_similarity(chains_by_model)
        
        # Agreement patterns (if consensus results provided)
        if consensus_results:
            report["agreement_analysis"] = self.analyze_agreement_patterns(consensus_results)
        
        # Pattern database statistics (if available)
        if self.pattern_db.patterns:
            report["pattern_statistics"] = {}
            for model_type in chains_by_model.keys():
                profile = self.pattern_db.get_model_vulnerability_profile(model_type)
                report["pattern_statistics"][model_type.value] = profile
        
        # Rankings
        report["rankings"] = self._compute_rankings(report)
        
        return report
    
    def _compute_rankings(self, report: Dict[str, Any]) -> Dict[str, List[str]]:
        """Compute model rankings based on various metrics."""
        rankings = {}
        
        # Rank by hallucination rate (lower is better)
        hall_rates = report["hallucination_rates"]
        rankings["lowest_hallucination_rate"] = sorted(
            hall_rates.keys(),
            key=lambda m: hall_rates[m]
        )
        
        # Rank by vulnerability score (if available)
        if "pattern_statistics" in report:
            vuln_scores = {
                model: stats.get("vulnerability_score", 0)
                for model, stats in report["pattern_statistics"].items()
            }
            rankings["lowest_vulnerability"] = sorted(
                vuln_scores.keys(),
                key=lambda m: vuln_scores[m]
            )
        
        return rankings
    
    def export_report(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ):
        """
        Export report to JSON file.
        
        Args:
            report: Report dictionary
            output_path: Path to save report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Exported cross-model analysis report to {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the report."""
        print("=" * 80)
        print("CROSS-MODEL ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\nModels Analyzed: {', '.join(report['models_analyzed'])}")
        print(f"Total Chains: {report['total_chains']}")
        
        print("\nHallucination Rates:")
        for model, rate in report["hallucination_rates"].items():
            print(f"  {model}: {rate*100:.2f}%")
        
        if "rankings" in report:
            print("\nRankings (Best to Worst):")
            if "lowest_hallucination_rate" in report["rankings"]:
                print(f"  By Hallucination Rate: {' > '.join(report['rankings']['lowest_hallucination_rate'])}")
        
        if "agreement_analysis" in report:
            print("\nAgreement Analysis:")
            print(f"  Consensus Rate: {report['agreement_analysis']['consensus_rate']*100:.2f}%")
            print(f"  Avg Agreement: {report['agreement_analysis']['avg_agreement_rate']*100:.2f}%")
            print(f"  Disagreement Rate: {report['agreement_analysis']['disagreement_rate']*100:.2f}%")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test cross-model analyzer
    print("=" * 80)
    print("Cross-Model Analyzer Test")
    print("=" * 80)
    
    from src.data_processing.unified_schema import ReasoningStep, DependencyGraph, ErrorType
    
    # Create sample chains
    def create_test_chain(query_id: str, domain: Domain, errors: List[int]) -> ReasoningChain:
        steps = []
        for i in range(5):
            is_correct = i not in errors
            error_type = ErrorType.FACTUAL if not is_correct else None
            steps.append(ReasoningStep(
                step_id=i,
                text=f"Step {i}: Some reasoning text",
                is_correct=is_correct,
                is_origin=(i == errors[0] if errors else False),
                error_type=error_type,
                depends_on=[i-1] if i > 0 else [],
            ))
        
        return ReasoningChain(
            domain=domain,
            query_id=query_id,
            query="Test query",
            ground_truth="Test answer",
            reasoning_steps=steps,
            dependency_graph=DependencyGraph(
                nodes=list(range(5)),
                edges=[[i, i+1] for i in range(4)]
            )
        )
    
    # Create test data
    chains_by_model = {
        ModelType.GPT4: [
            create_test_chain("gpt4_1", Domain.MATH, [2]),
            create_test_chain("gpt4_2", Domain.MATH, []),
        ],
        ModelType.GEMINI: [
            create_test_chain("gemini_1", Domain.MATH, [2, 3]),
            create_test_chain("gemini_2", Domain.MATH, [1]),
        ],
    }
    
    # Create analyzer
    analyzer = CrossModelAnalyzer()
    
    # Compute hallucination rates
    print("\nHallucination Rates:")
    rates = analyzer.compute_hallucination_rate(chains_by_model)
    for model, rate in rates.items():
        print(f"  {model}: {rate*100:.2f}%")
    
    # Compute model similarity
    print("\nModel Similarity:")
    similarities = analyzer.compute_model_similarity(chains_by_model)
    for pair, sim in similarities.items():
        print(f"  {pair}: {sim:.3f}")
    
    # Generate full report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_comparative_report(chains_by_model)
    
    print(f"\nReport Summary:")
    print(f"  Models: {len(report['models_analyzed'])}")
    print(f"  Total Chains: {report['total_chains']}")
    print(f"  Rankings: {list(report['rankings'].keys())}")
    
    print("\nCross-model analyzer test passed!")

