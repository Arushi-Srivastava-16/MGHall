"""
Consensus Evaluation.

Evaluates consensus-based hallucination detection across multiple models.
"""

import sys
from pathlib import Path
import json
import argparse
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.unified_schema import ReasoningChain
from src.multi_model.model_config import ModelType
from src.multi_model.consensus_detector import ConsensusDetector, ConsensusStrategy
from src.multi_model.cross_model_analyzer import CrossModelAnalyzer


def load_chains(file_path: Path):
    """Load chains from JSONL file."""
    chains = []
    with open(file_path, 'r') as f:
        for line in f:
            chain_dict = json.loads(line)
            chain = ReasoningChain.from_dict(chain_dict)
            chains.append(chain)
    return chains


def extract_predictions(chain: ReasoningChain):
    """Extract step correctness predictions from chain."""
    return [step.is_correct for step in chain.reasoning_steps]


def main():
    parser = argparse.ArgumentParser(description="Evaluate consensus detection")
    parser.add_argument("--domain", type=str, default="math")
    parser.add_argument("--strategy", type=str, default="majority_vote",
                       choices=["majority_vote", "weighted_vote", "unanimous"])
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CONSENSUS EVALUATION")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    chains_dir = project_root / "data/multi_model/generated_chains"
    
    # Load chains from all models
    print(f"\nLoading chains for domain: {args.domain}")
    chains_by_model = {}
    
    for file_path in chains_dir.glob(f"*_{args.domain}_chains.jsonl"):
        model_name = file_path.stem.split('_')[0]
        try:
            model_type = ModelType(model_name)
            chains = load_chains(file_path)
            chains_by_model[model_type] = chains
            print(f"  {model_name}: {len(chains)} chains")
        except Exception as e:
            print(f"  Skipping {file_path}: {e}")
    
    if len(chains_by_model) < 2:
        print("\nError: Need at least 2 models for consensus")
        return
    
    # Create consensus detector
    print(f"\nUsing strategy: {args.strategy}")
    detector = ConsensusDetector(strategy=ConsensusStrategy(args.strategy))
    
    # Evaluate consensus
    print("\nComputing consensus...")
    consensus_results = []
    
    # Get minimum number of chains across models
    min_chains = min(len(chains) for chains in chains_by_model.values())
    
    for i in range(min_chains):
        # Get chain from each model (same query)
        model_predictions = {}
        chain_id = None
        
        for model_type, chains in chains_by_model.items():
            if i < len(chains):
                chain = chains[i]
                if chain_id is None:
                    chain_id = chain.query_id
                model_predictions[model_type] = extract_predictions(chain)
        
        if model_predictions:
            result = detector.detect_consensus(chain_id, model_predictions)
            consensus_results.append(result)
    
    print(f"Evaluated {len(consensus_results)} chains")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("CONSENSUS ANALYSIS")
    print("=" * 80)
    
    analyzer = CrossModelAnalyzer()
    agreement_analysis = analyzer.analyze_agreement_patterns(consensus_results)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Chains: {agreement_analysis['total_chains']}")
    print(f"  Chains with Consensus: {agreement_analysis['chains_with_consensus']}")
    print(f"  Consensus Rate: {agreement_analysis['consensus_rate']*100:.2f}%")
    print(f"  Avg Agreement Rate: {agreement_analysis['avg_agreement_rate']*100:.2f}%")
    print(f"  Disagreement Rate: {agreement_analysis['disagreement_rate']*100:.2f}%")
    
    print(f"\nAgreement Distribution:")
    dist = agreement_analysis['agreement_distribution']
    print(f"  Mean: {dist['mean']*100:.2f}%")
    print(f"  Std: {dist['std']*100:.2f}%")
    print(f"  Min: {dist['min']*100:.2f}%")
    print(f"  Max: {dist['max']*100:.2f}%")
    
    # Save results
    output_path = project_root / f"experiments/consensus_{args.domain}_{args.strategy}_results.json"
    results_data = {
        "strategy": args.strategy,
        "domain": args.domain,
        "models": [mt.value for mt in chains_by_model.keys()],
        "agreement_analysis": agreement_analysis,
        "consensus_results": [r.to_dict() for r in consensus_results[:10]],  # Save first 10
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

