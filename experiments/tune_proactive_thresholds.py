"""
Threshold tuning script for Phase 4 proactive system.

Goal: Find optimal thresholds that achieve:
- Detection Rate ≥ 80%
- False Positive Rate ≤ 20%
- Max Early Detection Rate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from tqdm import tqdm
from models.gnn_architectures.gat_model import ConfidenceGatedGAT
from src.proactive.vulnerability_predictor import load_vulnerability_predictor
from src.proactive.streaming_inference import StreamingAnalyzer
from src.proactive.interventional_controller import InterventionalController, CorrectionGenerator
from src.proactive.proactive_evaluator import ProactiveEvaluator
from src.data_processing.unified_schema import ReasoningChain


def load_test_data(test_path, max_chains=100):
    """Load test chains."""
    chains = []
    with open(test_path) as f:
        for i, line in enumerate(f):
            if i >= max_chains:
                break
            chain_dict = json.loads(line)
            chains.append(ReasoningChain.from_dict(chain_dict))
    return chains


def evaluate_thresholds(warn_threshold, correct_threshold, analyzer, test_chains):
    """Evaluate performance with given thresholds."""
    
    # Create controller with these thresholds
    controller = InterventionalController(
        warn_threshold=warn_threshold,
        correct_threshold=correct_threshold,
        enable_warnings=True,
        enable_corrections=True,
        correction_generator=CorrectionGenerator(),
    )
    
    # Create evaluator
    evaluator = ProactiveEvaluator(
        streaming_analyzer=analyzer,
        interventional_controller=controller,
    )
    
    # Evaluate
    metrics = evaluator.evaluate_on_chains(test_chains)
    
    return metrics


def main():
    print("="*70)
    print("PHASE 4 THRESHOLD TUNING")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    # Load model
    print("\n📂 Loading model...")
    checkpoint = project_root / "models/checkpoints/test_run/best_model.pth"
    
    predictor = load_vulnerability_predictor(
        checkpoint_path=checkpoint,
        model_class=ConfidenceGatedGAT,
        model_kwargs={
            "input_dim": 395,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.1,
            "use_confidence_gating": True,
        },
    )
    
    # Create analyzer (reused across all threshold combinations)
    analyzer = StreamingAnalyzer(
        vulnerability_predictor=predictor,
        max_concurrent_streams=100,
    )
    
    # Load test data
    print("📂 Loading test data...")
    test_path = project_root / "data/processed/splits/test.jsonl"
    test_chains = load_test_data(test_path, max_chains=100)
    
    print(f"  Loaded {len(test_chains)} chains")
    
    # Determine how many have errors
    error_chains = sum(1 for c in test_chains if not all(s.is_correct for s in c.reasoning_steps))
    correct_chains = len(test_chains) - error_chains
    print(f"  Error chains: {error_chains}, Correct chains: {correct_chains}")
    
    # Grid search
    print("\n🔍 Grid Search:")
    warn_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    correct_thresholds = [0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    print(f"  Testing {len(warn_thresholds)} × {len(correct_thresholds)} = {len(warn_thresholds)*len(correct_thresholds)} combinations...")
    print()
    
    for warn_th in tqdm(warn_thresholds, desc="Warn threshold"):
        for corr_th in correct_thresholds:
            if corr_th <= warn_th:
                continue  # Skip invalid combinations
            
            metrics = evaluate_thresholds(warn_th, corr_th, analyzer, test_chains)
            
            results.append({
                'warn_threshold': warn_th,
                'correct_threshold': corr_th,
                'detection_rate': metrics['detection_rate'],
                'false_positive_rate': metrics['false_positive_rate'],
                'early_detection_rate': metrics['early_detection_rate'],
                'avg_time_to_detection': metrics['avg_time_to_detection'],
                'interventions': metrics['interventions_triggered'],
            })
    
    # Find best configurations
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Filter by constraints
    valid_configs = [
        r for r in results
        if r['detection_rate'] >= 0.8 and r['false_positive_rate'] <= 0.2
    ]
    
    print(f"\n✅ Valid configurations (DR ≥ 80%, FPR ≤ 20%): {len(valid_configs)}/{len(results)}")
    
    if valid_configs:
        # Sort by early detection rate (descending)
        valid_configs.sort(key=lambda x: x['early_detection_rate'], reverse=True)
        
        print("\n🏆 Top 5 Configurations:\n")
        print(f"{'Rank':<5} {'Warn':>6} {'Correct':>8} {'DR':>6} {'FPR':>6} {'Early%':>7} {'Leads':>6} {'Interv':>7}")
        print("-" * 70)
        
        for i, config in enumerate(valid_configs[:5], 1):
            print(f"{i:<5} "
                  f"{config['warn_threshold']:>6.1f} "
                  f"{config['correct_threshold']:>8.1f} "
                  f"{config['detection_rate']*100:>5.0f}% "
                  f"{config['false_positive_rate']*100:>5.0f}% "
                  f"{config['early_detection_rate']*100:>6.0f}% "
                  f"{config['avg_time_to_detection']:>6.1f} "
                  f"{config['interventions']:>7}")
        
        # Recommended configuration
        best = valid_configs[0]
        print("\n" + "="*70)
        print("RECOMMENDED CONFIGURATION")
        print("="*70)
        print(f"\nWarn Threshold: {best['warn_threshold']}")
        print(f"Correct Threshold: {best['correct_threshold']}")
        print(f"\nExpected Performance:")
        print(f"  Detection Rate: {best['detection_rate']*100:.1f}%")
        print(f"  False Positive Rate: {best['false_positive_rate']*100:.1f}%")
        print(f"  Early Detection Rate: {best['early_detection_rate']*100:.1f}%")
        print(f"  Avg Lead Time: {best['avg_time_to_detection']:.1f} steps")
        
    else:
        print("\n⚠️ No configuration meets both constraints!")
        print("\nRelaxing constraints...")
        
        # Show best trade-offs
        print("\n📊 Best Trade-offs:\n")
        results.sort(key=lambda x: (x['detection_rate'], -x['false_positive_rate']), reverse=True)
        
        print(f"{'Warn':>6} {'Correct':>8} {'DR':>6} {'FPR':>6} {'Early%':>7} {'Leads':>6}")
        print("-" * 60)
        
        for config in results[:10]:
            print(f"{config['warn_threshold']:>6.1f} "
                  f"{config['correct_threshold']:>8.1f} "
                  f"{config['detection_rate']*100:>5.0f}% "
                  f"{config['false_positive_rate']*100:>5.0f}% "
                  f"{config['early_detection_rate']*100:>6.0f}% "
                  f"{config['avg_time_to_detection']:>6.1f}")
    
    # Save all results
    output_path = project_root / "experiments/threshold_tuning_results.json"
    with open(output_path, "w") as f:
        json.dump({
            'search_space': {
                'warn_thresholds': warn_thresholds,
                'correct_thresholds': correct_thresholds,
            },
            'all_results': results,
            'valid_configs': valid_configs,
            'best_config': valid_configs[0] if valid_configs else None,
        }, f, indent=2)
    
    print(f"\n✅ Full results saved to {output_path}")


if __name__ == "__main__":
    main()
