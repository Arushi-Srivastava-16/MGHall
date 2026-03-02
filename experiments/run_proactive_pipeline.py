"""
End-to-End Proactive Pipeline.

Complete pipeline for proactive hallucination detection:
1. Load trained models
2. Process reasoning chains in streaming mode
3. Detect vulnerability in real-time
4. Trigger interventions
5. Generate comprehensive reports
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from tqdm import tqdm
from models.gnn_architectures.gat_model import ConfidenceGatedGAT
from src.proactive.vulnerability_predictor import load_vulnerability_predictor
from src.proactive.streaming_inference import StreamingAnalyzer
from src.proactive.interventional_controller import InterventionalController, CorrectionGenerator
from src.proactive.proactive_evaluator import ProactiveEvaluator
from src.data_processing.unified_schema import ReasoningChain


def main():
    print("="*80)
    print("PROACTIVE HALLUCINATION DETECTION PIPELINE")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Configuration
    config = {
        "domain": "math",
        "checkpoint": project_root / "models/checkpoints/test_run/best_model.pth",
        "test_data": project_root / "data/processed/splits/test.jsonl",
        "max_test_chains": 50,
        "warn_threshold": 0.3,
        "correct_threshold": 0.7,
    }
    
    # Load model
    print(f"\nLoading {config['domain']} model...")
    if not config["checkpoint"].exists():
        print(f"Error: Checkpoint not found at {config['checkpoint']}")
        return
    
    predictor = load_vulnerability_predictor(
        checkpoint_path=config["checkpoint"],
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
    print("Model loaded successfully!")
    
    # Create pipeline components
    print("\nInitializing pipeline components...")
    analyzer = StreamingAnalyzer(
        vulnerability_predictor=predictor,
        max_concurrent_streams=100,
    )
    
    controller = InterventionalController(
        warn_threshold=config["warn_threshold"],
        correct_threshold=config["correct_threshold"],
        enable_warnings=True,
        enable_corrections=True,
        correction_generator=CorrectionGenerator(),
    )
    
    evaluator = ProactiveEvaluator(
        streaming_analyzer=analyzer,
        interventional_controller=controller,
    )
    
    # Load test data
    print(f"\nLoading test data from {config['test_data']}...")
    test_chains = []
    with open(config["test_data"], "r") as f:
        for i, line in enumerate(f):
            if i >= config["max_test_chains"]:
                break
            chain_dict = json.loads(line)
            chain = ReasoningChain.from_dict(chain_dict)
            test_chains.append(chain)
    
    print(f"Loaded {len(test_chains)} test chains")
    
    # Run evaluation
    print("\n" + "="*80)
    print("RUNNING PROACTIVE EVALUATION")
    print("="*80)
    
    metrics = evaluator.evaluate_on_chains(test_chains)
    
    # Print results
    print("\n" + "="*80)
    print("PROACTIVE SYSTEM EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Chains: {metrics['total_chains']}")
    print(f"  Chains with Errors: {metrics['chains_with_errors']}")
    print(f"  Correct Chains: {metrics['total_chains'] - metrics['chains_with_errors']}")
    
    print(f"\nVulnerability Detection:")
    print(f"  Detection Rate: {metrics['detection_rate']*100:.2f}%")
    print(f"  Early Detection Rate: {metrics['early_detection_rate']*100:.2f}%")
    print(f"  Avg Time-to-Detection: {metrics['avg_time_to_detection']:.2f} steps before error")
    
    print(f"\nIntervention Performance:")
    print(f"  Total Interventions: {metrics['interventions_triggered']}")
    print(f"  False Positive Rate: {metrics['false_positive_rate']*100:.2f}%")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # Controller statistics
    ctrl_stats = controller.get_statistics()
    print(f"\nIntervention Breakdown:")
    print(f"  Warnings: {ctrl_stats['warnings']}")
    print(f"  Corrections: {ctrl_stats['corrections']}")
    print(f"  Risk Levels:")
    print(f"    - HIGH: {ctrl_stats['risk_level_counts']['HIGH']}")
    print(f"    - MEDIUM: {ctrl_stats['risk_level_counts']['MEDIUM']}")
    
    # Success assessment
    print("\n" + "="*80)
    print("SUCCESS ASSESSMENT")
    print("="*80)
    
    targets = {
        "detection_rate": (0.8, metrics['detection_rate']),
        "false_positive_rate": (0.2, metrics['false_positive_rate']),
        "early_detection_rate": (0.7, metrics['early_detection_rate']),
    }
    
    print()
    for metric_name, (target, actual) in targets.items():
        if metric_name == "false_positive_rate":
            passed = actual <= target
            print(f"  {metric_name}: {actual*100:.2f}% {'<=' if passed else '>'} {target*100:.0f}% target - {'PASS' if passed else 'FAIL'}")
        else:
            passed = actual >= target
            print(f"  {metric_name}: {actual*100:.2f}% {'>=' if passed else '<'} {target*100:.0f}% target - {'PASS' if passed else 'FAIL'}")
    
    # Save results
    output_path = project_root / "experiments" / "proactive_pipeline_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
            "metrics": metrics,
            "controller_stats": ctrl_stats,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "="*80)
    print("PROACTIVE PIPELINE COMPLETE")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("  - Real-time vulnerability detection")
    print("  - Early warning system (before errors occur)")
    print("  - Risk-based interventions (warnings + corrections)")
    print("  - Comprehensive evaluation metrics")


if __name__ == "__main__":
    main()

