"""
Complete Phase 5 Pipeline.

Runs the complete Phase 5 multi-model fingerprinting pipeline:
1. Load generated chains
2. Extract fingerprints
3. Train classifier
4. Evaluate consensus
5. Generate cross-model analysis
"""

import sys
from pathlib import Path
import json
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.unified_schema import ReasoningChain
from src.multi_model.model_config import ModelType
from src.multi_model.fingerprint_extractor import FingerprintExtractor
from src.multi_model.fingerprint_classifier import FingerprintClassifier
from src.multi_model.consensus_detector import ConsensusDetector, ConsensusStrategy
from src.multi_model.cross_model_analyzer import CrossModelAnalyzer
from src.multi_model.pattern_database import PatternDatabase


def load_chains(file_path: Path):
    """Load chains from JSONL file."""
    chains = []
    with open(file_path, 'r') as f:
        for line in f:
            chain_dict = json.loads(line)
            chain = ReasoningChain.from_dict(chain_dict)
            chains.append(chain)
    return chains


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 complete pipeline")
    parser.add_argument("--domain", type=str, default="math")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip classifier training (use existing model)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 5: MULTI-MODEL FINGERPRINTING - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    project_root = Path(__file__).parent.parent
    chains_dir = project_root / "data/multi_model/generated_chains"
    output_dir = project_root / "experiments"
    
    # Step 1: Load chains
    print("\n" + "=" * 80)
    print("STEP 1: Loading Generated Chains")
    print("=" * 80)
    
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
    
    if not chains_by_model:
        print("\nError: No chains found. Please run generate_multi_model_chains.py first")
        return
    
    # Step 2: Extract fingerprints
    print("\n" + "=" * 80)
    print("STEP 2: Extracting Fingerprints")
    print("=" * 80)
    
    extractor = FingerprintExtractor()
    fingerprints_by_model = {}
    
    for model_type, chains in chains_by_model.items():
        fingerprints = extractor.extract_batch(chains)
        fingerprints_by_model[model_type] = fingerprints
        print(f"  {model_type.value}: {len(fingerprints)} fingerprints extracted")
    
    # Step 3: Train/Load Fingerprint Classifier
    print("\n" + "=" * 80)
    print("STEP 3: Fingerprint Classifier")
    print("=" * 80)
    
    classifier_path = project_root / f"models/fingerprint_classifier/{args.domain}_classifier.pkl"
    
    if args.skip_training and classifier_path.exists():
        print(f"Loading existing classifier from {classifier_path}")
        classifier = FingerprintClassifier.load(classifier_path)
    else:
        print("Training new classifier...")
        classifier = FingerprintClassifier(classifier_type="random_forest", n_estimators=100)
        metrics = classifier.train(chains_by_model, test_size=0.2)
        
        print(f"\nClassifier Accuracy: {metrics.accuracy*100:.2f}%")
        print("Per-Model F1 Scores:")
        for model, f1 in metrics.f1_score.items():
            print(f"  {model}: {f1*100:.2f}%")
        
        classifier.save(classifier_path)
    
    # Step 4: Consensus Detection
    print("\n" + "=" * 80)
    print("STEP 4: Consensus Detection")
    print("=" * 80)
    
    detector = ConsensusDetector(strategy=ConsensusStrategy.MAJORITY_VOTE)
    consensus_results = []
    
    min_chains = min(len(chains) for chains in chains_by_model.values())
    
    for i in range(min_chains):
        model_predictions = {}
        chain_id = None
        
        for model_type, chains in chains_by_model.items():
            if i < len(chains):
                chain = chains[i]
                if chain_id is None:
                    chain_id = chain.query_id
                predictions = [step.is_correct for step in chain.reasoning_steps]
                model_predictions[model_type] = predictions
        
        if model_predictions:
            result = detector.detect_consensus(chain_id, model_predictions)
            consensus_results.append(result)
    
    print(f"  Evaluated {len(consensus_results)} chains")
    
    # Step 5: Pattern Database
    print("\n" + "=" * 80)
    print("STEP 5: Building Pattern Database")
    print("=" * 80)
    
    pattern_db = PatternDatabase()
    
    for model_type, chains in chains_by_model.items():
        for chain in chains:
            patterns = pattern_db.extract_patterns_from_chain(chain, model_type)
            for pattern in patterns:
                pattern_db.add_pattern(pattern)
    
    print(f"  Total patterns: {len(pattern_db.patterns)}")
    for model_type in chains_by_model.keys():
        profile = pattern_db.get_model_vulnerability_profile(model_type)
        print(f"  {model_type.value}: {profile['total_patterns']} patterns, " +
              f"vulnerability score: {profile['vulnerability_score']:.3f}")
    
    # Save pattern database
    pattern_db_path = project_root / f"data/multi_model/patterns/{args.domain}_patterns.json"
    pattern_db.save(pattern_db_path)
    
    # Step 6: Cross-Model Analysis
    print("\n" + "=" * 80)
    print("STEP 6: Cross-Model Analysis")
    print("=" * 80)
    
    analyzer = CrossModelAnalyzer(pattern_db)
    report = analyzer.generate_comparative_report(chains_by_model, consensus_results)
    
    analyzer.print_summary(report)
    
    # Save comprehensive results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "domain": args.domain,
        "models": [mt.value for mt in chains_by_model.keys()],
        "chains_per_model": {mt.value: len(chains) for mt, chains in chains_by_model.items()},
        "classifier_metrics": {
            "path": str(classifier_path),
            "accuracy": classifier.evaluate(
                classifier.scaler.transform([extractor.get_feature_vector(fp) for fp in fingerprints_by_model[next(iter(chains_by_model.keys()))]]),
                [0] * len(fingerprints_by_model[next(iter(chains_by_model.keys()))])
            ).accuracy if not args.skip_training else "loaded_from_disk"
        },
        "consensus_analysis": analyzer.analyze_agreement_patterns(consensus_results),
        "cross_model_report": report,
    }
    
    output_path = output_dir / f"phase5_{args.domain}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComprehensive results saved to: {output_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 5 PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nKey Outputs:")
    print(f"  - Fingerprint Classifier: {classifier_path}")
    print(f"  - Pattern Database: {pattern_db_path}")
    print(f"  - Analysis Report: {output_path}")
    
    print(f"\nSuccess Criteria:")
    if len(chains_by_model) >= 4:
        print(f"  ✓ Multi-model inference: {len(chains_by_model)} models")
    else:
        print(f"  ⚠ Multi-model inference: {len(chains_by_model)} models (target: 4-5)")
    
    # Check classifier accuracy if available
    print(f"  ✓ Fingerprint classifier trained/loaded")
    
    if report.get("agreement_analysis"):
        consensus_rate = report["agreement_analysis"]["consensus_rate"]
        if consensus_rate > 0.5:
            print(f"  ✓ Consensus detection: {consensus_rate*100:.1f}% consensus rate")
        else:
            print(f"  ⚠ Consensus detection: {consensus_rate*100:.1f}% consensus rate")
    
    print(f"  ✓ Cross-model analysis complete")
    print(f"  ✓ Pattern database: {len(pattern_db.patterns)} patterns")
    
    print("\nPhase 5 implementation complete!")


if __name__ == "__main__":
    main()

