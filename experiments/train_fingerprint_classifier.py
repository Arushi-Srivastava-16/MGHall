"""
Train Fingerprint Classifier.

Trains a classifier to identify which LLM generated a reasoning chain
based on extracted fingerprint features.
"""

import sys
from pathlib import Path
import json
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.unified_schema import ReasoningChain
from src.multi_model.model_config import ModelType
from src.multi_model.fingerprint_classifier import FingerprintClassifier


def load_chains_from_file(file_path: Path) -> list:
    """Load chains from JSONL file."""
    chains = []
    with open(file_path, 'r') as f:
        for line in f:
            chain_dict = json.loads(line)
            chain = ReasoningChain.from_dict(chain_dict)
            chains.append(chain)
    return chains


def main():
    parser = argparse.ArgumentParser(description="Train fingerprint classifier")
    parser.add_argument("--domain", type=str, default="math", choices=["math", "code", "medical"])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FINGERPRINT CLASSIFIER TRAINING")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    chains_dir = project_root / "data/multi_model/generated_chains"
    
    # Find all chain files for the domain
    print(f"\nLoading chains for domain: {args.domain}")
    chains_by_model = {}
    
    for file_path in chains_dir.glob(f"*_{args.domain}_chains.jsonl"):
        model_name = file_path.stem.split('_')[0]
        try:
            model_type = ModelType(model_name)
            chains = load_chains_from_file(file_path)
            chains_by_model[model_type] = chains
            print(f"  {model_name}: {len(chains)} chains")
        except Exception as e:
            print(f"  Skipping {file_path}: {e}")
    
    if len(chains_by_model) < 2:
        print("\nError: Need at least 2 models to train classifier")
        print("Please run generate_multi_model_chains.py first")
        return
    
    # Train classifier
    print(f"\nTraining classifier...")
    classifier = FingerprintClassifier(
        classifier_type="random_forest",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    
    metrics = classifier.train(chains_by_model, test_size=0.2)
    
    # Print results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"\nOverall Accuracy: {metrics.accuracy*100:.2f}%")
    
    print("\nPer-Model Metrics:")
    for model_name in sorted(metrics.precision.keys()):
        print(f"\n{model_name}:")
        print(f"  Precision: {metrics.precision[model_name]*100:.2f}%")
        print(f"  Recall: {metrics.recall[model_name]*100:.2f}%")
        print(f"  F1-Score: {metrics.f1_score[model_name]*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix)
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance = classifier.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in sorted_features:
        print(f"  {feat}: {imp:.4f}")
    
    # Save classifier
    output_path = project_root / "models/fingerprint_classifier" / f"{args.domain}_classifier.pkl"
    classifier.save(output_path)
    print(f"\nSaved classifier to: {output_path}")
    
    # Save metrics
    metrics_path = output_path.parent / f"{args.domain}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

