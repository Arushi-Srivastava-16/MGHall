"""
Generate Reasoning Chains from Multiple Models.

Generates reasoning chains for the same queries using different LLM models
(GPT-4, Gemini, Llama, Mistral) and saves them for analysis.
"""

import sys
from pathlib import Path
import json
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.unified_schema import ReasoningChain, Domain
from src.multi_model.model_config import ModelType, get_default_models
from src.multi_model.llm_inference import MultiModelInference


def load_test_queries(
    test_path: Path,
    max_queries: int = 50,
    domain: Domain = Domain.MATH,
) -> list:
    """Load test queries from dataset."""
    queries = []
    
    with open(test_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_queries:
                break
            
            chain_dict = json.loads(line)
            chain = ReasoningChain.from_dict(chain_dict)
            
            if chain.domain == domain:
                queries.append({
                    "query_id": chain.query_id,
                    "query": chain.query,
                    "domain": chain.domain,
                    "ground_truth": chain.ground_truth,
                })
    
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate multi-model reasoning chains")
    parser.add_argument("--domain", type=str, default="math", choices=["math", "code", "medical"],
                       help="Domain to generate chains for")
    parser.add_argument("--max-queries", type=int, default=50,
                       help="Maximum number of queries to process")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to use (default: all available)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MULTI-MODEL CHAIN GENERATION")
    print("=" * 80)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    
    # Domain-specific paths
    domain_paths = {
        "math": project_root / "data/processed/splits/test.jsonl",
        "code": project_root / "data/processed/code_test_splits/test.jsonl",
        "medical": project_root / "data/processed/medical_test_splits/test.jsonl",
    }
    
    test_path = domain_paths.get(args.domain)
    if not test_path or not test_path.exists():
        print(f"Error: Test data not found for domain '{args.domain}' at {test_path}")
        return
    
    domain = Domain(args.domain)
    output_dir = project_root / "data/multi_model/generated_chains"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queries
    print(f"\nLoading test queries from {test_path}...")
    queries = load_test_queries(test_path, args.max_queries, domain)
    print(f"Loaded {len(queries)} queries for domain: {domain.value}")
    
    if not queries:
        print("No queries found. Exiting.")
        return
    
    # Initialize models
    print("\nInitializing models...")
    if args.models:
        model_types = [ModelType(m) for m in args.models]
    else:
        model_types = get_default_models()
    
    if not model_types:
        print("Error: No models available. Please set API keys or install local models.")
        return
    
    print(f"Models to use: {[mt.value for mt in model_types]}")
    
    multi_model = MultiModelInference(model_types)
    
    # Generate chains
    print(f"\nGenerating reasoning chains...")
    print(f"Temperature: {args.temperature}")
    print(f"Queries: {len(queries)}")
    print()
    
    chains_by_model = {model_type: [] for model_type in model_types}
    
    for i, query_data in enumerate(tqdm(queries, desc="Processing queries")):
        query_id = query_data["query_id"]
        query = query_data["query"]
        
        # Generate from all models
        results = multi_model.infer_all(query, domain, args.temperature)
        
        # Convert to reasoning chains
        for model_type, result in results.items():
            if result.error:
                print(f"\n  Warning: {model_type.value} failed for query {query_id}: {result.error}")
                continue
            
            chain = multi_model.convert_to_reasoning_chain(
                result,
                query_id,
                domain,
                query_data.get("ground_truth", "")
            )
            
            chains_by_model[model_type].append(chain)
    
    # Save chains
    print("\nSaving generated chains...")
    for model_type, chains in chains_by_model.items():
        if not chains:
            print(f"  {model_type.value}: No chains generated")
            continue
        
        output_file = output_dir / f"{model_type.value}_{domain.value}_chains.jsonl"
        
        with open(output_file, 'w') as f:
            for chain in chains:
                f.write(json.dumps(chain.to_dict()) + '\n')
        
        print(f"  {model_type.value}: {len(chains)} chains -> {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated chains:")
    for model_type in model_types:
        num_chains = len(chains_by_model[model_type])
        print(f"  {model_type.value}: {num_chains} chains")
    
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Run experiments/train_fingerprint_classifier.py")
    print("  2. Run experiments/consensus_evaluation.py")
    print("  3. Run experiments/run_phase5_pipeline.py")


if __name__ == "__main__":
    main()

