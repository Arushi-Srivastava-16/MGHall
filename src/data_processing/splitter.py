"""
Stratified train/val/test split creation.

This module creates stratified splits preserving domain and error distributions.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .unified_schema import ReasoningChain, Domain
except ImportError:
    from unified_schema import ReasoningChain, Domain


def load_chains(input_path: Path) -> List[ReasoningChain]:
    """Load all reasoning chains from a JSONL file."""
    chains = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            chain_dict = json.loads(line)
            chain = ReasoningChain.from_dict(chain_dict)
            chains.append(chain)
    return chains


def get_stratification_key(chain: ReasoningChain) -> tuple:
    """
    Get stratification key for a chain.
    
    Returns tuple of (domain, error_category) where error_category is:
    - "no_error" if all steps are correct
    - "has_error" if any step is incorrect
    """
    domain = chain.domain.value if isinstance(chain.domain, Domain) else chain.domain
    has_error = any(not step.is_correct for step in chain.reasoning_steps)
    error_category = "has_error" if has_error else "no_error"
    return (domain, error_category)


def create_stratified_splits(
    chains: List[ReasoningChain],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[ReasoningChain]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        chains: List of reasoning chains
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    random.shuffle(chains)
    
    # Group by stratification key
    stratified_groups = defaultdict(list)
    for chain in chains:
        key = get_stratification_key(chain)
        stratified_groups[key].append(chain)
    
    # Split each group
    splits = {"train": [], "val": [], "test": []}
    
    for key, group_chains in stratified_groups.items():
        n = len(group_chains)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits["train"].extend(group_chains[:n_train])
        splits["val"].extend(group_chains[n_train:n_train + n_val])
        splits["test"].extend(group_chains[n_train + n_val:])
    
    # Shuffle each split
    for split_name in splits:
        random.shuffle(splits[split_name])
    
    return splits


def save_splits(
    splits: Dict[str, List[ReasoningChain]],
    output_dir: Path,
) -> None:
    """
    Save splits to JSONL files.
    
    Args:
        splits: Dictionary with 'train', 'val', 'test' keys
        output_dir: Directory to save split files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, chains in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for chain in tqdm(chains, desc=f"Saving {split_name}"):
                f.write(chain.to_json(pretty=False) + "\n")
        
        print(f"Saved {split_name}: {len(chains)} chains")


def create_splits_from_files(
    input_files: List[Path],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create splits from multiple input files.
    
    Args:
        input_files: List of JSONL files to combine
        output_dir: Directory to save split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed
        
    Returns:
        Dictionary with split statistics
    """
    # Load all chains
    all_chains = []
    for input_file in input_files:
        print(f"Loading {input_file.name}...")
        chains = load_chains(input_file)
        all_chains.extend(chains)
        print(f"  Loaded {len(chains)} chains")
    
    print(f"\nTotal chains: {len(all_chains)}")
    
    # Create splits
    splits = create_stratified_splits(
        all_chains,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    
    # Save splits
    save_splits(splits, output_dir)
    
    # Calculate statistics
    stats = {}
    for split_name, chains in splits.items():
        domain_counts = defaultdict(int)
        error_counts = defaultdict(int)
        
        for chain in chains:
            domain = chain.domain.value if isinstance(chain.domain, Domain) else chain.domain
            domain_counts[domain] += 1
            
            has_error = any(not step.is_correct for step in chain.reasoning_steps)
            error_counts["has_error" if has_error else "no_error"] += 1
        
        stats[split_name] = {
            "total": len(chains),
            "domains": dict(domain_counts),
            "errors": dict(error_counts),
        }
    
    # Save statistics
    stats_path = output_dir / "split_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


if __name__ == "__main__":
    # Example usage - use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data/processed"
    output_dir = project_root / "data/processed/splits"
    
    input_files = [
        data_dir / "prm800k_phase2_train.jsonl",
        data_dir / "humaneval.jsonl",
        data_dir / "medhallu.jsonl",
    ]
    
    # Filter to existing files
    input_files = [f for f in input_files if f.exists()]
    
    if input_files:
        stats = create_splits_from_files(input_files, output_dir)
        
        print("\n=== Split Statistics ===")
        for split_name, split_stats in stats.items():
            print(f"\n{split_name}:")
            print(f"  Total: {split_stats['total']}")
            print(f"  Domains: {split_stats['domains']}")
            print(f"  Errors: {split_stats['errors']}")
    else:
        print("No input files found. Run converters first.")

