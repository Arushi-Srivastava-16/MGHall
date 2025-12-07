"""
Converter for PRM800K dataset to unified format.

This module converts PRM800K JSONL files into the unified reasoning chain format.
"""

import json
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .unified_schema import (
        ReasoningChain,
        ReasoningStep,
        DependencyGraph,
        Domain,
        ErrorType,
        validate_reasoning_chain,
    )
except ImportError:
    from unified_schema import (
        ReasoningChain,
        ReasoningStep,
        DependencyGraph,
        Domain,
        ErrorType,
        validate_reasoning_chain,
    )


def convert_prm800k_sample(sample: Dict[str, Any], query_id: Optional[str] = None) -> ReasoningChain:
    """
    Convert a single PRM800K sample to unified format.
    
    Args:
        sample: PRM800K sample dictionary
        query_id: Optional query ID (generated if not provided)
        
    Returns:
        ReasoningChain in unified format
    """
    if query_id is None:
        query_id = str(uuid.uuid4())
    
    # Extract question information
    question = sample.get("question", {})
    problem = question.get("problem", "")
    ground_truth = question.get("ground_truth_answer", "")
    pre_generated_steps = question.get("pre_generated_steps", [])
    
    # Extract label information
    label = sample.get("label", {})
    label_steps = label.get("steps", [])
    
    # Build reasoning steps
    reasoning_steps = []
    origin_step_id = None
    
    for i, (pre_step, label_step) in enumerate(zip(pre_generated_steps, label_steps)):
        completions = label_step.get("completions", [])
        if not completions:
            continue
        
        completion = completions[0]
        rating = completion.get("rating", 0)
        
        # Map ratings: +1→correct, -1→incorrect, 0→neutral
        is_correct = rating == 1
        is_origin = False
        
        # Identify first error (origin of hallucination)
        if rating == -1 and origin_step_id is None:
            is_origin = True
            origin_step_id = i
        
        # Determine error type (default to logical for math problems)
        error_type = None
        if not is_correct:
            error_type = ErrorType.LOGICAL  # Most math errors are logical
        
        # Build sequential dependencies (step i depends on step i-1)
        depends_on = [i - 1] if i > 0 else []
        
        step = ReasoningStep(
            step_id=i,
            text=pre_step,
            is_correct=is_correct,
            is_origin=is_origin,
            error_type=error_type,
            depends_on=depends_on,
        )
        reasoning_steps.append(step)
    
    # Build dependency graph (sequential: i → i+1)
    nodes = list(range(len(reasoning_steps)))
    edges = [[i, i + 1] for i in range(len(reasoning_steps) - 1)]
    
    dependency_graph = DependencyGraph(nodes=nodes, edges=edges)
    
    # Create reasoning chain
    chain = ReasoningChain(
        domain=Domain.MATH,
        query_id=query_id,
        query=problem,
        ground_truth=ground_truth,
        reasoning_steps=reasoning_steps,
        dependency_graph=dependency_graph,
    )
    
    return chain


def convert_prm800k_file(
    input_path: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Convert PRM800K JSONL file to unified format.
    
    Args:
        input_path: Path to input PRM800K JSONL file
        output_path: Path to output unified format JSONL file
        max_samples: Maximum number of samples to process (None for all)
        validate: Whether to validate each converted chain
        
    Returns:
        Dictionary with conversion statistics
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": 0,
        "converted": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
    }
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        # Process with progress bar
        lines = f_in.readlines()
        if max_samples:
            lines = lines[:max_samples]
        
        for line in tqdm(lines, desc=f"Converting {input_path.name}"):
            stats["total"] += 1
            
            try:
                sample = json.loads(line)
                chain = convert_prm800k_sample(sample)
                
                # Validate if requested
                if validate:
                    errors = validate_reasoning_chain(chain)
                    if errors:
                        stats["invalid"] += 1
                        stats["errors"].extend(errors)
                        continue
                    stats["valid"] += 1
                
                # Write to output (JSONL format - single line per object)
                f_out.write(chain.to_json(pretty=False) + "\n")
                stats["converted"] += 1
                
            except Exception as e:
                stats["errors"].append(f"Error processing sample {stats['total']}: {str(e)}")
                continue
    
    return stats


def convert_prm800k_dataset(
    input_dir: Path,
    output_dir: Path,
    splits: List[str] = ["phase2_train", "phase2_test"],
    max_samples_per_split: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Convert entire PRM800K dataset.
    
    Args:
        input_dir: Directory containing PRM800K data files
        output_dir: Output directory for converted files
        splits: List of splits to convert
        max_samples_per_split: Optional dict limiting samples per split
        
    Returns:
        Dictionary with overall statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = input_dir / "prm800k" / "data"
    if not data_dir.exists():
        data_dir = input_dir / "data"
    
    overall_stats = {}
    
    for split in splits:
        input_file = data_dir / f"{split}.jsonl"
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue
        
        output_file = output_dir / f"prm800k_{split}.jsonl"
        
        max_samples = None
        if max_samples_per_split and split in max_samples_per_split:
            max_samples = max_samples_per_split[split]
        
        print(f"\nConverting {split}...")
        stats = convert_prm800k_file(input_file, output_file, max_samples=max_samples)
        overall_stats[split] = stats
        
        print(f"  Converted: {stats['converted']}/{stats['total']}")
        print(f"  Valid: {stats['valid']}/{stats['converted']}")
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")
    
    return overall_stats


if __name__ == "__main__":
    # Example usage - use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "data/raw/prm800k"
    output_dir = project_root / "data/processed"
    
    # Convert full dataset (all 97K+ samples)
    stats = convert_prm800k_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        splits=["phase2_train", "phase2_test"],
        max_samples_per_split=None,  # No limit - convert everything!
    )
    
    print("\n=== Conversion Complete ===")
    for split, split_stats in stats.items():
        print(f"\n{split}:")
        print(f"  Total: {split_stats['total']}")
        print(f"  Converted: {split_stats['converted']}")
        print(f"  Valid: {split_stats['valid']}")

