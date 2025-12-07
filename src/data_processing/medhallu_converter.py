"""
Converter for MedHallu dataset to unified format.

This module converts MedHallu medical Q&A data into reasoning chains using
LLM-based step decomposition.
"""

import json
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

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


def decompose_answer_with_llm(
    question: str,
    answer: str,
    ground_truth: str,
    hallucination_category: str,
    use_llm: bool = False,
) -> List[Dict[str, Any]]:
    """
    Decompose an answer into reasoning steps.
    
    For now, uses simple heuristics. Can be enhanced with LLM calls.
    
    Args:
        question: Medical question
        answer: Answer to decompose (may be hallucinated)
        ground_truth: Correct answer
        hallucination_category: Category of hallucination
        use_llm: Whether to use LLM for decomposition (requires API key)
        
    Returns:
        List of step dictionaries
    """
    steps = []
    
    if use_llm:
        # TODO: Implement LLM-based decomposition
        # This would call GPT-4 or similar to break down the answer
        # For now, fall through to heuristic method
        pass
    
    # Heuristic method: split by sentences and mark based on comparison
    answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
    ground_truth_sentences = [s.strip() for s in ground_truth.split('.') if s.strip()]
    
    # Simple matching: if sentence appears in ground truth, it's correct
    # Otherwise, it's likely a hallucination
    for i, sentence in enumerate(answer_sentences):
        is_correct = any(sentence.lower() in gt.lower() or gt.lower() in sentence.lower() 
                        for gt in ground_truth_sentences)
        
        # First incorrect sentence is the origin
        is_origin = not is_correct and all(
            any(prev_sent.lower() in gt.lower() or gt.lower() in prev_sent.lower() 
                for gt in ground_truth_sentences)
            for prev_sent in answer_sentences[:i]
        )
        
        # Map hallucination category to error type
        error_type = None
        if not is_correct:
            if "factual" in hallucination_category.lower():
                error_type = ErrorType.FACTUAL
            elif "logical" in hallucination_category.lower():
                error_type = ErrorType.LOGICAL
            else:
                error_type = ErrorType.FACTUAL  # Default for medical
        
        steps.append({
            "step_id": i,
            "text": sentence,
            "is_correct": is_correct,
            "is_origin": is_origin,
            "error_type": error_type,
            "depends_on": [i - 1] if i > 0 else [],
        })
    
    # If no steps extracted, create a single step
    if not steps:
        is_correct = answer.lower() == ground_truth.lower()
        steps.append({
            "step_id": 0,
            "text": answer,
            "is_correct": is_correct,
            "is_origin": not is_correct,
            "error_type": ErrorType.FACTUAL if not is_correct else None,
            "depends_on": [],
        })
    
    return steps


def convert_medhallu_sample(
    sample: Dict[str, Any],
    query_id: Optional[str] = None,
    use_llm: bool = False,
) -> ReasoningChain:
    """
    Convert a single MedHallu sample to unified format.
    
    Args:
        sample: MedHallu sample dictionary
        query_id: Optional query ID (generated if not provided)
        use_llm: Whether to use LLM for step decomposition
        
    Returns:
        ReasoningChain in unified format
    """
    if query_id is None:
        query_id = str(uuid.uuid4())
    
    question = sample.get("Question", "")
    hallucinated_answer = sample.get("Hallucinated Answer", "")
    ground_truth = sample.get("Ground Truth", "")
    hallucination_category = sample.get("Category of Hallucination", "Unknown")
    
    # Decompose answer into steps
    step_dicts = decompose_answer_with_llm(
        question=question,
        answer=hallucinated_answer,
        ground_truth=ground_truth,
        hallucination_category=hallucination_category,
        use_llm=use_llm,
    )
    
    # Build reasoning steps
    reasoning_steps = [
        ReasoningStep(
            step_id=step["step_id"],
            text=step["text"],
            is_correct=step["is_correct"],
            is_origin=step["is_origin"],
            error_type=step["error_type"],
            depends_on=step["depends_on"],
        )
        for step in step_dicts
    ]
    
    # Build dependency graph (sequential)
    nodes = [step.step_id for step in reasoning_steps]
    edges = [[i, i + 1] for i in range(len(reasoning_steps) - 1)]
    
    dependency_graph = DependencyGraph(nodes=nodes, edges=edges)
    
    # Create reasoning chain
    chain = ReasoningChain(
        domain=Domain.MEDICAL,
        query_id=query_id,
        query=question,
        ground_truth=ground_truth,
        reasoning_steps=reasoning_steps,
        dependency_graph=dependency_graph,
    )
    
    return chain


def convert_medhallu_dataset(
    input_dir: Path,
    output_path: Path,
    split: str = "pqa_labeled",
    max_samples: Optional[int] = None,
    use_llm: bool = False,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Convert MedHallu dataset to unified format.
    
    Args:
        input_dir: Directory containing MedHallu dataset
        output_path: Path to output unified format JSONL file
        split: Dataset split to convert
        max_samples: Maximum number of samples to process (None for all)
        use_llm: Whether to use LLM for step decomposition
        validate: Whether to validate each converted chain
        
    Returns:
        Dictionary with conversion statistics
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_path = input_dir / split
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {dataset_path}")
    
    dataset_dict = load_from_disk(str(dataset_path))
    
    # Handle DatasetDict (has 'train' split) vs Dataset
    if isinstance(dataset_dict, dict) or hasattr(dataset_dict, 'keys'):
        if 'train' in dataset_dict:
            dataset = dataset_dict['train']
        else:
            raise ValueError(f"Expected 'train' split in dataset, got: {list(dataset_dict.keys())}")
    else:
        dataset = dataset_dict
    
    stats = {
        "total": 0,
        "converted": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
    }
    
    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for sample in tqdm(samples, desc=f"Converting MedHallu {split}"):
            stats["total"] += 1
            
            try:
                # Convert sample to dict if needed
                if hasattr(sample, 'keys'):
                    sample_dict = dict(sample)
                else:
                    sample_dict = sample
                
                chain = convert_medhallu_sample(sample_dict, use_llm=use_llm)
                
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


if __name__ == "__main__":
    # Example usage - use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "data/raw/medhallu"
    output_path = project_root / "data/processed/medhallu.jsonl"
    
    stats = convert_medhallu_dataset(
        input_dir=input_dir,
        output_path=output_path,
        split="pqa_labeled",
        max_samples=None,  # Convert all samples
        use_llm=False,  # Set to True if you have OpenAI API key
    )
    
    print("\n=== Conversion Complete ===")
    print(f"Total samples: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Valid: {stats['valid']}")
    if stats['errors']:
        print(f"Errors: {len(stats['errors'])}")

