"""
Converter for HumanEval dataset to unified format.

This module converts HumanEval code problems into reasoning chains with
step-level decomposition using AST parsing.
"""

import json
import gzip
import ast
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
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


def extract_code_steps(code: str) -> List[Dict[str, Any]]:
    """
    Extract logical steps from Python code using AST parsing.
    
    Args:
        code: Python code string
        
    Returns:
        List of step dictionaries with text and dependencies
    """
    steps = []
    
    try:
        tree = ast.parse(code)
        step_id = 0
        
        def visit_node(node, parent_id: Optional[int] = None):
            nonlocal step_id
            current_id = step_id
            
            # Extract step text based on node type
            step_text = ""
            if isinstance(node, ast.FunctionDef):
                step_text = f"Define function: {node.name}"
            elif isinstance(node, ast.If):
                step_text = "Conditional check"
            elif isinstance(node, ast.For):
                step_text = f"For loop: {ast.unparse(node.target) if hasattr(ast, 'unparse') else 'loop'}"
            elif isinstance(node, ast.While):
                step_text = "While loop"
            elif isinstance(node, ast.Return):
                step_text = f"Return: {ast.unparse(node.value) if hasattr(ast, 'unparse') and node.value else 'value'}"
            elif isinstance(node, ast.Assign):
                step_text = f"Assign: {ast.unparse(node.targets[0]) if hasattr(ast, 'unparse') else 'variable'}"
            elif isinstance(node, ast.Expr):
                step_text = "Expression evaluation"
            
            if step_text:
                depends_on = [parent_id] if parent_id is not None else []
                steps.append({
                    "step_id": current_id,
                    "text": step_text,
                    "depends_on": depends_on,
                })
                step_id += 1
                parent_id = current_id
            
            # Recursively visit child nodes
            for child in ast.iter_child_nodes(node):
                visit_node(child, parent_id)
        
        # Visit all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While, ast.Return, ast.Assign, ast.Expr)):
                visit_node(node)
        
        # If no steps extracted, create a single step
        if not steps:
            steps.append({
                "step_id": 0,
                "text": code[:100] + "..." if len(code) > 100 else code,
                "depends_on": [],
            })
        
    except SyntaxError:
        # Fallback: split by lines
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        for i, line in enumerate(lines):
            steps.append({
                "step_id": i,
                "text": line[:100],
                "depends_on": [i - 1] if i > 0 else [],
            })
    
    return steps


def execute_code_with_tests(code: str, tests: str) -> Dict[str, Any]:
    """
    Execute code with tests to determine correctness.
    
    Args:
        code: Python code to execute
        tests: Test code string
        
    Returns:
        Dictionary with execution results
    """
    # This is a placeholder - actual execution should be done carefully
    # For now, we assume code is correct if it can be parsed
    try:
        ast.parse(code)
        ast.parse(tests)
        return {"can_parse": True, "is_correct": True}  # Simplified
    except SyntaxError:
        return {"can_parse": False, "is_correct": False}


def convert_humaneval_sample(
    problem: Dict[str, Any],
    query_id: Optional[str] = None,
    generate_incorrect_variants: bool = False,
) -> List[ReasoningChain]:
    """
    Convert a single HumanEval problem to unified format.
    
    Args:
        problem: HumanEval problem dictionary
        query_id: Optional query ID (generated if not provided)
        generate_incorrect_variants: Whether to generate incorrect variants
        
    Returns:
        List of ReasoningChain objects (one correct + variants if requested)
    """
    if query_id is None:
        query_id = problem.get("task_id", str(uuid.uuid4()))
    
    prompt = problem.get("prompt", "")
    solution = problem.get("canonical_solution", "")
    test = problem.get("test", "")
    
    # Extract code steps from solution
    code_steps = extract_code_steps(solution)
    
    # Build correct reasoning chain
    reasoning_steps = []
    for step_info in code_steps:
        step = ReasoningStep(
            step_id=step_info["step_id"],
            text=step_info["text"],
            is_correct=True,
            is_origin=False,
            error_type=None,
            depends_on=step_info["depends_on"],
        )
        reasoning_steps.append(step)
    
    # Build dependency graph
    nodes = [step.step_id for step in reasoning_steps]
    edges = []
    for step in reasoning_steps:
        for dep_id in step.depends_on:
            edges.append([dep_id, step.step_id])
    # Add sequential edges if no dependencies
    if not edges:
        edges = [[i, i + 1] for i in range(len(reasoning_steps) - 1)]
    
    dependency_graph = DependencyGraph(nodes=nodes, edges=edges)
    
    # Create correct chain
    chains = [ReasoningChain(
        domain=Domain.CODE,
        query_id=query_id,
        query=prompt,
        ground_truth=solution,
        reasoning_steps=reasoning_steps,
        dependency_graph=dependency_graph,
    )]
    
    # Generate incorrect variants if requested
    if generate_incorrect_variants:
        # Simple variant: introduce error in middle step
        if len(reasoning_steps) > 2:
            variant_steps = reasoning_steps.copy()
            error_step_idx = len(variant_steps) // 2
            variant_steps[error_step_idx] = ReasoningStep(
                step_id=variant_steps[error_step_idx].step_id,
                text=variant_steps[error_step_idx].text + " [ERROR: Incorrect logic]",
                is_correct=False,
                is_origin=True,
                error_type=ErrorType.LOGICAL,
                depends_on=variant_steps[error_step_idx].depends_on,
            )
            # Mark downstream steps as incorrect
            for i in range(error_step_idx + 1, len(variant_steps)):
                variant_steps[i] = ReasoningStep(
                    step_id=variant_steps[i].step_id,
                    text=variant_steps[i].text,
                    is_correct=False,
                    is_origin=False,
                    error_type=ErrorType.LOGICAL,
                    depends_on=variant_steps[i].depends_on,
                )
            
            chains.append(ReasoningChain(
                domain=Domain.CODE,
                query_id=f"{query_id}_variant",
                query=prompt,
                ground_truth=solution,
                reasoning_steps=variant_steps,
                dependency_graph=dependency_graph,
            ))
    
    return chains


def convert_humaneval_file(
    input_path: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
    augment_to: Optional[int] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Convert HumanEval JSONL file to unified format.
    
    Args:
        input_path: Path to input HumanEval JSONL.gz file
        output_path: Path to output unified format JSONL file
        max_samples: Maximum number of samples to process (None for all)
        augment_to: Target number of samples (will generate variants)
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
    
    # Read all problems first
    problems = []
    open_func = gzip.open if input_path.suffix == ".gz" else open
    mode = "rt" if input_path.suffix == ".gz" else "r"
    
    with open_func(input_path, mode) as f:
        for line in f:
            problems.append(json.loads(line))
    
    if max_samples:
        problems = problems[:max_samples]
    
    # Calculate augmentation factor
    augment_factor = 1
    if augment_to and len(problems) < augment_to:
        augment_factor = (augment_to // len(problems)) + 1
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for problem in tqdm(problems, desc=f"Converting {input_path.name}"):
            stats["total"] += 1
            
            try:
                # Generate chains (correct + variants if augmenting)
                generate_variants = augment_factor > 1
                chains = convert_humaneval_sample(
                    problem,
                    generate_incorrect_variants=generate_variants,
                )
                
                # Limit to augment_factor chains per problem
                chains = chains[:augment_factor]
                
                for chain in chains:
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
    input_path = project_root / "data/raw/human-eval/data/HumanEval.jsonl.gz"
    output_path = project_root / "data/processed/humaneval.jsonl"
    
    stats = convert_humaneval_file(
        input_path=input_path,
        output_path=output_path,
        augment_to=3000,  # Augment to 3K samples
    )
    
    print("\n=== Conversion Complete ===")
    print(f"Total problems: {stats['total']}")
    print(f"Converted chains: {stats['converted']}")
    print(f"Valid: {stats['valid']}")
    if stats['errors']:
        print(f"Errors: {len(stats['errors'])}")

