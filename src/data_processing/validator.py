"""
Data quality validation pipeline.

This module validates converted reasoning chains and generates quality reports.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .unified_schema import ReasoningChain, validate_reasoning_chain, Domain
except ImportError:
    from unified_schema import ReasoningChain, validate_reasoning_chain, Domain


def validate_file(input_path: Path) -> Dict[str, Any]:
    """
    Validate all chains in a JSONL file.
    
    Args:
        input_path: Path to JSONL file with reasoning chains
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "domain_counts": defaultdict(int),
        "chain_lengths": [],
        "error_rates": [],
        "origin_positions": [],
    }
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Validating {input_path.name}"):
            stats["total"] += 1
            
            try:
                chain_dict = json.loads(line)
                chain = ReasoningChain.from_dict(chain_dict)
                
                # Track domain
                domain = chain.domain.value if isinstance(chain.domain, Domain) else chain.domain
                stats["domain_counts"][domain] += 1
                
                # Track chain length
                stats["chain_lengths"].append(len(chain.reasoning_steps))
                
                # Track error rate
                incorrect_steps = sum(1 for step in chain.reasoning_steps if not step.is_correct)
                error_rate = incorrect_steps / len(chain.reasoning_steps) if chain.reasoning_steps else 0
                stats["error_rates"].append(error_rate)
                
                # Track origin positions
                origin_steps = [step.step_id for step in chain.reasoning_steps if step.is_origin]
                if origin_steps:
                    stats["origin_positions"].append(origin_steps[0])
                
                # Validate structure
                validation_errors = validate_reasoning_chain(chain)
                if validation_errors:
                    stats["invalid"] += 1
                    stats["errors"].extend(validation_errors)
                else:
                    stats["valid"] += 1
                    
            except Exception as e:
                stats["invalid"] += 1
                stats["errors"].append(f"Parse error: {str(e)}")
    
    return stats


def check_graph_connectivity(chain: ReasoningChain) -> List[str]:
    """
    Check that graph is connected (no isolated nodes).
    
    Args:
        chain: ReasoningChain to check
        
    Returns:
        List of connectivity issues (empty if connected)
    """
    issues = []
    
    # Build adjacency list
    adj = defaultdict(set)
    for edge in chain.dependency_graph.edges:
        if len(edge) == 2:
            adj[edge[0]].add(edge[1])
            adj[edge[1]].add(edge[0])  # Undirected for connectivity check
    
    # Check all nodes are reachable from node 0
    visited = set()
    stack = [0] if chain.reasoning_steps else []
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                stack.append(neighbor)
    
    # Check for isolated nodes
    all_nodes = set(chain.dependency_graph.nodes)
    isolated = all_nodes - visited
    if isolated:
        issues.append(f"Isolated nodes found: {sorted(isolated)}")
    
    return issues


def verify_label_consistency(chain: ReasoningChain) -> List[str]:
    """
    Verify that labels are consistent (origin → propagation).
    
    Args:
        chain: ReasoningChain to check
        
    Returns:
        List of consistency issues (empty if consistent)
    """
    issues = []
    
    origin_steps = [step for step in chain.reasoning_steps if step.is_origin]
    
    if len(origin_steps) > 1:
        issues.append(f"Multiple origin steps: {[s.step_id for s in origin_steps]}")
    
    if origin_steps:
        origin_id = origin_steps[0].step_id
        
        # Check that origin is incorrect
        if origin_steps[0].is_correct:
            issues.append(f"Origin step {origin_id} is marked as correct")
        
        # Check that downstream steps after origin are also incorrect
        # (if they depend on the origin)
        origin_dependents = set()
        for edge in chain.dependency_graph.edges:
            if edge[0] == origin_id:
                origin_dependents.add(edge[1])
        
        for step in chain.reasoning_steps:
            if step.step_id in origin_dependents and step.is_correct:
                issues.append(f"Step {step.step_id} depends on origin but is marked correct")
    
    return issues


def generate_quality_report(
    input_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive quality report for a dataset file.
    
    Args:
        input_path: Path to JSONL file with reasoning chains
        output_path: Optional path to save report JSON
        
    Returns:
        Dictionary with quality metrics
    """
    stats = validate_file(input_path)
    
    # Calculate statistics
    import numpy as np
    
    report = {
        "file": str(input_path),
        "total_samples": stats["total"],
        "valid_samples": stats["valid"],
        "invalid_samples": stats["invalid"],
        "validity_rate": stats["valid"] / stats["total"] if stats["total"] > 0 else 0,
        "domain_distribution": dict(stats["domain_counts"]),
        "chain_length_stats": {
            "mean": float(np.mean(stats["chain_lengths"])) if stats["chain_lengths"] else 0,
            "median": float(np.median(stats["chain_lengths"])) if stats["chain_lengths"] else 0,
            "min": int(np.min(stats["chain_lengths"])) if stats["chain_lengths"] else 0,
            "max": int(np.max(stats["chain_lengths"])) if stats["chain_lengths"] else 0,
            "std": float(np.std(stats["chain_lengths"])) if stats["chain_lengths"] else 0,
        },
        "error_rate_stats": {
            "mean": float(np.mean(stats["error_rates"])) if stats["error_rates"] else 0,
            "median": float(np.median(stats["error_rates"])) if stats["error_rates"] else 0,
        },
        "origin_position_stats": {
            "mean": float(np.mean(stats["origin_positions"])) if stats["origin_positions"] else 0,
            "median": float(np.median(stats["origin_positions"])) if stats["origin_positions"] else 0,
        },
        "error_summary": {
            "total_errors": len(stats["errors"]),
            "unique_errors": len(set(stats["errors"])),
            "error_types": dict(Counter(stats["errors"])),
        },
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    
    return report


def validate_all_datasets(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Validate all converted datasets in a directory.
    
    Args:
        data_dir: Directory containing converted JSONL files
        
    Returns:
        Dictionary mapping filename to quality report
    """
    reports = {}
    
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        print(f"\nValidating {jsonl_file.name}...")
        report = generate_quality_report(
            jsonl_file,
            output_path=data_dir / f"{jsonl_file.stem}_quality_report.json",
        )
        reports[jsonl_file.name] = report
        
        print(f"  Valid: {report['valid_samples']}/{report['total_samples']} ({report['validity_rate']:.2%})")
        print(f"  Mean chain length: {report['chain_length_stats']['mean']:.2f}")
        print(f"  Mean error rate: {report['error_rate_stats']['mean']:.2%}")
    
    return reports


if __name__ == "__main__":
    # Example usage - use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data/processed"
    reports = validate_all_datasets(data_dir)
    
    print("\n=== Validation Complete ===")
    for filename, report in reports.items():
        print(f"\n{filename}:")
        print(f"  Validity: {report['validity_rate']:.2%}")
        print(f"  Domains: {report['domain_distribution']}")

