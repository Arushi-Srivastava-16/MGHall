"""
Graph Statistics and Visualization.

This module provides tools for analyzing and visualizing CRG structures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

from ..data_processing.unified_schema import ReasoningChain, Domain


def compute_graph_statistics(chain: ReasoningChain) -> Dict[str, Any]:
    """
    Compute statistics for a reasoning chain graph.
    
    Args:
        chain: ReasoningChain to analyze
        
    Returns:
        Dictionary with graph statistics
    """
    G = nx.DiGraph()
    G.add_nodes_from(chain.dependency_graph.nodes)
    G.add_edges_from(chain.dependency_graph.edges)
    
    stats = {
        "num_nodes": len(chain.reasoning_steps),
        "num_edges": len(chain.dependency_graph.edges),
        "avg_degree": 2 * len(chain.dependency_graph.edges) / max(len(chain.reasoning_steps), 1),
        "max_depth": 0,
        "density": nx.density(G),
        "is_dag": nx.is_directed_acyclic_graph(G),
    }
    
    # Compute max depth
    if G.number_of_nodes() > 0:
        try:
            longest_path = nx.dag_longest_path(G)
            stats["max_depth"] = len(longest_path) - 1 if longest_path else 0
        except:
            stats["max_depth"] = stats["num_nodes"] - 1
    
    # Error statistics
    incorrect_steps = [step for step in chain.reasoning_steps if not step.is_correct]
    origin_steps = [step for step in chain.reasoning_steps if step.is_origin]
    
    stats["num_errors"] = len(incorrect_steps)
    stats["error_rate"] = len(incorrect_steps) / max(len(chain.reasoning_steps), 1)
    stats["has_origin"] = len(origin_steps) > 0
    if origin_steps:
        stats["origin_position"] = origin_steps[0].step_id
    
    return stats


def plot_degree_distribution(
    chains: List[ReasoningChain],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot degree distribution across chains.
    
    Args:
        chains: List of reasoning chains
        output_path: Optional path to save figure
    """
    degrees = []
    for chain in chains:
        G = nx.DiGraph()
        G.add_nodes_from(chain.dependency_graph.nodes)
        G.add_edges_from(chain.dependency_graph.edges)
        
        for node in G.nodes():
            degrees.append(G.degree(node))
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20, edgecolor='black')
    plt.xlabel('Node Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution Across All Graphs')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_chain_length_distribution(
    chains: List[ReasoningChain],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot distribution of chain lengths.
    
    Args:
        chains: List of reasoning chains
        output_path: Optional path to save figure
    """
    lengths = [len(chain.reasoning_steps) for chain in chains]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor='black')
    plt.xlabel('Chain Length (Number of Steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reasoning Chain Lengths')
    plt.axvline(np.mean(lengths), color='r', linestyle='--', label=f'Mean: {np.mean(lengths):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_propagation(
    chain: ReasoningChain,
    output_path: Optional[Path] = None,
) -> None:
    """
    Visualize error propagation in a single chain.
    
    Args:
        chain: ReasoningChain to visualize
        output_path: Optional path to save figure
    """
    G = nx.DiGraph()
    G.add_nodes_from(chain.dependency_graph.nodes)
    G.add_edges_from(chain.dependency_graph.edges)
    
    # Color nodes by correctness
    node_colors = []
    for step in chain.reasoning_steps:
        if step.is_origin:
            node_colors.append('red')  # Origin of error
        elif not step.is_correct:
            node_colors.append('orange')  # Propagated error
        else:
            node_colors.append('green')  # Correct
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"Error Propagation in Chain {chain.query_id}")
    plt.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Correct'),
        Patch(facecolor='orange', label='Propagated Error'),
        Patch(facecolor='red', label='Error Origin'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_domain_comparison_report(
    chains: List[ReasoningChain],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate comparison report across domains.
    
    Args:
        chains: List of reasoning chains
        output_path: Optional path to save report
        
    Returns:
        Dictionary with comparison statistics
    """
    domain_stats = defaultdict(list)
    
    for chain in chains:
        domain = chain.domain.value if isinstance(chain.domain, Domain) else chain.domain
        stats = compute_graph_statistics(chain)
        domain_stats[domain].append(stats)
    
    report = {}
    for domain, stats_list in domain_stats.items():
        report[domain] = {
            "num_chains": len(stats_list),
            "avg_nodes": np.mean([s["num_nodes"] for s in stats_list]),
            "avg_edges": np.mean([s["num_edges"] for s in stats_list]),
            "avg_depth": np.mean([s["max_depth"] for s in stats_list]),
            "avg_error_rate": np.mean([s["error_rate"] for s in stats_list]),
        }
    
    # Create visualization
    if output_path:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        domains = list(report.keys())
        metrics = ["avg_nodes", "avg_edges", "avg_depth", "avg_error_rate"]
        metric_labels = ["Avg Nodes", "Avg Edges", "Avg Depth", "Avg Error Rate"]
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            values = [report[d][metric] for d in domains]
            ax.bar(domains, values)
            ax.set_ylabel(label)
            ax.set_title(f"{label} by Domain")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return report


if __name__ == "__main__":
    # Example usage
    from ..data_processing.unified_schema import ReasoningStep, DependencyGraph
    
    # Create sample chains
    chains = []
    for i in range(5):
        steps = [
            ReasoningStep(step_id=0, text=f"Step 1 chain {i}", is_correct=True, depends_on=[]),
            ReasoningStep(step_id=1, text=f"Step 2 chain {i}", is_correct=False, is_origin=True, depends_on=[0]),
        ]
        chain = ReasoningChain(
            domain=Domain.MATH,
            query_id=f"test_{i}",
            query="Test",
            ground_truth="Answer",
            reasoning_steps=steps,
            dependency_graph=DependencyGraph(nodes=[0, 1], edges=[[0, 1]]),
        )
        chains.append(chain)
    
    # Generate visualizations
    output_dir = Path("../../experiments/results")
    plot_degree_distribution(chains, output_dir / "degree_dist.png")
    plot_chain_length_distribution(chains, output_dir / "chain_length_dist.png")
    plot_error_propagation(chains[0], output_dir / "error_propagation_example.png")
    report = generate_domain_comparison_report(chains, output_dir / "domain_comparison.png")
    
    print("Visualizations generated!")

