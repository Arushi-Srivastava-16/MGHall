"""Weights & Biases configuration for experiment tracking."""

import wandb
from typing import Dict, Any, Optional


def init_wandb(
    project_name: str = "chg-framework",
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> None:
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        project_name: Name of the W&B project
        experiment_name: Name of this specific experiment
        config: Dictionary of hyperparameters and config
        tags: List of tags for organizing experiments
    """
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config or {},
        tags=tags or [],
    )


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number (epoch, iteration, etc.)
    """
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def log_graph_statistics(
    num_nodes: int,
    num_edges: int,
    avg_degree: float,
    max_depth: int,
    domain: str,
    step: Optional[int] = None,
) -> None:
    """
    Log graph structure statistics.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
        avg_degree: Average node degree
        max_depth: Maximum path depth
        domain: Domain name (math, code, medical)
        step: Optional step number
    """
    metrics = {
        f"graph/{domain}/num_nodes": num_nodes,
        f"graph/{domain}/num_edges": num_edges,
        f"graph/{domain}/avg_degree": avg_degree,
        f"graph/{domain}/max_depth": max_depth,
    }
    log_metrics(metrics, step)


def log_training_metrics(
    loss: float,
    accuracy: float,
    f1: float,
    domain: str,
    split: str = "train",
    step: Optional[int] = None,
) -> None:
    """
    Log training metrics.
    
    Args:
        loss: Loss value
        accuracy: Accuracy metric
        f1: F1 score
        domain: Domain name
        split: Data split (train, val, test)
        step: Optional step number
    """
    metrics = {
        f"{split}/{domain}/loss": loss,
        f"{split}/{domain}/accuracy": accuracy,
        f"{split}/{domain}/f1": f1,
    }
    log_metrics(metrics, step)

