"""
Results API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List
from src.web.models.schemas import ExperimentResult
from pathlib import Path
import json
import os

router = APIRouter()


@router.get("/experiments", response_model=List[ExperimentResult])
async def list_experiments():
    """List all experiments."""
    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / "experiments"
    
    experiments = []
    
    # Look for result JSON files
    result_files = [
        "cross_domain_results.json",
        "ensemble_results.json",
        "error_propagation_analysis.json",
        "evaluation_summary.json",
        "proactive_pipeline_results.json",
        "training_summary.json",
    ]
    
    for result_file in result_files:
        file_path = results_dir / result_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                experiments.append(ExperimentResult(
                    experiment_id=result_file.replace(".json", ""),
                    domain=data.get("domain", "unknown"),
                    metrics=data,
                    timestamp=data.get("timestamp", ""),
                ))
    
    return experiments


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / "experiments"
    file_path = results_dir / f"{experiment_id}.json"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    with open(file_path, 'r') as f:
        return json.load(f)


@router.get("/metrics/aggregate")
async def get_aggregate_metrics():
    """Get aggregate metrics across all experiments."""
    experiments = await list_experiments()
    
    return {
        "total_experiments": len(experiments),
        "domains": list(set(e.domain for e in experiments)),
        "experiments": [e.experiment_id for e in experiments],
    }

