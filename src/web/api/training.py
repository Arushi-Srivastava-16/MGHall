"""
Training API endpoints.
"""

from fastapi import APIRouter, HTTPException
from src.web.models.schemas import TrainingRequest, TrainingStatus
import uuid
from datetime import datetime

router = APIRouter()
training_jobs = {}  # In-memory job storage (use Redis in production)


@router.post("/start", response_model=TrainingStatus)
async def start_training(request: TrainingRequest):
    """Start a training job."""
    job_id = str(uuid.uuid4())
    
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "domain": request.domain,
        "model_type": request.model_type,
        "config": request.config,
        "started_at": datetime.now().isoformat(),
    }
    
    # In a real implementation, this would queue a background job
    # For now, return the job status
    
    return TrainingStatus(
        job_id=job_id,
        status="queued",
        progress=0.0,
    )


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    return TrainingStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        current_epoch=job.get("current_epoch"),
        total_epochs=job.get("total_epochs"),
    )


@router.get("/results/{job_id}")
async def get_training_results(job_id: str):
    """Get training results."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    return {
        "job_id": job_id,
        "results": job.get("results", {}),
    }

