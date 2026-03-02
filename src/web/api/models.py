"""
Model API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List
from src.web.models.schemas import ModelInfo, ComparisonRequest
from src.web.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()


@router.get("/", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    return model_service.list_models()


@router.get("/{model_id}/info", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """Get model information."""
    info = model_service.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return info


@router.post("/compare")
async def compare_models(request: ComparisonRequest):
    """Compare multiple models."""
    return model_service.compare_models(
        model_ids=request.model_ids,
        domain=request.domain,
    )

