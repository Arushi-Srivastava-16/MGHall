"""
Inference API endpoints.
"""

from fastapi import APIRouter, HTTPException
from src.web.models.schemas import InferenceRequest, InferenceResponse
from src.web.services.model_service import ModelService
from src.multi_model.llm_inference import MultiModelInference
from src.multi_model.model_config import ModelType
from src.data_processing.unified_schema import Domain

router = APIRouter()
model_service = ModelService()


@router.post("/generate", response_model=InferenceResponse)
async def generate_chain(request: InferenceRequest):
    """Generate reasoning chain from query."""
    try:
        # Initialize models
        model_types = [ModelType(mt) for mt in request.model_types]
        multi_model = MultiModelInference(model_types)
        
        # Run inference
        domain = Domain(request.domain)
        results = multi_model.infer_all(request.query, domain, request.temperature)
        
        # Return first result (or could return all)
        if not results:
            raise HTTPException(status_code=500, detail="No models available")
        
        first_result = next(iter(results.values()))
        
        return InferenceResponse(
            chain_id=f"inference_{first_result.model_type.value}",
            model_type=first_result.model_type.value,
            steps=first_result.reasoning_steps,
            latency=first_result.latency,
            error=first_result.error,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_inference(request: InferenceRequest):
    """Batch inference (placeholder)."""
    return {"message": "Batch inference not yet implemented"}

