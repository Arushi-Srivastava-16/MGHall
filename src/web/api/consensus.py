"""
Consensus API endpoints.
"""

from fastapi import APIRouter, HTTPException
from src.web.models.schemas import ConsensusRequest, ConsensusResponse
from src.multi_model.consensus_detector import ConsensusDetector, ConsensusStrategy
from src.multi_model.model_config import ModelType

router = APIRouter()


@router.get("/strategies")
async def list_strategies():
    """List available consensus strategies."""
    return {
        "strategies": [
            {"id": "majority_vote", "name": "Majority Vote"},
            {"id": "weighted_vote", "name": "Weighted Vote"},
            {"id": "unanimous", "name": "Unanimous"},
            {"id": "expert_selection", "name": "Expert Selection"},
        ]
    }


@router.post("/detect", response_model=ConsensusResponse)
async def detect_consensus(request: ConsensusRequest):
    """Detect consensus across model predictions."""
    try:
        # Convert model IDs to ModelType
        model_predictions = {}
        for model_id, predictions in request.model_predictions.items():
            try:
                model_type = ModelType(model_id)
                model_predictions[model_type] = predictions
            except ValueError:
                # If not a ModelType, use string as-is (for compatibility)
                model_predictions[model_id] = predictions
        
        # Create detector
        strategy_map = {
            "majority_vote": ConsensusStrategy.MAJORITY_VOTE,
            "weighted_vote": ConsensusStrategy.WEIGHTED_VOTE,
            "unanimous": ConsensusStrategy.UNANIMOUS,
            "expert_selection": ConsensusStrategy.EXPERT_SELECTION,
        }
        
        strategy = strategy_map.get(request.strategy, ConsensusStrategy.MAJORITY_VOTE)
        detector = ConsensusDetector(strategy=strategy)
        
        # Detect consensus
        result = detector.detect_consensus("temp_chain", model_predictions)
        
        return ConsensusResponse(
            consensus_exists=result.consensus_exists,
            consensus_confidence=result.consensus_confidence,
            step_consensus=result.step_consensus,
            disagreement_points=result.disagreement_points,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

