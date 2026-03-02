"""
Fingerprint API endpoints.
"""

from fastapi import APIRouter, HTTPException
from src.web.models.schemas import FingerprintResponse
from src.web.services.chain_service import ChainService
from src.multi_model.fingerprint_extractor import FingerprintExtractor
from src.multi_model.fingerprint_classifier import FingerprintClassifier
from pathlib import Path

router = APIRouter()
chain_service = ChainService()
extractor = FingerprintExtractor()
classifier = None  # Will be loaded on demand


def load_classifier(domain: str = "math"):
    """Load fingerprint classifier."""
    global classifier
    if classifier is None:
        classifier_path = Path(__file__).parent.parent.parent.parent / f"models/fingerprint_classifier/{domain}_classifier.pkl"
        if classifier_path.exists():
            classifier = FingerprintClassifier.load(classifier_path)
    return classifier


@router.get("/{chain_id}", response_model=FingerprintResponse)
async def get_fingerprint(chain_id: str):
    """Get fingerprint for a chain."""
    chain_data = chain_service.get_chain(chain_id)
    if not chain_data:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    # Convert to ReasoningChain
    from src.data_processing.unified_schema import ReasoningChain
    chain = ReasoningChain.from_dict(chain_data)
    
    # Extract fingerprint
    fingerprint = extractor.extract(chain)
    feature_vector = extractor.get_feature_vector(fingerprint)
    
    # Try to classify
    predicted_model = None
    confidence = None
    try:
        clf = load_classifier(chain_data["domain"])
        if clf:
            predicted_model, confidence = clf.predict(chain)
            predicted_model = predicted_model.value if hasattr(predicted_model, 'value') else str(predicted_model)
    except:
        pass
    
    return FingerprintResponse(
        chain_id=chain_id,
        features=fingerprint,
        feature_vector=feature_vector.tolist(),
        predicted_model=predicted_model,
        confidence=confidence,
    )


@router.post("/classify")
async def classify_chain(chain_id: str):
    """Classify which model generated a chain."""
    fingerprint = await get_fingerprint(chain_id)
    return {
        "chain_id": chain_id,
        "predicted_model": fingerprint.predicted_model,
        "confidence": fingerprint.confidence,
    }


@router.get("/features/importance")
async def get_feature_importance(domain: str = "math"):
    """Get feature importance from classifier."""
    clf = load_classifier(domain)
    if not clf:
        raise HTTPException(status_code=404, detail="Classifier not found")
    
    try:
        importance = clf.get_feature_importance()
        return {"features": importance}
    except:
        raise HTTPException(status_code=500, detail="Could not get feature importance")

