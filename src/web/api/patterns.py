"""
Pattern API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from src.web.models.schemas import PatternInfo
from src.multi_model.pattern_database import PatternDatabase
from pathlib import Path

router = APIRouter()


def load_pattern_db(domain: str = "math") -> PatternDatabase:
    """Load pattern database."""
    db_path = Path(__file__).parent.parent.parent.parent / f"data/multi_model/patterns/{domain}_patterns.json"
    return PatternDatabase(db_path if db_path.exists() else None)


@router.get("/", response_model=List[PatternInfo])
async def list_patterns(
    domain: Optional[str] = Query(None),
    model_type: Optional[str] = Query(None),
    pattern_type: Optional[str] = Query(None),
):
    """List hallucination patterns."""
    db = load_pattern_db(domain or "math")
    
    # If database is empty, return empty list
    if not db.patterns:
        return []
    
    patterns = list(db.patterns.values())
    
    # Filter by model type
    if model_type:
        patterns = [p for p in patterns if p.model_type.value == model_type]
    
    # Filter by pattern type
    if pattern_type:
        patterns = [p for p in patterns if p.hallucination_type.value == pattern_type]
    
    return [
        PatternInfo(
            pattern_id=p.pattern_id,
            model_type=p.model_type.value,
            domain=p.domain.value,
            hallucination_type=p.hallucination_type.value,
            frequency=p.frequency,
            severity=p.severity,
        )
        for p in patterns
    ]


@router.get("/stats")
async def get_pattern_stats(domain: Optional[str] = None):
    """Get pattern statistics."""
    db = load_pattern_db(domain or "math")
    if not db.patterns:
        return {
            "total_patterns": 0,
            "models": [],
            "domains": [],
            "pattern_types": {},
        }
    return db.get_summary()


@router.get("/{pattern_id}")
async def get_pattern(pattern_id: str):
    """Get pattern details."""
    db = load_pattern_db()
    if pattern_id not in db.patterns:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    pattern = db.patterns[pattern_id]
    return pattern.to_dict()

