"""
Chain API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from src.web.models.schemas import ChainSummary, ChainDetail, GraphData
from src.web.services.chain_service import ChainService

router = APIRouter()
chain_service = ChainService()


@router.get("/", response_model=List[ChainSummary])
async def list_chains(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    limit: int = Query(100, description="Maximum number of chains"),
):
    """List available reasoning chains."""
    chains = chain_service.list_chains(domain=domain, limit=limit)
    return chains


@router.get("/{chain_id}", response_model=ChainDetail)
async def get_chain(chain_id: str):
    """Get chain details by ID."""
    chain = chain_service.get_chain(chain_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    return chain


@router.get("/{chain_id}/graph", response_model=GraphData)
async def get_chain_graph(chain_id: str):
    """Get graph visualization data for a chain."""
    graph_data = chain_service.get_graph_data(chain_id)
    if not graph_data:
        raise HTTPException(status_code=404, detail="Chain not found")
    return graph_data

