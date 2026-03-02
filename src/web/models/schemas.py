"""
Pydantic schemas for API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChainSummary(BaseModel):
    """Summary of a reasoning chain."""
    chain_id: str
    domain: str
    query: str
    num_steps: int
    has_errors: bool
    error_count: int


class ChainDetail(BaseModel):
    """Detailed chain information."""
    chain_id: str
    domain: str
    query: str
    ground_truth: str
    steps: List[Dict[str, Any]]
    dependency_graph: Dict[str, Any]


class GraphData(BaseModel):
    """Graph visualization data."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    model_name: str
    provider: str
    available: bool


class ComparisonRequest(BaseModel):
    """Request for model comparison."""
    model_ids: List[str]
    domain: Optional[str] = None


class FingerprintResponse(BaseModel):
    """Fingerprint data."""
    chain_id: str
    features: Dict[str, float]
    feature_vector: List[float]
    predicted_model: Optional[str] = None
    confidence: Optional[float] = None


class ConsensusRequest(BaseModel):
    """Request for consensus detection."""
    model_predictions: Dict[str, List[bool]]
    strategy: str = "majority_vote"


class ConsensusResponse(BaseModel):
    """Consensus detection result."""
    consensus_exists: bool
    consensus_confidence: float
    step_consensus: List[bool]
    disagreement_points: List[int]


class PatternInfo(BaseModel):
    """Pattern information."""
    pattern_id: str
    model_type: str
    domain: str
    hallucination_type: str
    frequency: int
    severity: float


class InferenceRequest(BaseModel):
    """Request for inference."""
    query: str
    domain: str
    model_types: List[str]
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    """Inference result."""
    chain_id: str
    model_type: str
    steps: List[str]
    latency: float
    error: Optional[str] = None


class TrainingRequest(BaseModel):
    """Request to start training."""
    domain: str
    model_type: str = "gat"
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainingStatus(BaseModel):
    """Training job status."""
    job_id: str
    status: str
    progress: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None


class ExperimentResult(BaseModel):
    """Experiment result."""
    experiment_id: str
    domain: str
    metrics: Dict[str, Any]
    timestamp: str

