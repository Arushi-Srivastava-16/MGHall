"""Multi-turn package for cross-conversation hallucination detection."""

from .conversation_memory import (
    ConversationMemory,
    EntityTracker,
    ClaimDatabase,
    Entity,
    Claim,
)

from .cross_turn_features import (
    extract_memory_features,
    compute_contradiction_score,
)

__all__ = [
    'ConversationMemory',
    'EntityTracker',
    'ClaimDatabase',
    'Entity',
    'Claim',
    'extract_memory_features',
    'compute_contradiction_score',
]
