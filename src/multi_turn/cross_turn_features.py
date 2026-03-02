"""
Cross-Turn Feature Extraction for Memory-Augmented CHG.

Extracts memory-aware features that capture cross-turn consistency.
"""

from typing import Dict, List, Any
import numpy as np
from .conversation_memory import ConversationMemory, Entity, Claim


def extract_memory_features(
    reasoning_chain: Dict,
    conversation_memory: ConversationMemory,
    current_turn: int
) -> np.ndarray:
    """
    Extract memory-aware features for a reasoning chain.
    
    Args:
        reasoning_chain: Current reasoning chain with steps
        conversation_memory: Memory from previous turns
        current_turn: Current turn number
        
    Returns:
        Feature vector with cross-turn signals
    """
    features = {}
    
    # Get memory context
    memory_context = conversation_memory.get_memory_context()
    
    # 1. Contradiction score
    contradictions = conversation_memory.check_consistency(reasoning_chain, current_turn)
    features['num_contradictions'] = len(contradictions)
    features['has_entity_contradiction'] = any(
        c['type'] == 'entity_contradiction' for c in contradictions
    )
    features['has_claim_contradiction'] = any(
        c['type'] == 'claim_contradiction' for c in contradictions
    )
    
    # 2. Entity consistency
    entity_features = _compute_entity_features(
        reasoning_chain, conversation_memory, current_turn
    )
    features.update(entity_features)
    
    # 3. Claim verification
    claim_features = _compute_claim_features(
        reasoning_chain, conversation_memory, current_turn
    )
    features.update(claim_features)
    
    # 4. Temporal decay
    temporal_features = _compute_temporal_features(
        reasoning_chain, conversation_memory, current_turn
    )
    features.update(temporal_features)
    
    # Convert to array
    feature_vector = np.array([
        features.get('num_contradictions', 0),
        1.0 if features.get('has_entity_contradiction') else 0.0,
        1.0 if features.get('has_claim_contradiction') else 0.0,
        features.get('entity_mentions', 0),
        features.get('entity_value_changes', 0),
        features.get('claim_support_count', 0),
        features.get('claim_conflict_count', 0),
        features.get('confidence_decay_avg', 1.0),
        features.get('turn_distance_avg', 0.0),
    ])
    
    return feature_vector


def _compute_entity_features(
    reasoning_chain: Dict,
    memory: ConversationMemory,
    current_turn: int
) -> Dict[str, float]:
    """Compute entity-related features."""
    features = {}
    
    # Extract entities from current chain
    current_entities = memory._extract_entities_from_crg(reasoning_chain, current_turn)
    
    # Count entity mentions
    features['entity_mentions'] = len(current_entities)
    
    # Count entity value changes
    value_changes = 0
    for entity in current_entities:
        prev_history = memory.entity_tracker.get_entity_history(entity.name)
        if prev_history:
            # Check if value changed
            prev_values = [e.value for e in prev_history]
            if entity.value not in prev_values:
                value_changes += 1
    
    features['entity_value_changes'] = value_changes
    
    # Entity consistency ratio
    if len(current_entities) > 0:
        features['entity_consistency_ratio'] = 1.0 - (value_changes / len(current_entities))
    else:
        features['entity_consistency_ratio'] = 1.0
    
    return features


def _compute_claim_features(
    reasoning_chain: Dict,
    memory: ConversationMemory,
    current_turn: int
) -> Dict[str, float]:
    """Compute claim-related features."""
    features = {}
    
    # Extract claims from current chain
    current_claims = memory._extract_claims_from_crg(reasoning_chain, current_turn)
    
    # Count supporting claims
    support_count = 0
    conflict_count = 0
    
    for claim in current_claims:
        # Find supporting claims
        supporting = memory.claim_database.find_supporting_claims(claim)
        if supporting:
            support_count += len(supporting)
        
        # Check for conflicts
        conflict = memory.claim_database.check_consistency(claim)
        if conflict:
            conflict_count += 1
    
    features['claim_support_count'] = support_count
    features['claim_conflict_count'] = conflict_count
    
    # Claim verification ratio
    total_claims = len(current_claims)
    if total_claims > 0:
        features['claim_verification_ratio'] = support_count / total_claims
        features['claim_conflict_ratio'] = conflict_count / total_claims
    else:
        features['claim_verification_ratio'] = 0.0
        features['claim_conflict_ratio'] = 0.0
    
    return features


def _compute_temporal_features(
    reasoning_chain: Dict,
    memory: ConversationMemory,
    current_turn: int
) -> Dict[str, float]:
    """Compute temporal decay features."""
    features = {}
    
    # Get recent turns
    recent_turns = memory.crg_history[-3:] if memory.crg_history else []
    
    if not recent_turns:
        features['confidence_decay_avg'] = 1.0
        features['turn_distance_avg'] = 0.0
        return features
    
    # Compute average confidence decay
    decay_values = []
    turn_distances = []
    
    for past_turn in recent_turns:
        age = past_turn.get('age', 0)
        decay = memory.compute_confidence_decay(age)
        decay_values.append(decay)
        turn_distances.append(current_turn - past_turn.get('turn_id', current_turn))
    
    features['confidence_decay_avg'] = np.mean(decay_values) if decay_values else 1.0
    features['turn_distance_avg'] = np.mean(turn_distances) if turn_distances else 0.0
    features['confidence_decay_min'] = np.min(decay_values) if decay_values else 1.0
    
    return features


def compute_contradiction_score(
    reasoning_chain: Dict,
    conversation_memory: ConversationMemory,
    current_turn: int
) -> float:
    """
    Compute overall contradiction score for the chain.
    
    Returns:
        Score from 0.0 (no contradictions) to 1.0 (severe contradictions)
    """
    contradictions = conversation_memory.check_consistency(reasoning_chain, current_turn)
    
    if not contradictions:
        return 0.0
    
    # Weight different contradiction types
    weights = {
        'entity_contradiction': 0.8,
        'claim_contradiction': 1.0,
    }
    
    total_score = 0.0
    for contradiction in contradictions:
        contradiction_type = contradiction['type']
        weight = weights.get(contradiction_type, 0.5)
        total_score += weight
    
    # Normalize by number of steps
    num_steps = len(reasoning_chain.get('reasoning_steps', []))
    if num_steps > 0:
        total_score /= num_steps
    
    # Clip to [0, 1]
    return min(1.0, total_score)


# Example usage
if __name__ == "__main__":
    from conversation_memory import ConversationMemory
    
    # Create memory
    memory = ConversationMemory()
    
    # Add turn 1
    turn1 = {
        'reasoning_steps': [
            {'text': 'Let x = 5'},
            {'text': 'Sarah lives in Boston'},
        ]
    }
    memory.add_turn(turn1, turn_id=0)
    
    # Turn 2 (has contradiction)
    turn2 = {
        'reasoning_steps': [
            {'text': 'Now x = 3'},  # Contradicts x=5
            {'text': 'Emma went to the store'},  # Different name
        ]
    }
    
    # Extract features
    features = extract_memory_features(turn2, memory, current_turn=1)
    print("Memory Features:", features)
    
    # Compute contradiction score
    score = compute_contradiction_score(turn2, memory, current_turn=1)
    print(f"Contradiction Score: {score:.3f}")
    
    print("\n✓ Cross-turn feature extraction demo complete!")
