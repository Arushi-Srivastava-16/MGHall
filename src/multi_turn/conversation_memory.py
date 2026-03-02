"""
Multi-Turn Conversation Memory for CHG.

This module tracks conversation state across multiple turns to detect
cross-turn hallucinations like entity contradictions and claim reversals.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class Entity:
    """Represents an entity mentioned in conversation."""
    name: str
    entity_type: str  # person, number, location, etc.
    value: Any
    turn_id: int
    step_id: int
    
    def __hash__(self):
        return hash((self.name, self.entity_type))


@dataclass
class Claim:
    """Represents a factual claim made in conversation."""
    subject: str
    predicate: str
    object: Any
    turn_id: int
    step_id: int
    confidence: float = 1.0


class EntityTracker:
    """Tracks entities across conversation turns."""
    
    def __init__(self):
        self.entities: Dict[str, List[Entity]] = defaultdict(list)
        
    def add_entity(self, entity: Entity):
        """Add entity mention."""
        self.entities[entity.name].append(entity)
        
    def check_consistency(self, entity: Entity) -> Optional[Dict]:
        """Check if entity is consistent with previous mentions."""
        if entity.name not in self.entities:
            return None
            
        previous_entities = self.entities[entity.name]
        
        for prev_entity in previous_entities:
            # Check for value contradictions
            if prev_entity.value != entity.value:
                return {
                    'type': 'entity_contradiction',
                    'entity_name': entity.name,
                    'previous_value': prev_entity.value,
                    'current_value': entity.value,
                    'previous_turn': prev_entity.turn_id,
                    'current_turn': entity.turn_id,
                }
        
        return None
    
    def get_entity_history(self, entity_name: str) -> List[Entity]:
        """Get all mentions of an entity."""
        return self.entities.get(entity_name, [])


class ClaimDatabase:
    """Stores and verifies factual claims."""
    
    def __init__(self):
        self.claims: List[Claim] = []
        
    def add_claim(self, claim: Claim):
        """Add a claim to database."""
        self.claims.append(claim)
        
    def check_consistency(self, claim: Claim) -> Optional[Dict]:
        """Check if claim contradicts previous claims."""
        for prev_claim in self.claims:
            # Same subject + predicate, different object = contradiction
            if (prev_claim.subject == claim.subject and
                prev_claim.predicate == claim.predicate and
                prev_claim.object != claim.object):
                
                return {
                    'type': 'claim_contradiction',
                    'subject': claim.subject,
                    'predicate': claim.predicate,
                    'previous_value': prev_claim.object,
                    'current_value': claim.object,
                    'previous_turn': prev_claim.turn_id,
                    'current_turn': claim.turn_id,
                }
        
        return None
    
    def find_supporting_claims(self, claim: Claim) -> List[Claim]:
        """Find claims that support this claim."""
        supporting = []
        for prev_claim in self.claims:
            if (prev_claim.subject == claim.subject and
                prev_claim.predicate == claim.predicate and
                prev_claim.object == claim.object):
                supporting.append(prev_claim)
        return supporting


class ConversationMemory:
    """
    Manages multi-turn conversation history.
    
    Tracks:
    - Previous CRGs from earlier turns
    - Entity mentions and values
    - Factual claims made
    - Temporal decay of confidence
    """
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.crg_history: List[Dict] = []
        self.entity_tracker = EntityTracker()
        self.claim_database = ClaimDatabase()
        
    def add_turn(self, crg_data: Dict, turn_id: int):
        """
        Add a new turn to memory.
        
        Args:
            crg_data: CRG representation with nodes and edges
            turn_id: Turn number in conversation
        """
        # Add metadata
        crg_data['turn_id'] = turn_id
        crg_data['age'] = 0
        
        # Age existing turns
        for past_crg in self.crg_history:
            past_crg['age'] += 1
        
        # Add to history
        self.crg_history.append(crg_data)
        
        # Limit history size
        if len(self.crg_history) > self.max_turns:
            self.crg_history.pop(0)
        
        # Extract and track entities
        self._extract_entities(crg_data, turn_id)
        
        # Extract and track claims
        self._extract_claims(crg_data, turn_id)
    
    def check_consistency(self, current_crg: Dict, current_turn: int) -> List[Dict]:
        """
        Check current CRG for contradictions with past turns.
        
        Returns:
            List of contradiction objects with details
        """
        contradictions = []
        
        # Extract entities from current turn
        current_entities = self._extract_entities_from_crg(current_crg, current_turn)
        
        # Check each entity
        for entity in current_entities:
            issue = self.entity_tracker.check_consistency(entity)
            if issue:
                contradictions.append(issue)
        
        # Extract claims from current turn
        current_claims = self._extract_claims_from_crg(current_crg, current_turn)
        
        # Check each claim
        for claim in current_claims:
            issue = self.claim_database.check_consistency(claim)
            if issue:
                contradictions.append(issue)
        
        return contradictions
    
    def get_memory_context(self) -> Dict[str, Any]:
        """
        Get memory context for current turn.
        
        Returns:
            Dictionary with entity history, claims, etc.
        """
        return {
            'num_previous_turns': len(self.crg_history),
            'entity_tracker': self.entity_tracker,
            'claim_database': self.claim_database,
            'recent_turns': self.crg_history[-3:] if self.crg_history else [],
        }
    
    def compute_confidence_decay(self, turn_age: int, base_confidence: float = 1.0) -> float:
        """
        Compute confidence decay based on turn age.
        
        Older turns have lower confidence (temporal discount).
        """
        decay_rate = 0.1
        return base_confidence * (1.0 - decay_rate * turn_age)
    
    def _extract_entities(self, crg_data: Dict, turn_id: int):
        """Extract entities from CRG and add to tracker."""
        entities = self._extract_entities_from_crg(crg_data, turn_id)
        for entity in entities:
            self.entity_tracker.add_entity(entity)
    
    def _extract_claims(self, crg_data: Dict, turn_id: int):
        """Extract claims from CRG and add to database."""
        claims = self._extract_claims_from_crg(crg_data, turn_id)
        for claim in claims:
            self.claim_database.add_claim(claim)
    
    def _extract_entities_from_crg(self, crg_data: Dict, turn_id: int) -> List[Entity]:
        """
        Extract entities from CRG nodes.
        
        Simple heuristic-based extraction for now.
        """
        entities = []
        
        # Get reasoning steps
        steps = crg_data.get('reasoning_steps', [])
        
        for step_idx, step in enumerate(steps):
            text = step.get('text', '')
            
            # Extract numbers (e.g., "x = 5")
            number_matches = re.findall(r'(\w+)\s*=\s*(\d+\.?\d*)', text)
            for var, value in number_matches:
                entities.append(Entity(
                    name=var,
                    entity_type='number',
                    value=float(value),
                    turn_id=turn_id,
                    step_id=step_idx
                ))
            
            # Extract proper nouns (capitalized words as potential names)
            name_matches = re.findall(r'\b([A-Z][a-z]+)\b', text)
            for name in name_matches:
                if name not in ['The', 'A', 'An', 'If', 'Then']:  # Filter common words
                    entities.append(Entity(
                        name=name,
                        entity_type='person',
                        value=name,
                        turn_id=turn_id,
                        step_id=step_idx
                    ))
        
        return entities
    
    def _extract_claims_from_crg(self, crg_data: Dict, turn_id: int) -> List[Claim]:
        """
        Extract factual claims from CRG nodes.
        
        Simple pattern matching for now.
        """
        claims = []
        
        steps = crg_data.get('reasoning_steps', [])
        
        for step_idx, step in enumerate(steps):
            text = step.get('text', '')
            
            # Pattern: "X is Y" or "X equals Y"
            is_pattern = re.findall(r'(\w+)\s+(?:is|equals)\s+(\w+)', text, re.IGNORECASE)
            for subject, obj in is_pattern:
                claims.append(Claim(
                    subject=subject,
                    predicate='is',
                    object=obj,
                    turn_id=turn_id,
                    step_id=step_idx
                ))
        
        return claims


# Example usage
if __name__ == "__main__":
    # Demo
    memory = ConversationMemory(max_turns=5)
    
    # Turn 1
    turn1 = {
        'reasoning_steps': [
            {'text': 'Let x = 5'},
            {'text': 'Then 2x = 10'},
        ]
    }
    memory.add_turn(turn1, turn_id=0)
    
    # Turn 2 (consistent)
    turn2 = {
        'reasoning_steps': [
            {'text': 'Since x = 5, we know x^2 = 25'},
        ]
    }
    contradictions = memory.check_consistency(turn2, current_turn=1)
    print(f"Turn 2 contradictions: {contradictions}")  # Should be empty
    
    memory.add_turn(turn2, turn_id=1)
    
    # Turn 3 (contradiction!)
    turn3 = {
        'reasoning_steps': [
            {'text': 'Now x = 3, so x + 2 = 5'},
        ]
    }
    contradictions = memory.check_consistency(turn3, current_turn=2)
    print(f"Turn 3 contradictions: {contradictions}")  # Should detect x value changed
    
    print("\n✓ ConversationMemory demo complete!")
