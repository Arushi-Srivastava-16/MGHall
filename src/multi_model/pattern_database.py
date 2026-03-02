"""
Hallucination Pattern Database.

Stores and analyzes model-specific hallucination patterns across different LLMs.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import json
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import ReasoningChain, Domain, ErrorType
from src.multi_model.model_config import ModelType


class HallucinationType(Enum):
    """Types of hallucinations."""
    FACTUAL = "factual"  # Incorrect facts/numbers
    LOGICAL = "logical"  # Flawed reasoning/logic
    SYNTAX = "syntax"  # Code/math syntax errors
    CONSISTENCY = "consistency"  # Self-contradictions
    GROUNDING = "grounding"  # Unsupported claims
    UNKNOWN = "unknown"


@dataclass
class HallucinationPattern:
    """A specific hallucination pattern."""
    pattern_id: str
    model_type: ModelType
    domain: Domain
    hallucination_type: HallucinationType
    description: str
    frequency: int = 1
    examples: List[str] = None
    severity: float = 0.5  # 0-1 scale
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_id": self.pattern_id,
            "model_type": self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            "domain": self.domain.value if isinstance(self.domain, Domain) else self.domain,
            "hallucination_type": self.hallucination_type.value if isinstance(self.hallucination_type, HallucinationType) else self.hallucination_type,
            "description": self.description,
            "frequency": self.frequency,
            "examples": self.examples[:5],  # Store max 5 examples
            "severity": self.severity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HallucinationPattern":
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            model_type=ModelType(data["model_type"]),
            domain=Domain(data["domain"]),
            hallucination_type=HallucinationType(data["hallucination_type"]),
            description=data["description"],
            frequency=data.get("frequency", 1),
            examples=data.get("examples", []),
            severity=data.get("severity", 0.5),
        )


class PatternDatabase:
    """Database for storing and analyzing hallucination patterns."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize pattern database.
        
        Args:
            db_path: Optional path to save/load database
        """
        self.db_path = db_path
        self.patterns: Dict[str, HallucinationPattern] = {}
        self.model_stats: Dict[ModelType, Dict[str, Any]] = defaultdict(lambda: {
            "total_patterns": 0,
            "by_type": defaultdict(int),
            "by_domain": defaultdict(int),
            "avg_severity": 0.0,
        })
        
        if db_path and db_path.exists():
            self.load(db_path)
    
    def add_pattern(self, pattern: HallucinationPattern):
        """
        Add a pattern to the database.
        
        Args:
            pattern: Hallucination pattern to add
        """
        if pattern.pattern_id in self.patterns:
            # Pattern exists, increment frequency
            existing = self.patterns[pattern.pattern_id]
            existing.frequency += 1
            existing.examples.extend(pattern.examples)
            # Keep only unique examples
            existing.examples = list(set(existing.examples))[:5]
        else:
            # New pattern
            self.patterns[pattern.pattern_id] = pattern
        
        # Update model statistics
        self._update_stats(pattern.model_type)
    
    def _update_stats(self, model_type: ModelType):
        """Update statistics for a model."""
        model_patterns = [p for p in self.patterns.values() if p.model_type == model_type]
        
        self.model_stats[model_type]["total_patterns"] = len(model_patterns)
        
        # By type
        by_type = defaultdict(int)
        for p in model_patterns:
            by_type[p.hallucination_type.value] += p.frequency
        self.model_stats[model_type]["by_type"] = dict(by_type)
        
        # By domain
        by_domain = defaultdict(int)
        for p in model_patterns:
            by_domain[p.domain.value] += p.frequency
        self.model_stats[model_type]["by_domain"] = dict(by_domain)
        
        # Average severity
        if model_patterns:
            avg_severity = np.mean([p.severity * p.frequency for p in model_patterns]) / sum(p.frequency for p in model_patterns)
            self.model_stats[model_type]["avg_severity"] = float(avg_severity)
    
    def get_patterns_by_model(self, model_type: ModelType) -> List[HallucinationPattern]:
        """Get all patterns for a specific model."""
        return [p for p in self.patterns.values() if p.model_type == model_type]
    
    def get_patterns_by_domain(self, domain: Domain) -> List[HallucinationPattern]:
        """Get all patterns for a specific domain."""
        return [p for p in self.patterns.values() if p.domain == domain]
    
    def get_patterns_by_type(
        self,
        hallucination_type: HallucinationType
    ) -> List[HallucinationPattern]:
        """Get all patterns of a specific type."""
        return [p for p in self.patterns.values() if p.hallucination_type == hallucination_type]
    
    def get_model_vulnerability_profile(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get vulnerability profile for a model.
        
        Args:
            model_type: Model to get profile for
            
        Returns:
            Dictionary with vulnerability statistics
        """
        patterns = self.get_patterns_by_model(model_type)
        
        if not patterns:
            return {
                "model_type": model_type.value,
                "total_patterns": 0,
                "vulnerability_score": 0.0,
                "most_common_type": None,
                "most_vulnerable_domain": None,
            }
        
        # Total frequency of hallucinations
        total_freq = sum(p.frequency for p in patterns)
        
        # Most common hallucination type
        type_counts = defaultdict(int)
        for p in patterns:
            type_counts[p.hallucination_type.value] += p.frequency
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Most vulnerable domain
        domain_counts = defaultdict(int)
        for p in patterns:
            domain_counts[p.domain.value] += p.frequency
        most_vulnerable_domain = max(domain_counts.items(), key=lambda x: x[1])[0]
        
        # Vulnerability score (weighted by severity)
        vulnerability_score = sum(p.frequency * p.severity for p in patterns) / total_freq
        
        return {
            "model_type": model_type.value,
            "total_patterns": len(patterns),
            "total_frequency": total_freq,
            "vulnerability_score": vulnerability_score,
            "most_common_type": most_common_type,
            "type_distribution": dict(type_counts),
            "most_vulnerable_domain": most_vulnerable_domain,
            "domain_distribution": dict(domain_counts),
            "avg_severity": self.model_stats[model_type]["avg_severity"],
        }
    
    def compare_models(
        self,
        model_types: List[ModelType]
    ) -> Dict[str, Any]:
        """
        Compare hallucination patterns across models.
        
        Args:
            model_types: List of models to compare
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "models": [mt.value for mt in model_types],
            "profiles": {},
            "relative_vulnerability": {},
        }
        
        # Get profiles for each model
        for model_type in model_types:
            comparison["profiles"][model_type.value] = self.get_model_vulnerability_profile(model_type)
        
        # Compute relative vulnerability
        scores = {
            mt.value: comparison["profiles"][mt.value]["vulnerability_score"]
            for mt in model_types
        }
        
        if scores:
            max_score = max(scores.values()) if scores.values() else 1.0
            comparison["relative_vulnerability"] = {
                model: score / max_score if max_score > 0 else 0
                for model, score in scores.items()
            }
        
        return comparison
    
    def find_similar_patterns(
        self,
        pattern: HallucinationPattern,
        threshold: float = 0.7
    ) -> List[HallucinationPattern]:
        """
        Find patterns similar to the given pattern.
        
        Args:
            pattern: Pattern to find similar to
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar patterns
        """
        similar = []
        
        for existing_pattern in self.patterns.values():
            if existing_pattern.pattern_id == pattern.pattern_id:
                continue
            
            # Compute similarity score
            similarity = 0.0
            
            # Same hallucination type: +0.4
            if existing_pattern.hallucination_type == pattern.hallucination_type:
                similarity += 0.4
            
            # Same domain: +0.3
            if existing_pattern.domain == pattern.domain:
                similarity += 0.3
            
            # Similar severity: +0.3
            severity_diff = abs(existing_pattern.severity - pattern.severity)
            similarity += 0.3 * (1 - severity_diff)
            
            if similarity >= threshold:
                similar.append(existing_pattern)
        
        return similar
    
    def extract_patterns_from_chain(
        self,
        chain: ReasoningChain,
        model_type: ModelType,
    ) -> List[HallucinationPattern]:
        """
        Extract hallucination patterns from a reasoning chain.
        
        Args:
            chain: Reasoning chain to analyze
            model_type: Model that generated the chain
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        for i, step in enumerate(chain.reasoning_steps):
            if not step.is_correct:
                # Map ErrorType to HallucinationType
                hall_type = self._map_error_to_hallucination_type(step.error_type)
                
                # Create pattern
                pattern_id = f"{model_type.value}_{chain.domain.value}_{hall_type.value}_{i}"
                pattern = HallucinationPattern(
                    pattern_id=pattern_id,
                    model_type=model_type,
                    domain=chain.domain,
                    hallucination_type=hall_type,
                    description=f"Error in step {i}: {step.text[:100]}",
                    frequency=1,
                    examples=[step.text[:200]],
                    severity=0.7 if step.is_origin else 0.5,  # Origin errors more severe
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _map_error_to_hallucination_type(
        self,
        error_type: Optional[ErrorType]
    ) -> HallucinationType:
        """Map ErrorType to HallucinationType."""
        if error_type is None:
            return HallucinationType.UNKNOWN
        
        mapping = {
            ErrorType.FACTUAL: HallucinationType.FACTUAL,
            ErrorType.LOGICAL: HallucinationType.LOGICAL,
            ErrorType.SYNTAX: HallucinationType.SYNTAX,
            ErrorType.UNKNOWN: HallucinationType.UNKNOWN,
        }
        
        return mapping.get(error_type, HallucinationType.UNKNOWN)
    
    def save(self, path: Optional[Path] = None):
        """
        Save database to JSON file.
        
        Args:
            path: Optional path override
        """
        save_path = path or self.db_path
        if not save_path:
            raise ValueError("No save path specified")
        
        data = {
            "patterns": [p.to_dict() for p in self.patterns.values()],
            "model_stats": {
                k.value: v for k, v in self.model_stats.items()
            }
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.patterns)} patterns to {save_path}")
    
    def load(self, path: Optional[Path] = None):
        """
        Load database from JSON file.
        
        Args:
            path: Optional path override
        """
        load_path = path or self.db_path
        if not load_path or not load_path.exists():
            print(f"No database found at {load_path}")
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # Load patterns
        self.patterns = {
            p["pattern_id"]: HallucinationPattern.from_dict(p)
            for p in data["patterns"]
        }
        
        # Rebuild stats
        for model_type in set(p.model_type for p in self.patterns.values()):
            self._update_stats(model_type)
        
        print(f"Loaded {len(self.patterns)} patterns from {load_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get database summary statistics."""
        return {
            "total_patterns": len(self.patterns),
            "total_models": len(self.model_stats),
            "models": list(self.model_stats.keys()),
            "hallucination_types": list(set(p.hallucination_type.value for p in self.patterns.values())),
            "domains": list(set(p.domain.value for p in self.patterns.values())),
        }


if __name__ == "__main__":
    # Test pattern database
    print("=" * 80)
    print("Pattern Database Test")
    print("=" * 80)
    
    # Create database
    db = PatternDatabase()
    
    # Add sample patterns
    pattern1 = HallucinationPattern(
        pattern_id="gpt4_math_factual_001",
        model_type=ModelType.GPT4,
        domain=Domain.MATH,
        hallucination_type=HallucinationType.FACTUAL,
        description="Incorrect calculation in arithmetic step",
        frequency=1,
        examples=["2 + 2 = 5"],
        severity=0.8,
    )
    
    pattern2 = HallucinationPattern(
        pattern_id="gemini_code_syntax_001",
        model_type=ModelType.GEMINI,
        domain=Domain.CODE,
        hallucination_type=HallucinationType.SYNTAX,
        description="Missing semicolon in code",
        frequency=1,
        examples=["print('hello')"],
        severity=0.6,
    )
    
    db.add_pattern(pattern1)
    db.add_pattern(pattern2)
    
    # Add duplicate pattern (should increase frequency)
    db.add_pattern(pattern1)
    
    print(f"\nTotal patterns: {len(db.patterns)}")
    print(f"Pattern 1 frequency: {db.patterns['gpt4_math_factual_001'].frequency}")
    
    # Get vulnerability profile
    print("\nGPT-4 Vulnerability Profile:")
    profile = db.get_model_vulnerability_profile(ModelType.GPT4)
    print(f"  Total patterns: {profile['total_patterns']}")
    print(f"  Vulnerability score: {profile['vulnerability_score']:.3f}")
    print(f"  Most common type: {profile['most_common_type']}")
    
    # Save and load
    test_path = Path("test_pattern_db.json")
    db.save(test_path)
    
    db2 = PatternDatabase(test_path)
    print(f"\nLoaded database: {len(db2.patterns)} patterns")
    
    # Cleanup
    test_path.unlink()
    
    print("\nPattern database test passed!")

