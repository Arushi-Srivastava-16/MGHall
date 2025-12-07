"""
Unified data schema for multi-domain reasoning chains.

This module defines the JSON schema and validation for the unified format
that all datasets will be converted to.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json


class Domain(str, Enum):
    """Supported domains."""
    MATH = "math"
    CODE = "code"
    MEDICAL = "medical"


class ErrorType(str, Enum):
    """Types of errors in reasoning steps."""
    FACTUAL = "factual"
    LOGICAL = "logical"
    SYNTAX = "syntax"
    UNKNOWN = "unknown"


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_id: int
    text: str
    is_correct: bool
    is_origin: bool = False  # True if this is the first error location
    error_type: Optional[ErrorType] = None
    depends_on: List[int] = None  # List of step_ids this step depends on
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.error_type and isinstance(self.error_type, str):
            self.error_type = ErrorType(self.error_type)


@dataclass
class DependencyGraph:
    """Graph structure representing dependencies between steps."""
    nodes: List[int]  # List of step_ids
    edges: List[List[int]]  # List of [from_id, to_id] pairs


@dataclass
class ReasoningChain:
    """Complete reasoning chain in unified format."""
    domain: Domain
    query_id: str
    query: str
    ground_truth: str
    reasoning_steps: List[ReasoningStep]
    dependency_graph: DependencyGraph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "domain": self.domain.value if isinstance(self.domain, Domain) else self.domain,
            "query_id": self.query_id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "reasoning_steps": [
                {
                    "step_id": step.step_id,
                    "text": step.text,
                    "is_correct": step.is_correct,
                    "is_origin": step.is_origin,
                    "error_type": step.error_type.value if step.error_type else None,
                    "depends_on": step.depends_on,
                }
                for step in self.reasoning_steps
            ],
            "dependency_graph": {
                "nodes": self.dependency_graph.nodes,
                "edges": self.dependency_graph.edges,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningChain":
        """Create ReasoningChain from dictionary."""
        steps = [
            ReasoningStep(
                step_id=step["step_id"],
                text=step["text"],
                is_correct=step["is_correct"],
                is_origin=step.get("is_origin", False),
                error_type=ErrorType(step["error_type"]) if step.get("error_type") else None,
                depends_on=step.get("depends_on", []),
            )
            for step in data["reasoning_steps"]
        ]
        
        return cls(
            domain=Domain(data["domain"]),
            query_id=data["query_id"],
            query=data["query"],
            ground_truth=data["ground_truth"],
            reasoning_steps=steps,
            dependency_graph=DependencyGraph(
                nodes=data["dependency_graph"]["nodes"],
                edges=data["dependency_graph"]["edges"],
            ),
        )
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Serialize to JSON string.
        
        Args:
            pretty: If True, pretty-print with indentation. If False, single line (JSONL format).
        """
        if pretty:
            return json.dumps(self.to_dict(), indent=2)
        else:
            return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningChain":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def validate_reasoning_chain(chain: ReasoningChain) -> List[str]:
    """
    Validate a reasoning chain and return list of errors (empty if valid).
    
    Args:
        chain: ReasoningChain to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check that step_ids are unique and sequential
    step_ids = [step.step_id for step in chain.reasoning_steps]
    if len(step_ids) != len(set(step_ids)):
        errors.append("Duplicate step_ids found")
    
    if step_ids != sorted(step_ids):
        errors.append("step_ids must be sorted")
    
    # Check that only one step is marked as origin
    origin_steps = [step for step in chain.reasoning_steps if step.is_origin]
    if len(origin_steps) > 1:
        errors.append(f"Multiple origin steps found: {[s.step_id for s in origin_steps]}")
    
    # Check that origin step is incorrect
    for step in origin_steps:
        if step.is_correct:
            errors.append(f"Origin step {step.step_id} is marked as correct")
    
    # Check dependency graph consistency
    all_node_ids = set(chain.dependency_graph.nodes)
    step_id_set = set(step_ids)
    
    if all_node_ids != step_id_set:
        errors.append("Dependency graph nodes don't match step_ids")
    
    # Check that edges reference valid nodes
    for edge in chain.dependency_graph.edges:
        if len(edge) != 2:
            errors.append(f"Invalid edge format: {edge}")
        elif edge[0] not in all_node_ids or edge[1] not in all_node_ids:
            errors.append(f"Edge references invalid node: {edge}")
    
    # Check that depends_on references valid step_ids
    for step in chain.reasoning_steps:
        for dep_id in step.depends_on:
            if dep_id not in step_id_set:
                errors.append(f"Step {step.step_id} depends on invalid step_id {dep_id}")
    
    return errors

