"""
Multi-Model Fingerprinting Module.

This module provides infrastructure for multi-model hallucination analysis,
including model inference, fingerprint extraction, pattern analysis, and
consensus detection across GPT-4, Gemini, Llama, Mistral, and other LLMs.
"""

from .model_config import ModelConfig, ModelType, get_model_config
from .llm_inference import LLMInference, MultiModelInference
from .prompt_templates import PromptTemplate, get_prompt_for_domain
from .fingerprint_extractor import FingerprintExtractor
from .pattern_database import PatternDatabase, HallucinationPattern
from .fingerprint_classifier import FingerprintClassifier
from .consensus_detector import ConsensusDetector
from .cross_model_analyzer import CrossModelAnalyzer

__all__ = [
    "ModelConfig",
    "ModelType",
    "get_model_config",
    "LLMInference",
    "MultiModelInference",
    "PromptTemplate",
    "get_prompt_for_domain",
    "FingerprintExtractor",
    "PatternDatabase",
    "HallucinationPattern",
    "FingerprintClassifier",
    "ConsensusDetector",
    "CrossModelAnalyzer",
]

