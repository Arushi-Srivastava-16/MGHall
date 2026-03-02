"""
Fingerprint Extractor for Model Identification.

Extracts linguistic, structural, and stylistic features from reasoning chains
to create unique fingerprints for different LLM models.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import Counter
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import ReasoningChain


class FingerprintExtractor:
    """Extract model-specific fingerprint features from reasoning chains."""
    
    # Common hedge words indicating uncertainty
    HEDGE_WORDS = [
        "maybe", "perhaps", "possibly", "likely", "probably",
        "might", "could", "may", "seem", "appear", "suggest",
        "approximately", "roughly", "about", "around"
    ]
    
    # Confidence indicators
    CONFIDENCE_HIGH = ["definitely", "certainly", "clearly", "obviously", "undoubtedly"]
    CONFIDENCE_LOW = ["uncertain", "unsure", "unclear", "ambiguous", "questionable"]
    
    def __init__(self):
        """Initialize fingerprint extractor."""
        pass
    
    def extract(self, chain: ReasoningChain) -> Dict[str, Any]:
        """
        Extract comprehensive fingerprint from reasoning chain.
        
        Args:
            chain: Reasoning chain to extract features from
            
        Returns:
            Dictionary of fingerprint features
        """
        text = self._concatenate_steps(chain)
        
        fingerprint = {
            **self._extract_linguistic_features(text, chain),
            **self._extract_structural_features(chain),
            **self._extract_stylistic_features(text, chain),
            **self._extract_confidence_features(text),
            **self._extract_formatting_features(chain),
        }
        
        return fingerprint
    
    def _concatenate_steps(self, chain: ReasoningChain) -> str:
        """Concatenate all reasoning step texts."""
        return " ".join([step.text for step in chain.reasoning_steps])
    
    def _extract_linguistic_features(
        self,
        text: str,
        chain: ReasoningChain
    ) -> Dict[str, float]:
        """
        Extract linguistic features.
        
        Features:
        - Vocabulary richness (unique words / total words)
        - Average word length
        - Sentence complexity (avg words per sentence)
        - Technical term density
        - Punctuation density
        """
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Vocabulary richness
        unique_words = len(set(words))
        total_words = len(words)
        vocab_richness = unique_words / total_words if total_words > 0 else 0
        
        # Average word length
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Sentence complexity
        avg_words_per_sentence = total_words / len(sentences) if sentences else 0
        
        # Technical term density (words > 8 characters)
        long_words = [w for w in words if len(w) > 8]
        technical_density = len(long_words) / total_words if total_words > 0 else 0
        
        # Punctuation density
        punctuation = len(re.findall(r'[,;:!?]', text))
        punct_density = punctuation / len(text) if text else 0
        
        return {
            "vocab_richness": vocab_richness,
            "avg_word_length": avg_word_length,
            "avg_words_per_sentence": avg_words_per_sentence,
            "technical_density": technical_density,
            "punctuation_density": punct_density,
            "total_words": total_words,
            "total_sentences": len(sentences),
        }
    
    def _extract_structural_features(self, chain: ReasoningChain) -> Dict[str, float]:
        """
        Extract structural features.
        
        Features:
        - Number of reasoning steps
        - Average step length
        - Dependency complexity (avg dependencies per step)
        - Graph depth (longest path in dependency graph)
        - Branching factor
        """
        steps = chain.reasoning_steps
        num_steps = len(steps)
        
        # Average step length
        step_lengths = [len(step.text.split()) for step in steps]
        avg_step_length = np.mean(step_lengths) if step_lengths else 0
        std_step_length = np.std(step_lengths) if len(step_lengths) > 1 else 0
        
        # Dependency complexity
        dependencies_per_step = [len(step.depends_on) for step in steps]
        avg_dependencies = np.mean(dependencies_per_step) if dependencies_per_step else 0
        
        # Graph depth (simplified: just max step_id assuming sequential)
        graph_depth = num_steps
        
        # Branching factor (steps with multiple dependencies)
        branching_steps = sum(1 for num_deps in dependencies_per_step if num_deps > 1)
        branching_factor = branching_steps / num_steps if num_steps > 0 else 0
        
        return {
            "num_steps": num_steps,
            "avg_step_length": avg_step_length,
            "std_step_length": std_step_length,
            "avg_dependencies": avg_dependencies,
            "graph_depth": graph_depth,
            "branching_factor": branching_factor,
        }
    
    def _extract_stylistic_features(
        self,
        text: str,
        chain: ReasoningChain
    ) -> Dict[str, float]:
        """
        Extract stylistic features.
        
        Features:
        - Verbosity (words per step)
        - Explanation style (use of "because", "therefore", etc.)
        - Question usage
        - First-person usage ("I", "we")
        - Passive voice density
        """
        steps = chain.reasoning_steps
        words = text.lower().split()
        
        # Verbosity
        verbosity = len(words) / len(steps) if steps else 0
        
        # Explanation markers
        explanation_markers = ["because", "therefore", "thus", "hence", "so", "since"]
        explanation_count = sum(text.lower().count(marker) for marker in explanation_markers)
        explanation_density = explanation_count / len(text) if text else 0
        
        # Question usage
        question_marks = text.count('?')
        question_density = question_marks / len(steps) if steps else 0
        
        # First-person usage
        first_person = ["i ", "we ", "i'", "we'", "my ", "our "]
        first_person_count = sum(text.lower().count(fp) for fp in first_person)
        first_person_density = first_person_count / len(words) if words else 0
        
        # Passive voice indicators (simple heuristic)
        passive_indicators = ["is ", "are ", "was ", "were ", "been ", "being "]
        passive_count = sum(text.lower().count(pi) for pi in passive_indicators)
        passive_density = passive_count / len(words) if words else 0
        
        return {
            "verbosity": verbosity,
            "explanation_density": explanation_density,
            "question_density": question_density,
            "first_person_density": first_person_density,
            "passive_density": passive_density,
        }
    
    def _extract_confidence_features(self, text: str) -> Dict[str, float]:
        """
        Extract confidence-related features.
        
        Features:
        - Hedge word density
        - High confidence word density
        - Low confidence word density
        - Certainty score
        """
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        # Hedge words
        hedge_count = sum(text_lower.count(hw) for hw in self.HEDGE_WORDS)
        hedge_density = hedge_count / total_words if total_words > 0 else 0
        
        # High confidence
        high_conf_count = sum(text_lower.count(hw) for hw in self.CONFIDENCE_HIGH)
        high_conf_density = high_conf_count / total_words if total_words > 0 else 0
        
        # Low confidence
        low_conf_count = sum(text_lower.count(lw) for lw in self.CONFIDENCE_LOW)
        low_conf_density = low_conf_count / total_words if total_words > 0 else 0
        
        # Certainty score (high confidence - hedge words - low confidence)
        certainty_score = high_conf_density - hedge_density - low_conf_density
        
        return {
            "hedge_density": hedge_density,
            "high_confidence_density": high_conf_density,
            "low_confidence_density": low_conf_density,
            "certainty_score": certainty_score,
        }
    
    def _extract_formatting_features(self, chain: ReasoningChain) -> Dict[str, float]:
        """
        Extract formatting-related features.
        
        Features:
        - Use of numbered steps
        - Use of bullet points
        - Use of code blocks
        - Use of mathematical notation
        """
        steps = chain.reasoning_steps
        
        # Numbered steps (steps starting with numbers)
        numbered_steps = sum(1 for step in steps if re.match(r'^\d+\.?\s', step.text.strip()))
        numbered_ratio = numbered_steps / len(steps) if steps else 0
        
        # Bullet points
        bullet_steps = sum(1 for step in steps if re.match(r'^[-*•]\s', step.text.strip()))
        bullet_ratio = bullet_steps / len(steps) if steps else 0
        
        # Code blocks (looking for backticks or indented code)
        code_indicators = sum(step.text.count('```') + step.text.count('`') for step in steps)
        code_density = code_indicators / len(steps) if steps else 0
        
        # Mathematical notation
        math_indicators = sum(
            step.text.count('$') + 
            step.text.count('=') +
            step.text.count('+') +
            step.text.count('*')
            for step in steps
        )
        math_density = math_indicators / len(steps) if steps else 0
        
        return {
            "numbered_ratio": numbered_ratio,
            "bullet_ratio": bullet_ratio,
            "code_density": code_density,
            "math_density": math_density,
        }
    
    def extract_batch(self, chains: List[ReasoningChain]) -> List[Dict[str, Any]]:
        """
        Extract fingerprints from multiple chains.
        
        Args:
            chains: List of reasoning chains
            
        Returns:
            List of fingerprint dictionaries
        """
        return [self.extract(chain) for chain in chains]
    
    def get_feature_vector(self, fingerprint: Dict[str, Any]) -> np.ndarray:
        """
        Convert fingerprint dictionary to feature vector.
        
        Args:
            fingerprint: Fingerprint dictionary
            
        Returns:
            Numpy array of features
        """
        # Extract numerical features in consistent order
        feature_names = [
            "vocab_richness", "avg_word_length", "avg_words_per_sentence",
            "technical_density", "punctuation_density",
            "num_steps", "avg_step_length", "std_step_length",
            "avg_dependencies", "graph_depth", "branching_factor",
            "verbosity", "explanation_density", "question_density",
            "first_person_density", "passive_density",
            "hedge_density", "high_confidence_density", 
            "low_confidence_density", "certainty_score",
            "numbered_ratio", "bullet_ratio", "code_density", "math_density",
        ]
        
        vector = np.array([fingerprint.get(name, 0.0) for name in feature_names])
        return vector
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return [
            "vocab_richness", "avg_word_length", "avg_words_per_sentence",
            "technical_density", "punctuation_density",
            "num_steps", "avg_step_length", "std_step_length",
            "avg_dependencies", "graph_depth", "branching_factor",
            "verbosity", "explanation_density", "question_density",
            "first_person_density", "passive_density",
            "hedge_density", "high_confidence_density", 
            "low_confidence_density", "certainty_score",
            "numbered_ratio", "bullet_ratio", "code_density", "math_density",
        ]


if __name__ == "__main__":
    # Test fingerprint extraction
    print("=" * 80)
    print("Fingerprint Extractor Test")
    print("=" * 80)
    
    from src.data_processing.unified_schema import ReasoningStep, DependencyGraph, Domain
    
    # Create sample chain
    steps = [
        ReasoningStep(0, "First, we need to identify the problem variables.", True, False, None, []),
        ReasoningStep(1, "Let x be the unknown value we're solving for.", True, False, None, [0]),
        ReasoningStep(2, "We can set up the equation: 2x + 5 = 13", True, False, None, [1]),
        ReasoningStep(3, "Subtracting 5 from both sides: 2x = 8", True, False, None, [2]),
        ReasoningStep(4, "Therefore, dividing by 2, we get x = 4", True, False, None, [3]),
    ]
    
    chain = ReasoningChain(
        domain=Domain.MATH,
        query_id="test_001",
        query="Solve: 2x + 5 = 13",
        ground_truth="x = 4",
        reasoning_steps=steps,
        dependency_graph=DependencyGraph(
            nodes=[0, 1, 2, 3, 4],
            edges=[[0, 1], [1, 2], [2, 3], [3, 4]]
        )
    )
    
    # Extract fingerprint
    extractor = FingerprintExtractor()
    fingerprint = extractor.extract(chain)
    
    print("\nExtracted Fingerprint:")
    print(f"  Linguistic Features:")
    print(f"    - Vocab Richness: {fingerprint['vocab_richness']:.3f}")
    print(f"    - Avg Word Length: {fingerprint['avg_word_length']:.2f}")
    print(f"    - Technical Density: {fingerprint['technical_density']:.3f}")
    
    print(f"\n  Structural Features:")
    print(f"    - Number of Steps: {fingerprint['num_steps']}")
    print(f"    - Avg Step Length: {fingerprint['avg_step_length']:.2f}")
    print(f"    - Graph Depth: {fingerprint['graph_depth']}")
    
    print(f"\n  Stylistic Features:")
    print(f"    - Verbosity: {fingerprint['verbosity']:.2f}")
    print(f"    - Explanation Density: {fingerprint['explanation_density']:.4f}")
    
    print(f"\n  Confidence Features:")
    print(f"    - Certainty Score: {fingerprint['certainty_score']:.4f}")
    
    # Get feature vector
    vector = extractor.get_feature_vector(fingerprint)
    print(f"\nFeature Vector Shape: {vector.shape}")
    print(f"Feature Vector: {vector[:5]}... (showing first 5)")
    
    print("\nFingerprint extraction test passed!")

