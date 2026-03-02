"""
Model Configuration Management.

Handles configuration for all supported LLM models including API keys,
model parameters, and unified interface.
"""

import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env from project root (2 levels up from this file)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass


class ModelType(str, Enum):
    """Supported LLM model types."""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GEMINI = "gemini-pro"
    GEMINI_15 = "gemini-1.5-pro"
    LLAMA = "llama-3-8b"
    LLAMA_70B = "llama-3-70b"
    TINYLLAMA = "tinyllama"  # Open-source alternative, no auth required
    MISTRAL = "mistral-7b"
    MISTRAL_MEDIUM = "mistral-medium"
    
    # Legacy models found in training datasets
    CLAUDE_V1 = "claude-v1"
    LLAMA_2_70B = "llama-2-70b-chat"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    
    model_type: ModelType
    model_name: str  # Full model identifier for API
    provider: str  # "openai", "google", "local"
    api_key_env: Optional[str] = None  # Environment variable name for API key
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Model-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    # Local model parameters (for Llama, Mistral)
    model_path: Optional[str] = None
    device: str = "cuda"
    load_in_8bit: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.provider in ["openai", "google"] and not self.api_key_env:
            raise ValueError(f"API key environment variable required for {self.provider}")
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment: {self.api_key_env}")
            return api_key
        return None


# Predefined configurations for all supported models
MODEL_CONFIGS: Dict[ModelType, ModelConfig] = {
    # OpenAI GPT-4
    ModelType.GPT4: ModelConfig(
        model_type=ModelType.GPT4,
        model_name="gpt-4",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        temperature=0.7,
        max_tokens=2048,
    ),
    
    ModelType.GPT4_TURBO: ModelConfig(
        model_type=ModelType.GPT4_TURBO,
        model_name="gpt-4-turbo-preview",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        temperature=0.7,
        max_tokens=4096,
    ),
    
    # Google Gemini
    ModelType.GEMINI: ModelConfig(
        model_type=ModelType.GEMINI,
        model_name="gemini-2.5-flash",  # Updated to current API model name
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        temperature=0.7,
        max_tokens=2048,
    ),
    
    ModelType.GEMINI_15: ModelConfig(
        model_type=ModelType.GEMINI_15,
        model_name="gemini-2.5-pro",  # Updated to current API model name
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        temperature=0.7,
        max_tokens=4096,
    ),
    
    # Local Llama models
    ModelType.LLAMA: ModelConfig(
        model_type=ModelType.LLAMA,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        provider="local",
        temperature=0.7,
        max_tokens=2048,
        model_path=None,  # Will use HuggingFace cache
        device="cuda",
        load_in_8bit=True,  # For memory efficiency
    ),
    
    ModelType.LLAMA_70B: ModelConfig(
        model_type=ModelType.LLAMA_70B,
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        provider="local",
        temperature=0.7,
        max_tokens=2048,
        model_path=None,
        device="cuda",
        load_in_8bit=True,
    ),
    
    # Mistral models
    ModelType.MISTRAL: ModelConfig(
        model_type=ModelType.MISTRAL,
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="local",
        temperature=0.7,
        max_tokens=2048,
        model_path=None,
        device="cuda",
        load_in_8bit=True,
    ),
    
    ModelType.MISTRAL_MEDIUM: ModelConfig(
        model_type=ModelType.MISTRAL_MEDIUM,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        provider="local",
        temperature=0.7,
        max_tokens=2048,
        model_path=None,
        device="cuda",
        load_in_8bit=True,
    ),
    
    # Dataset Legacy Models
    ModelType.CLAUDE_V1: ModelConfig(
        model_type=ModelType.CLAUDE_V1,
        model_name="claude-v1",
        provider="anthropic", # Placeholder
        temperature=0.7,
        max_tokens=2048,
    ),
    
    ModelType.LLAMA_2_70B: ModelConfig(
        model_type=ModelType.LLAMA_2_70B,
        model_name="meta-llama/Llama-2-70b-chat-hf",
        provider="local",
        temperature=0.7,
        max_tokens=2048,
    ),
}


def get_model_config(model_type: ModelType) -> ModelConfig:
    """
    Get configuration for a specific model type.
    
    Args:
        model_type: Type of model to get configuration for
        
    Returns:
        ModelConfig instance
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return MODEL_CONFIGS[model_type]


def get_available_models() -> Dict[str, ModelConfig]:
    """
    Get all available models with valid API keys or local availability.
    
    Returns:
        Dictionary of available model configs
    """
    available = {}
    
    for model_type, config in MODEL_CONFIGS.items():
        try:
            if config.provider in ["openai", "google"]:
                # Check if API key is available
                config.get_api_key()
                available[model_type.value] = config
            else:
                # Local models are always available (assuming hardware support)
                available[model_type.value] = config
        except ValueError:
            # API key not available, skip this model
            pass
    
    return available


def get_default_models() -> list[ModelType]:
    """
    Get default set of models for Phase 5 (4-5 models).
    
    Tries to use: GPT-4, Gemini, Llama-8B, Mistral-7B
    Falls back to available alternatives (including TinyLlama if Llama unavailable).
    
    Returns:
        List of default model types
    """
    preferred = [
        ModelType.GPT4,
        ModelType.GEMINI,
        ModelType.LLAMA,
        ModelType.MISTRAL,
    ]
    
    # Add TinyLlama as fallback if Llama fails
    fallback_models = [ModelType.TINYLLAMA]
    
    available = get_available_models()
    defaults = []
    
    for model_type in preferred:
        if model_type.value in available:
            defaults.append(model_type)
    
    # If we don't have at least 3 models, add alternatives
    if len(defaults) < 3:
        # First try fallback models
        for model_type in fallback_models:
            if model_type not in defaults and model_type.value in available:
                defaults.append(model_type)
                if len(defaults) >= 4:
                    break
        
        # Then try any other available models
        if len(defaults) < 4:
            for model_type in MODEL_CONFIGS.keys():
                if model_type not in defaults and model_type.value in available:
                    defaults.append(model_type)
                    if len(defaults) >= 4:
                        break
    
    return defaults


class ModelInterface:
    """
    Unified interface for all model types.
    
    Provides consistent API across different providers and local models.
    """
    
    def __init__(self, model_type: ModelType):
        """
        Initialize model interface.
        
        Args:
            model_type: Type of model to interface with
        """
        self.config = get_model_config(model_type)
        self.model_type = model_type
        
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response from model.
        
        Args:
            prompt: Input prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
        """
        # This is a placeholder - actual implementation will be in llm_inference.py
        raise NotImplementedError("Use LLMInference class for actual inference")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.model_type.value,
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }


if __name__ == "__main__":
    # Test configuration
    print("=" * 80)
    print("Multi-Model Configuration Test")
    print("=" * 80)
    
    print("\nAvailable Models:")
    available = get_available_models()
    for model_name, config in available.items():
        print(f"  - {model_name}: {config.model_name} ({config.provider})")
    
    print("\nDefault Models for Phase 5:")
    defaults = get_default_models()
    for model_type in defaults:
        config = get_model_config(model_type)
        print(f"  - {model_type.value}: {config.model_name}")
    
    print("\nConfiguration test passed!")

