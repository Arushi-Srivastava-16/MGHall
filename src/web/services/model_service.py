"""
Model Service.

Handles model loading, inference, and comparison.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.multi_model.model_config import ModelType, get_available_models, get_model_config
from src.multi_model.llm_inference import MultiModelInference
from src.data_processing.unified_schema import Domain


class ModelService:
    """Service for model operations."""
    
    def __init__(self):
        """Initialize model service."""
        self.available_models = get_available_models()
        self.multi_model: Optional[MultiModelInference] = None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models (only those that can actually be used)."""
        models = []
        for model_type, config in self.available_models.items():
            # Only include models that are actually available
            # For API models, check if API key exists
            # For local models, they're available if transformers is installed
            try:
                if config.provider in ["openai", "google"]:
                    # Check if API key is available
                    config.get_api_key()
                    available = True
                else:
                    # Local models are available if transformers is installed
                    try:
                        import torch
                        from transformers import AutoTokenizer
                        available = True
                    except ImportError:
                        available = False
                
                if available:
                    models.append({
                        "model_id": model_type,
                        "model_name": config.model_name,
                        "provider": config.provider,
                        "available": True,
                    })
            except (ValueError, Exception):
                # API key not available or other error, skip this model
                continue
        return models
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        if model_id not in self.available_models:
            return None
        
        config = self.available_models[model_id]
        return {
            "model_id": model_id,
            "model_name": config.model_name,
            "provider": config.provider,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "available": True,
        }
    
    def initialize_multi_model(self, model_types: List[str]) -> bool:
        """Initialize multi-model inference."""
        try:
            model_type_objs = [ModelType(mt) for mt in model_types]
            self.multi_model = MultiModelInference(model_type_objs)
            return True
        except Exception as e:
            print(f"Error initializing models: {e}")
            return False
    
    def compare_models(
        self,
        model_ids: List[str],
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare models (placeholder - would load actual comparison data)."""
        return {
            "models": model_ids,
            "domain": domain,
            "comparison": "Model comparison data would be loaded here",
        }

