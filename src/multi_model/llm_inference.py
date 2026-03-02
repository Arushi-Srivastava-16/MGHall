"""
Unified LLM Inference Pipeline.

Handles inference for multiple models: GPT-4, Gemini, Llama, Mistral.
Provides consistent interface across different providers.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import Domain, ReasoningChain, ReasoningStep, DependencyGraph
from src.multi_model.model_config import ModelType, ModelConfig, get_model_config
from src.multi_model.prompt_templates import PromptTemplate, get_prompt_for_domain

# Import model-specific libraries
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


@dataclass
class InferenceResult:
    """Result from model inference."""
    model_type: ModelType
    query: str
    response: str
    reasoning_steps: List[str]
    latency: float  # seconds
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class LLMInference:
    """Inference wrapper for a single LLM model."""
    
    def __init__(self, model_type: ModelType):
        """
        Initialize LLM inference.
        
        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.config = get_model_config(model_type)
        self.client = None
        self.model = None
        self.tokenizer = None
        
        # Initialize based on provider
        if self.config.provider == "openai":
            self._init_openai()
        elif self.config.provider == "google":
            self._init_gemini()
        elif self.config.provider == "local":
            self._init_local()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        api_key = self.config.get_api_key()
        self.client = OpenAI(api_key=api_key)
    
    def _init_gemini(self):
        """Initialize Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI library not available. Install with: pip install google-generativeai")
        
        api_key = self.config.get_api_key()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config.model_name)
    
    def _init_local(self):
        """Initialize local model (Llama/Mistral)."""
        if not LOCAL_MODELS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers torch accelerate")
        
        # Detect actual device (override config if CUDA not available)
        use_cuda = torch.cuda.is_available() and self.config.device == "cuda"
        actual_device = "cuda" if use_cuda else "cpu"
        
        print(f"Loading local model: {self.config.model_name}...")
        print(f"Using device: {actual_device}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            raise ImportError(f"Failed to load tokenizer for {self.config.model_name}. "
                            f"Error: {e}. "
                            f"For gated models like Llama, you may need to authenticate with HuggingFace: "
                            f"`huggingface-cli login`")
        
        # Load model with optional quantization
        load_kwargs = {
            "dtype": torch.float16 if use_cuda else torch.float32,  # Use dtype instead of torch_dtype
        }
        
        # Only use device_map with accelerate on CUDA
        try:
            import accelerate
            if use_cuda:
                load_kwargs["device_map"] = "auto"
        except ImportError:
            # accelerate not available, use manual device placement
            if use_cuda:
                load_kwargs["device_map"] = None
            else:
                load_kwargs["device_map"] = None
        
        # 8-bit quantization only on CUDA
        if self.config.load_in_8bit and use_cuda:
            try:
                import bitsandbytes
                load_kwargs["load_in_8bit"] = True
            except ImportError:
                print("Warning: bitsandbytes not available, skipping 8-bit quantization")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in load_kwargs or load_kwargs["device_map"] is None:
                self.model = self.model.to(actual_device)
            
            print(f"Model loaded successfully on {actual_device}")
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "401" in error_msg:
                raise ImportError(
                    f"Model {self.config.model_name} is gated and requires HuggingFace authentication. "
                    f"Run: `huggingface-cli login` and accept the model's terms at "
                    f"https://huggingface.co/{self.config.model_name}"
                )
            else:
                raise ImportError(f"Failed to load model {self.config.model_name}: {e}")
    
    def generate(
        self,
        prompt: Dict[str, str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate response from model.
        
        Args:
            prompt: Dictionary with 'system' and 'user' prompts
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Generated response text
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        if self.config.provider == "openai":
            return self._generate_openai(prompt, temp, max_tok)
        elif self.config.provider == "google":
            return self._generate_gemini(prompt, temp, max_tok)
        elif self.config.provider == "local":
            return self._generate_local(prompt, temp, max_tok)
        
        raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _generate_openai(self, prompt: Dict[str, str], temp: float, max_tok: int) -> str:
        """Generate using OpenAI API."""
        messages = []
        
        if prompt.get("system"):
            messages.append({"role": "system", "content": prompt["system"]})
        
        messages.append({"role": "user", "content": prompt["user"]})
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        
        return response.choices[0].message.content
    
    def _generate_gemini(self, prompt: Dict[str, str], temp: float, max_tok: int) -> str:
        """Generate using Gemini API."""
        # Gemini combines system and user prompts
        full_prompt = prompt["user"]
        if prompt.get("system"):
            full_prompt = f"{prompt['system']}\n\n{prompt['user']}"
        
        generation_config = {
            "temperature": temp,
            "max_output_tokens": max_tok,
            "top_p": self.config.top_p,
        }
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    def _generate_local(self, prompt: Dict[str, str], temp: float, max_tok: int) -> str:
        """Generate using local model."""
        # Format prompt for instruct models
        full_prompt = prompt["user"]
        if prompt.get("system"):
            full_prompt = f"<|system|>\n{prompt['system']}\n<|user|>\n{prompt['user']}\n<|assistant|>\n"
        
        # Detect actual device
        use_cuda = torch.cuda.is_available() and self.config.device == "cuda"
        device = "cuda" if use_cuda else "cpu"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        if use_cuda:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if full_prompt in response:
            response = response.replace(full_prompt, "").strip()
        
        return response
    
    def infer(
        self,
        query: str,
        domain: Domain,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> InferenceResult:
        """
        Run inference on a query.
        
        Args:
            query: Query to solve
            domain: Reasoning domain
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            InferenceResult with response and parsed steps
        """
        start_time = time.time()
        
        try:
            # Get prompt template
            prompt_template = get_prompt_for_domain(domain, self.config.provider)
            prompt = prompt_template.format_prompt(query)
            
            # Generate response
            response = self.generate(prompt, temperature, max_tokens)
            
            # Parse steps
            steps = prompt_template.parse_response(response)
            
            latency = time.time() - start_time
            
            return InferenceResult(
                model_type=self.model_type,
                query=query,
                response=response,
                reasoning_steps=steps,
                latency=latency,
                tokens_used=None,  # TODO: Track tokens
                error=None,
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return InferenceResult(
                model_type=self.model_type,
                query=query,
                response="",
                reasoning_steps=[],
                latency=latency,
                tokens_used=None,
                error=str(e),
            )


class MultiModelInference:
    """
    Manage inference across multiple models.
    
    Provides batch inference and result comparison.
    """
    
    def __init__(self, model_types: List[ModelType]):
        """
        Initialize multi-model inference.
        
        Args:
            model_types: List of model types to use
        """
        self.model_types = model_types
        self.models: Dict[ModelType, LLMInference] = {}
        
        # Initialize models
        for model_type in model_types:
            try:
                self.models[model_type] = LLMInference(model_type)
                print(f"✓ Initialized {model_type.value}")
            except Exception as e:
                print(f"✗ Failed to initialize {model_type.value}: {e}")
    
    def infer_all(
        self,
        query: str,
        domain: Domain,
        temperature: Optional[float] = None,
    ) -> Dict[ModelType, InferenceResult]:
        """
        Run inference on all models.
        
        Args:
            query: Query to solve
            domain: Reasoning domain
            temperature: Optional temperature
            
        Returns:
            Dictionary mapping model types to inference results
        """
        results = {}
        
        for model_type, model in self.models.items():
            print(f"Running inference on {model_type.value}...")
            result = model.infer(query, domain, temperature)
            results[model_type] = result
            
            if result.error:
                print(f"  Error: {result.error}")
            else:
                print(f"  Generated {len(result.reasoning_steps)} steps in {result.latency:.2f}s")
        
        return results
    
    def infer_batch(
        self,
        queries: List[str],
        domain: Domain,
        temperature: Optional[float] = None,
    ) -> List[Dict[ModelType, InferenceResult]]:
        """
        Run inference on multiple queries across all models.
        
        Args:
            queries: List of queries to solve
            domain: Reasoning domain
            temperature: Optional temperature
            
        Returns:
            List of result dictionaries (one per query)
        """
        all_results = []
        
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}/{len(queries)}")
            results = self.infer_all(query, domain, temperature)
            all_results.append(results)
        
        return all_results
    
    def convert_to_reasoning_chain(
        self,
        result: InferenceResult,
        query_id: str,
        domain: Domain,
        ground_truth: str = "",
    ) -> ReasoningChain:
        """
        Convert inference result to ReasoningChain format.
        
        Args:
            result: Inference result
            query_id: Unique query identifier
            domain: Reasoning domain
            ground_truth: Ground truth answer (if known)
            
        Returns:
            ReasoningChain object
        """
        # Create reasoning steps
        steps = []
        for i, step_text in enumerate(result.reasoning_steps):
            step = ReasoningStep(
                step_id=i,
                text=step_text,
                is_correct=True,  # Unknown at this point
                is_origin=False,
                error_type=None,
                depends_on=[i-1] if i > 0 else [],
            )
            steps.append(step)
        
        # Build dependency graph (sequential for now)
        nodes = list(range(len(steps)))
        edges = [[i-1, i] for i in range(1, len(steps))]
        dep_graph = DependencyGraph(nodes=nodes, edges=edges)
        
        # Create chain
        chain = ReasoningChain(
            domain=domain,
            query_id=f"{query_id}_{result.model_type.value}",
            query=result.query,
            ground_truth=ground_truth,
            reasoning_steps=steps,
            dependency_graph=dep_graph,
        )
        
        return chain


if __name__ == "__main__":
    # Test inference
    print("=" * 80)
    print("LLM Inference Test")
    print("=" * 80)
    
    # Test with a simple math problem
    test_query = "Solve: 2x + 5 = 13. What is x?"
    
    # Try to initialize available models
    print("\nInitializing models...")
    try:
        from src.multi_model.model_config import get_default_models
        model_types = get_default_models()[:2]  # Test with first 2 available models
        
        if not model_types:
            print("No models available. Please set API keys or install local models.")
        else:
            multi_model = MultiModelInference(model_types)
            
            print(f"\nTesting with query: '{test_query}'")
            results = multi_model.infer_all(test_query, Domain.MATH, temperature=0.7)
            
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)
            
            for model_type, result in results.items():
                print(f"\n{model_type.value}:")
                if result.error:
                    print(f"  Error: {result.error}")
                else:
                    print(f"  Steps: {len(result.reasoning_steps)}")
                    print(f"  Latency: {result.latency:.2f}s")
                    print(f"  First step: {result.reasoning_steps[0][:80]}...")
            
            print("\nInference test passed!")
    
    except Exception as e:
        print(f"Test skipped: {e}")
        print("This is normal if API keys are not set or local models not installed.")

