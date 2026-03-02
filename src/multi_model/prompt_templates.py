"""
Prompt Templates for Multi-Model Inference.

Provides unified prompt templates for reasoning tasks across different domains,
with proper formatting for chain-of-thought reasoning and step extraction.
"""

from typing import Dict, List, Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import Domain


class PromptStyle(Enum):
    """Different prompt styles for different models."""
    OPENAI = "openai"  # For GPT-4
    GOOGLE = "google"  # For Gemini
    INSTRUCT = "instruct"  # For Llama/Mistral instruct models


class PromptTemplate:
    """Template for generating prompts for reasoning tasks."""
    
    # System prompts for different domains
    SYSTEM_PROMPTS = {
        Domain.MATH: """You are a helpful mathematical reasoning assistant. 
Solve the given problem step-by-step, showing your work clearly. 
Each step should be numbered and should follow logically from the previous ones.""",
        
        Domain.CODE: """You are an expert programmer and code analyst.
Solve the given coding problem by breaking it down into clear, logical steps.
Explain your reasoning for each step.""",
        
        Domain.MEDICAL: """You are a medical reasoning assistant with expertise in clinical decision-making.
Answer the given medical question by reasoning through it step-by-step.
Base your reasoning on established medical knowledge.""",
    }
    
    # Task prompts
    TASK_PROMPTS = {
        Domain.MATH: """Problem: {query}

Please solve this problem step-by-step. Break down your solution into clear, numbered steps.

Format your response exactly as follows:
Step 1: [Your first reasoning step]
Step 2: [Your second reasoning step]
...
Final Answer: [Your final answer]""",
        
        Domain.CODE: """Problem: {query}

Please solve this coding problem by reasoning through it step-by-step.

Format your response exactly as follows:
Step 1: [First step of your reasoning]
Step 2: [Second step of your reasoning]
...
Step N: [Final step and solution]""",
        
        Domain.MEDICAL: """Question: {query}

Please answer this medical question by reasoning through it step-by-step.

Format your response exactly as follows:
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
Final Answer: [Your conclusion]""",
    }
    
    def __init__(self, domain: Domain, style: PromptStyle = PromptStyle.OPENAI):
        """
        Initialize prompt template.
        
        Args:
            domain: Reasoning domain
            style: Prompt style for specific model type
        """
        self.domain = domain
        self.style = style
    
    def format_prompt(self, query: str, include_system: bool = True) -> Dict[str, str]:
        """
        Format a prompt for the given query.
        
        Args:
            query: Query/problem to solve
            include_system: Whether to include system prompt
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        system_prompt = self.SYSTEM_PROMPTS.get(self.domain, "")
        task_prompt = self.TASK_PROMPTS.get(self.domain, "").format(query=query)
        
        if self.style == PromptStyle.OPENAI:
            # OpenAI format with separate system and user messages
            return {
                "system": system_prompt if include_system else "",
                "user": task_prompt,
            }
        
        elif self.style == PromptStyle.GOOGLE:
            # Gemini format (combines system and user)
            combined = f"{system_prompt}\n\n{task_prompt}" if include_system else task_prompt
            return {
                "system": "",
                "user": combined,
            }
        
        elif self.style == PromptStyle.INSTRUCT:
            # Instruct format for Llama/Mistral
            combined = f"""<|system|>
{system_prompt}
<|user|>
{task_prompt}
<|assistant|>
"""
            return {
                "system": "",
                "user": combined,
            }
        
        return {"system": system_prompt, "user": task_prompt}
    
    def parse_response(self, response: str) -> List[str]:
        """
        Parse model response into individual reasoning steps.
        
        Args:
            response: Raw model response
            
        Returns:
            List of reasoning step texts
        """
        steps = []
        lines = response.split('\n')
        
        current_step = ""
        for line in lines:
            line = line.strip()
            
            # Check if line starts a new step
            if line.startswith("Step ") or line.startswith("Final Answer:"):
                if current_step:
                    steps.append(current_step.strip())
                
                # Extract step content (remove "Step N:" prefix)
                if ':' in line:
                    current_step = line.split(':', 1)[1].strip()
                else:
                    current_step = line
            elif current_step:
                # Continue current step
                current_step += " " + line
        
        # Add last step
        if current_step:
            steps.append(current_step.strip())
        
        # If no steps found, try alternative parsing
        if not steps:
            steps = self._alternative_parse(response)
        
        return steps
    
    def _alternative_parse(self, response: str) -> List[str]:
        """Alternative parsing if structured format not found."""
        # Split by double newlines or numbered lists
        parts = response.split('\n\n')
        steps = []
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 10:  # Filter out very short fragments
                steps.append(part)
        
        # If still no steps, return entire response as single step
        if not steps:
            steps = [response.strip()]
        
        return steps


def get_prompt_for_domain(domain: Domain, model_provider: str = "openai") -> PromptTemplate:
    """
    Get appropriate prompt template for domain and model.
    
    Args:
        domain: Reasoning domain
        model_provider: Model provider ("openai", "google", "local")
        
    Returns:
        PromptTemplate instance
    """
    style_map = {
        "openai": PromptStyle.OPENAI,
        "google": PromptStyle.GOOGLE,
        "local": PromptStyle.INSTRUCT,
    }
    
    style = style_map.get(model_provider, PromptStyle.OPENAI)
    return PromptTemplate(domain=domain, style=style)


def create_few_shot_prompt(
    domain: Domain,
    query: str,
    examples: List[Dict[str, str]],
    style: PromptStyle = PromptStyle.OPENAI
) -> Dict[str, str]:
    """
    Create a few-shot prompt with examples.
    
    Args:
        domain: Reasoning domain
        query: Query to solve
        examples: List of example dicts with 'query' and 'solution' keys
        style: Prompt style
        
    Returns:
        Formatted prompt dictionary
    """
    template = PromptTemplate(domain=domain, style=style)
    
    # Build examples string
    examples_str = "\n\n".join([
        f"Example {i+1}:\nProblem: {ex['query']}\nSolution:\n{ex['solution']}"
        for i, ex in enumerate(examples)
    ])
    
    # Add examples before the actual query
    base_prompt = template.format_prompt(query)
    base_prompt['user'] = f"{examples_str}\n\nNow solve this problem:\n\n{base_prompt['user']}"
    
    return base_prompt


if __name__ == "__main__":
    # Test prompt templates
    print("=" * 80)
    print("Prompt Template Test")
    print("=" * 80)
    
    # Test math prompt
    print("\n1. Math Domain (OpenAI style):")
    math_template = PromptTemplate(Domain.MATH, PromptStyle.OPENAI)
    prompt = math_template.format_prompt("Solve: 2x + 5 = 13")
    print(f"System: {prompt['system'][:100]}...")
    print(f"User: {prompt['user'][:100]}...")
    
    # Test response parsing
    print("\n2. Response Parsing:")
    sample_response = """Step 1: Subtract 5 from both sides: 2x = 8
Step 2: Divide both sides by 2: x = 4
Final Answer: x = 4"""
    
    steps = math_template.parse_response(sample_response)
    print(f"Parsed {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"  Step {i}: {step[:60]}...")
    
    # Test different domains
    print("\n3. Different Domains:")
    for domain in [Domain.MATH, Domain.CODE, Domain.MEDICAL]:
        template = get_prompt_for_domain(domain, "openai")
        print(f"  - {domain.value}: {template.SYSTEM_PROMPTS[domain][:60]}...")
    
    print("\nPrompt template test passed!")

