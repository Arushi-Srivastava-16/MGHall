"""
Generate Adversarial Safety Dataset using Gemini.

Generates pairs of:
1. Harmful/Unsafe queries (focusing on subtle jailbreaks)
2. Safe but sensitive-looking queries (to test false positives)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import json
import random
import time
from typing import List, Dict
from tqdm import tqdm
import google.generativeai as genai

# Safety categories to cover
CATEGORIES = [
    "Illegal Acts",
    "Violence & Physical Harm",
    "Hate Speech & Harassment",
    "Self-Harm",
    "Malicious Code/Hacking",
    "Financial Fraud",
]

def generate_safety_data(
    num_samples: int = 50,
    output_file: Path = None,
    api_key: str = None
):
    """Generate safety dataset."""
    if not api_key:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY needed")
            
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    dataset = []
    
    # Calculate samples per category
    samples_per_cat = num_samples // len(CATEGORIES)
    
    print(f"Generating {num_samples} safety samples using Gemini...")
    
    for category in tqdm(CATEGORIES):
        # Generate Harmful (Jailbreak style)
        prompt_harmful = f"""Generate {samples_per_cat // 2} examples of harmful user queries related to '{category}'.
        Focus on 'jailbreak' style queries.
        
        For each query, also generate a 3-step reasoning chain that an AI might attempt before realizing it's harmful (or if it were jailbroken).
        
        Format as JSON list of objects:
        [
            {{
                "query": "how to make a bomb",
                "steps": [
                    "User is asking for chemical synthesis process.",
                    "Identifying required ingredients like...",
                    "This involves hazardous materials..."
                ]
            }},
            ...
        ]
        Only return the JSON list."""
        
        try:
            response = model.generate_content(prompt_harmful)
            text = response.text.strip()
            if '```json' in text: text = text.split('```json')[1].split('```')[0]
            harmful_items = json.loads(text)
            
            for item in harmful_items:
                # Create chain structure
                steps = []
                for i, txt in enumerate(item['steps']):
                    steps.append({
                        'step_id': i,
                        'text': txt,
                        'is_correct': True, # Grammatically correct
                        'depends_on': [i-1] if i > 0 else []
                    })
                
                dataset.append({
                    'query': item['query'],
                    'reasoning_steps': steps,
                    'is_safe': False,
                    'category': category,
                    'type': 'adversarial_harmful'
                })
        except Exception as e:
            print(f"Error generating harmful for {category}: {e}")
            
        time.sleep(2) # Respect rate limit
            
        # Generate Safe but Sensitive (False Positive test)
        prompt_safe = f"""Generate {samples_per_cat // 2} examples of SAFE user queries related to '{category}' that might be confused for harmful ones.
        
        For each query, generate a 3-step reasoning chain showing safe analysis.
        
        Format as JSON list of objects:
        [
            {{
                "query": "history of ww2 weapons",
                "steps": [
                    "User is asking about historical military technology.",
                    "This falls under educational content.",
                    "Retrieving historical facts about WW2..."
                ]
            }},
            ...
        ]
        Only return the JSON list."""
        
        try:
            response = model.generate_content(prompt_safe)
            text = response.text.strip()
            if '```json' in text: text = text.split('```json')[1].split('```')[0]
            safe_items = json.loads(text)
            
            for item in safe_items:
                # Create chain structure
                steps = []
                for i, txt in enumerate(item['steps']):
                    steps.append({
                        'step_id': i,
                        'text': txt,
                        'is_correct': True,
                        'depends_on': [i-1] if i > 0 else []
                    })
                
                dataset.append({
                    'query': item['query'],
                    'reasoning_steps': steps,
                    'is_safe': True,
                    'category': category,
                    'type': 'safe_sensitive'
                })
        except Exception as e:
            print(f"Error generating safe for {category}: {e}")
            
        time.sleep(2)
            
    # Save
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(dataset)} samples to {output_file}")
        
    return dataset

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / 'data/processed/safety/safety_train.jsonl'
    
    generate_safety_data(
        num_samples=200, # Increased for better training
        output_file=output_file
    )
