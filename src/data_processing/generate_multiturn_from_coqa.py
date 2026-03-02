"""
Generate multi-turn dataset from CoQA + injected hallucinations.

Hybrid approach:
1. Load CoQA for conversation structure
2. Generate reasoning chains for each Q&A
3. Inject cross-turn hallucinations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import random
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm


def generate_reasoning_for_qa(
    question: str,
    answer: str,
    story: str,
    use_gemini: bool = True
) -> List[Dict]:
    """
    Generate a reasoning chain for Q&A.
    
    Uses Gemini API if available, otherwise falls back to templates.
    """
    if use_gemini:
        try:
            reasoning_chain = _generate_with_gemini(question, answer, story)
            return reasoning_chain
        except Exception as e:
            print(f"Warning: Gemini generation failed ({e}), using template")
            # Fall through to template
    
    # Template fallback
    reasoning_chain = [
        {
            'step_id': 0,
            'text': f'I need to find the answer to: {question}',
            'is_correct': True,
            'is_origin': False,
            'error_type': None,
            'depends_on': [],
        },
        {
            'step_id': 1,
            'text': f'Looking through the story for relevant details...',
            'is_correct': True,
            'is_origin': False,
            'error_type': None,
            'depends_on': [0],
        },
        {
            'step_id': 2,
            'text': f'Based on the information, the answer is: {answer}',
            'is_correct': True,
            'is_origin': False,
            'error_type': None,
            'depends_on': [0, 1],
        },
    ]
    
    return reasoning_chain


def _generate_with_gemini(question: str, answer: str, story: str) -> List[Dict]:
    """
    Use Gemini to generate realistic reasoning chain.
    
    Requires GOOGLE_API_KEY environment variable.
    """
    import os
    import google.generativeai as genai
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Prompt for step-by-step reasoning
    prompt = f"""Given this story and question, generate a step-by-step reasoning chain that leads to the answer.

Story: {story[:500]}  # Truncate long stories

Question: {question}
Correct Answer: {answer}

Generate 3-5 reasoning steps showing how to arrive at this answer. Each step should be a single sentence.

Format as JSON array:
[
  {{"step": 1, "text": "First, I identify..."}},
  {{"step": 2, "text": "Then, I consider..."}},
  ...
]

Only return the JSON array, nothing else."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Extract JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        steps_raw = json.loads(text)
        
        # Convert to our format
        reasoning_chain = []
        for i, step_data in enumerate(steps_raw):
            reasoning_chain.append({
                'step_id': i,
                'text': step_data.get('text', step_data.get('step', '')),
                'is_correct': True,
                'is_origin': False,
                'error_type': None,
                'depends_on': list(range(i)),  # Depends on all previous
            })
        
        return reasoning_chain
        
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}")


def extract_entities_simple(text: str) -> Dict[str, Any]:
    """
    Extract entities from text using simple patterns.
    
    Returns dict of entity_name -> value.
    """
    import re
    
    entities = {}
    
    # Extract capitalized names (proper nouns)
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    for name in names:
        if name not in ['The', 'A', 'An', 'And', 'But', 'Or']:
            entities[name] = name
    
    # Extract numbers
    numbers = re.findall(r'\b(\d+)\b', text)
    for num in numbers:
        entities[f'number_{num}'] = int(num)
    
    # Extract locations (capitalized place names)
    # Simple heuristic: words ending in common suffixes
    locations = re.findall(r'\b([A-Z][a-z]+(?:ville|ton|berg|burg|polis|chester))\b', text)
    for loc in locations:
        entities[loc] = loc
    
    return entities


def inject_cross_turn_error(
    reasoning_chain: List[Dict],
    entity_tracker: Dict[str, Any],
    error_type: str = 'entity'
) -> List[Dict]:
    """
    Inject a cross-turn hallucination into the reasoning chain.
    
    Args:
        reasoning_chain: Original reasoning chain
        entity_tracker: Entities from previous turns
        error_type: 'entity', 'claim', or 'buildup'
        
    Returns:
        Modified reasoning chain with injected error
    """
    if not entity_tracker:
        # No previous entities to contradict
        return reasoning_chain
    
    # Pick a random step to modify (prefer later steps)
    step_idx = random.randint(1, len(reasoning_chain) - 1)
    
    if error_type == 'entity':
        # Entity contradiction: change entity value
        # Pick a random entity from history
        entity_names = list(entity_tracker.keys())
        if not entity_names:
            return reasoning_chain
            
        entity_name = random.choice(entity_names)
        old_value = entity_tracker[entity_name]
        
        # Generate new conflicting value
        if isinstance(old_value, str):
            # For names, use a different name
            new_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Emma', 'Frank']
            new_value = random.choice([n for n in new_names if n != old_value])
        elif isinstance(old_value, int):
            # For numbers, change by ±10%
            new_value = old_value + random.choice([-3, -2, -1, 1, 2, 3])
        else:
            new_value = 'MODIFIED'
        
        # Inject error into text
        original_text = reasoning_chain[step_idx]['text']
        modified_text = original_text.replace(str(old_value), str(new_value))
        
        reasoning_chain[step_idx]['text'] = modified_text
        reasoning_chain[step_idx]['is_correct'] = False
        reasoning_chain[step_idx]['error_type'] = 'factual'
        reasoning_chain[step_idx]['is_origin'] = True
    
    elif error_type == 'claim':
        # Claim reversal: negate a statement
        negations = [
            (' is ', ' is not '),
            (' was ', ' was not '),
            (' can ', ' cannot '),
            (' will ', ' will not '),
        ]
        
        original_text = reasoning_chain[step_idx]['text']
        for pos, neg in negations:
            if pos in original_text.lower():
                modified_text = original_text.replace(pos, neg)
                reasoning_chain[step_idx]['text'] = modified_text
                reasoning_chain[step_idx]['is_correct'] = False
                reasoning_chain[step_idx]['error_type'] = 'logical'
                reasoning_chain[step_idx]['is_origin'] = True
                break
    
    elif error_type == 'buildup':
        # Progressive buildup: reference wrong entity from earlier
        if entity_tracker:
            entity_names = list(entity_tracker.keys())
            wrong_entity = random.choice(entity_names)
            
            original_text = reasoning_chain[step_idx]['text']
            # Add confusing reference
            modified_text = f"{original_text} (Note: {wrong_entity} was mentioned)"
            
            reasoning_chain[step_idx]['text'] = modified_text
            # This might not be incorrect, just confusing
    
    return reasoning_chain


def generate_multiturn_from_coqa(
    num_conversations: int = 100,
    turns_per_conversation: int = 5,
    error_injection_rate: float = 0.4,
    use_gemini: bool = True,
    seed: int = 42
) -> List[List[Dict]]:
    """
    Generate multi-turn dataset from CoQA.
    
    Args:
        num_conversations: Number of conversations to generate
        turns_per_conversation: Max turns per conversation
        error_injection_rate: Probability of injecting error per turn
        seed: Random seed
        
    Returns:
        List of conversations, each is a list of turn dicts
    """
    random.seed(seed)
    
    print("Loading CoQA dataset...")
    try:
        coqa_dataset = load_dataset("stanfordnlp/coqa", split="train")
    except Exception as e:
        print(f"Warning: Could not load CoQA ({e}). Using mock data for testing.")
        # Create mock data for testing
        coqa_dataset = _create_mock_coqa_data(num_conversations)
    
    print(f"Generating {num_conversations} multi-turn conversations...")
    
    conversations = []
    
    for conv_idx in tqdm(range(min(num_conversations, len(coqa_dataset)))):
        coqa_conv = coqa_dataset[conv_idx]
        
        # Get story and Q&A
        story = coqa_conv.get('story', '')
        questions = coqa_conv.get('questions', [])
        answers = coqa_conv.get('answers', {}).get('input_text', [])
        
        if not questions or not answers:
            continue
        
        # Limit turns
        max_turns = min(turns_per_conversation, len(questions), len(answers))
        
        turns = []
        entity_tracker = {}
        
        for turn_idx in range(max_turns):
            question = questions[turn_idx]
            answer = answers[turn_idx] if turn_idx < len(answers) else "Unknown"
            
            # Generate reasoning chain (with Gemini if available)
            reasoning_chain = generate_reasoning_for_qa(
                question, answer, story, use_gemini=use_gemini
            )
            
            # Inject error with some probability (not on first turn)
            if turn_idx > 0 and random.random() < error_injection_rate:
                error_type = random.choice(['entity', 'claim', 'buildup'])
                reasoning_chain = inject_cross_turn_error(
                    reasoning_chain,
                    entity_tracker,
                    error_type=error_type
                )
            
            # Update entity tracker
            for step in reasoning_chain:
                entities = extract_entities_simple(step['text'])
                entity_tracker.update(entities)
            
            # Create turn dict
            turn = {
                'turn_id': turn_idx,
                'question': question,
                'answer': answer,
                'reasoning_steps': reasoning_chain,
                'domain': 'qa',
                'query_id': f'coqa_{conv_idx}_turn_{turn_idx}',
            }
            
            turns.append(turn)
        
        conversations.append(turns)
    
    print(f"✓ Generated {len(conversations)} conversations")
    print(f"  Avg turns per conversation: {sum(len(c) for c in conversations) / len(conversations):.1f}")
    
    return conversations


def _create_mock_coqa_data(num_samples: int = 10) -> List[Dict]:
    """Create mock CoQA-like data for testing."""
    mock_data = []
    
    for i in range(num_samples):
        mock_data.append({
            'story': f'Sarah lives in Boston. She is {20 + i} years old and loves reading.',
            'questions': [
                'What is the main character\'s name?',
                'Where does she live?',
                'What does she like to do?',
            ],
            'answers': {
                'input_text': ['Sarah', 'Boston', 'reading']
            }
        })
    
    return mock_data


def save_multiturn_dataset(
    conversations: List[List[Dict]],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """Save multi-turn dataset to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train/val/test
    num_conversations = len(conversations)
    num_train = int(num_conversations * train_ratio)
    num_val = int(num_conversations * val_ratio)
    
    train_conversations = conversations[:num_train]
    val_conversations = conversations[num_train:num_train + num_val]
    test_conversations = conversations[num_train + num_val:]
    
    # Save each split
    for split_name, split_data in [
        ('train', train_conversations),
        ('val', val_conversations),
        ('test', test_conversations)
    ]:
        output_file = output_dir / f'{split_name}.jsonl'
        
        with open(output_file, 'w') as f:
            for conversation in split_data:
                # Write entire conversation as one line (for multi-turn grouping)
                f.write(json.dumps({'turns': conversation}) + '\n')
        
        print(f"✓ Saved {len(split_data)} conversations to {output_file}")


def main():
    """Generate and save multi-turn dataset."""
    import os
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'data/processed/multiturn_coqa'
    
    # Check if Gemini API key available
    has_gemini = bool(os.getenv('GOOGLE_API_KEY'))
    if has_gemini:
        print("✓ Using Gemini API for reasoning chain generation")
    else:
        print("⚠️ No GOOGLE_API_KEY found, using template-based generation")
    
    # Generate conversations
    conversations = generate_multiturn_from_coqa(
        num_conversations=50,  # Quick test with 50
        turns_per_conversation=5,
        error_injection_rate=0.4,
        use_gemini=has_gemini,
    )
    
    # Save to disk
    save_multiturn_dataset(conversations, output_dir)
    
    print("\n✓ Multi-turn dataset generation complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
