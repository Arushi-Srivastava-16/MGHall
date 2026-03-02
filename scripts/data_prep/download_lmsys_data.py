
import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_processing.unified_schema import ReasoningChain, ReasoningStep, Domain, DependencyGraph

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_to_chain(conv_obj, model_name):
    """
    Convert LMSYS conversation object to ReasoningChain schema.
    LMSYS format: "conversation": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """
    try:
        messages = conv_obj['conversation']
        if len(messages) < 2: return None
        
        # Find first user-assistant pair
        query = ""
        response = ""
        
        for msg in messages:
            if msg['role'] == 'user':
                query = msg['content']
            elif msg['role'] == 'assistant' and query:
                response = msg['content']
                break
        
        if not response: return None

        # Synthesize Steps from Response (e.g. split by newline or sentences)
        # Real reasoning is often structured.
        step_texts = [s.strip() for s in re.split(r'\n', response) if s.strip()]
        if not step_texts: step_texts = [response]
        
        steps = []
        for i, text in enumerate(step_texts):
            # First few steps = reasoning, last = answer? We don't know, so just all neutral
            steps.append(ReasoningStep(
                step_id=i,
                text=text,
                is_correct=True,
                is_origin=False,
                error_type=None,
                depends_on=[i-1] if i > 0 else []
            ))
            
        # Dummy graph
        nodes = [i for i in range(len(steps))]
        edges = [[i, i+1] for i in range(len(steps)-1)]

        return ReasoningChain(
            domain=Domain.MATH, # Placeholder
            query_id=conv_obj['conversation_id'],
            query=query[:200], # Truncate for sanity
            ground_truth="",
            reasoning_steps=steps,
            dependency_graph=DependencyGraph(nodes=nodes, edges=edges)
        )
    except Exception as e:
        return None

def main():
    print("🚀 Downloading Real-World LLM Data (Argilla UltraFeedback)...")
    
    # Argilla dataset has 'model_chosen' and 'model_rejected' columns
    # We will scan both to find our targets.
    
    # Map from Dataset Model Name -> Our internal ID
    # The dataset uses full names like "gpt-4-0613", "claude-2", "meta-llama/Llama-2-70b-chat-hf"
    target_mapping = {
        'gpt-4': 'gpt-4',
        'wizardlm-70b': 'claude-v1', # Proxy for demo (dataset lacks Claude)
        'llama-2-70b-chat': 'llama-2-70b-chat'
    }
    
    counts = {v: 0 for v in target_mapping.values()}
    max_count = 500
    
    output_dir = Path(__file__).parent.parent.parent / "data/multi_model/generated_chains"
    ensure_dir(output_dir)
    
    files = {v: open(output_dir / f"{v}_math_chains.jsonl", 'w') for v in counts.keys()}

    try:
        # Use argilla/ultrafeedback-binarized-preferences-cleaned (Public, Non-Gated)
        dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train", streaming=True)
        
        for row in tqdm(dataset):
            # Check Chosen
            model_chosen = row['chosen-model']
            chosen_response = row['chosen'] # list of dicts [{'content': ...}]
            
            # Check Rejected
            model_rejected = row['rejected-model']
            rejected_response = row['rejected']
            
            # Helper to process one side
            def process_side(model_name, response_msgs):
                for key_part, internal_id in target_mapping.items():
                    if key_part in model_name and counts[internal_id] < max_count:
                         # Found a match!
                         # Convert message list to chain input format
                         # response_msgs is [{'role': 'user', ...}, {'role': 'assistant', ...}]
                         # We need to reshape this to pass to convert_to_chain logic
                         # actually convert_to_chain expects a dict with 'conversation' key
                         
                         wrapper = {'conversation': response_msgs, 'conversation_id': f"real_{counts[internal_id]}"}
                         chain = convert_to_chain(wrapper, internal_id)
                         if chain:
                             files[internal_id].write(chain.to_json() + '\n')
                             counts[internal_id] += 1
                             return True
                return False

            process_side(model_chosen, chosen_response)
            process_side(model_rejected, rejected_response)
            
            if all(c >= max_count for c in counts.values()):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        for f in files.values():
            f.close()
            
    print("✅ Download Complete!")
    for k, v in counts.items():
        print(f"  {k}: {v} samples")

if __name__ == "__main__":
    main()
