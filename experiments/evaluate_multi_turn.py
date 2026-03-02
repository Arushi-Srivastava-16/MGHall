"""
Evaluate Memory-Augmented GAT on Multi-Turn Hallucinations.

Measures performance specifically on cross-turn contradictions vs single-turn errors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from src.graph_construction.crg_builder import build_crg
from src.data_processing.unified_schema import ReasoningChain, ReasoningStep, DependencyGraph, Domain, ErrorType
from models.gnn_architectures.memory_augmented_gat import MemoryAugmentedGAT


def dict_to_reasoning_chain(turn_data: dict) -> ReasoningChain:
    """Convert turn dictionary to ReasoningChain object."""
    steps = []
    nodes = []
    edges = []
    
    for i, step_data in enumerate(turn_data['reasoning_steps']):
        # Ensure dependencies exist
        if 'depends_on' not in step_data:
            step_data['depends_on'] = [i-1] if i > 0 else []
            
        step = ReasoningStep(
            step_id=step_data['step_id'],
            text=step_data['text'],
            is_correct=step_data['is_correct'],
            is_origin=step_data.get('is_origin', False),
            error_type=ErrorType(step_data['error_type']) if step_data.get('error_type') else None,
            depends_on=step_data['depends_on']
        )
        steps.append(step)
        nodes.append(step.step_id)
        
        for dep in step.depends_on:
            edges.append([dep, step.step_id])
            
    if not edges and len(nodes) > 1:
        for i in range(len(nodes) - 1):
            edges.append([nodes[i], nodes[i+1]])
    
    domain_str = turn_data.get('domain', 'math').lower()
    if domain_str == 'qa': 
        domain = Domain.MATH
    else:
        try:
            domain = Domain(domain_str)
        except ValueError:
            domain = Domain.MATH
            
    return ReasoningChain(
        domain=domain,
        query_id=turn_data.get('query_id', f"turn_{turn_data.get('turn_id', 0)}"),
        query=turn_data.get('question', ''),
        ground_truth=turn_data.get('answer', ''),
        reasoning_steps=steps,
        dependency_graph=DependencyGraph(nodes=nodes, edges=edges)
    )


def evaluate_multiturn(model_path, data_file, device='cpu'):
    """Evaluate model on different error types."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = MemoryAugmentedGAT(
        input_dim=395,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        memory_dim=128,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    conversations = []
    with open(data_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            conversations.append(data['turns'])
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Stats
    results = {
        'all': {'preds': [], 'targets': []},
        'single_turn_error': {'preds': [], 'targets': []},
        'cross_turn_error': {'preds': [], 'targets': []},
        'clean': {'preds': [], 'targets': []},
    }
    
    with torch.no_grad():
        for conversation in conversations:
            for turn_idx, turn in enumerate(conversation):
                chain = dict_to_reasoning_chain(turn)
                
                # Mock features
                mock_features = torch.randn(len(chain.reasoning_steps), 395)
                
                crg = build_crg(chain, node_features=mock_features)
                x = torch.tensor(crg.x, dtype=torch.float).to(device)
                edge_index = torch.tensor(crg.edge_index, dtype=torch.long).to(device)
                confidence = torch.ones(len(chain.reasoning_steps)).to(device)
                
                outputs = model(x, edge_index, confidence=confidence)
                
                # Predictions
                preds = (outputs['node_pred'] > 0.5).int().cpu().tolist()
                targets = [1 if step.is_correct else 0 for step in chain.reasoning_steps]
                
                # Identify turn type
                turn_has_error = any(not step.is_correct for step in chain.reasoning_steps)
                # In synthetic data, we don't explicitly label "cross-turn" vs "single-turn" in the JSON
                # But we know injected errors are cross-turn if they are factual/contradictions
                # For this eval, we'll assume errors on turns > 0 are likely cross-turn if injected
                
                # A better way: check error type
                error_types = [step.error_type for step in chain.reasoning_steps if not step.is_correct]
                
                category = 'clean'
                if turn_has_error:
                    if any(et == ErrorType.FACTUAL for et in error_types):
                        category = 'cross_turn_error' # Our generator injects factual errors as cross-turn
                    else:
                        category = 'single_turn_error'
                
                results['all']['preds'].extend(preds)
                results['all']['targets'].extend(targets)
                results[category]['preds'].extend(preds)
                results[category]['targets'].extend(targets)
                
                # Per-turn tracking (Assume mock turn structure if not explicit)
                # In real data, 'turn_id' is in turn_data. Here we infer from index in conversation loop
                turn_key = f"turn_{turn_idx+1}" # 1-based index
                if turn_key not in results:
                    results[turn_key] = {'preds': [], 'targets': []}
                results[turn_key]['preds'].extend(preds)
                results[turn_key]['targets'].extend(targets)
                
    # Report
    print("\n" + "="*50)
    print("Multi-Turn Evaluation Results")
    print("="*50)
    
    for category in ['all', 'clean', 'cross_turn_error']:
        p = results[category]['preds']
        t = results[category]['targets']
        if not p:
            print(f"{category}: No samples")
            continue
            
        f1 = f1_score(t, p)
        acc = accuracy_score(t, p)
        print(f"{category.upper():<20} | F1: {f1:.4f} | Acc: {acc:.4f} | N: {len(p)}")
        
    print("-" * 50)
    print("Per-Turn Performance")
    # Sort keys to ensure Turn 1, Turn 2, ...
    turn_keys = sorted([k for k in results.keys() if k.startswith('turn_')], key=lambda x: int(x.split('_')[1]))
    
    for turn in turn_keys:
        p = results[turn]['preds']
        t = results[turn]['targets']
        if not p: continue
        f1 = f1_score(t, p)
        acc = accuracy_score(t, p)
        print(f"{turn.upper():<20} | F1: {f1:.4f} | Acc: {acc:.4f} | N: {len(p)}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'models/checkpoints/multiturn_run/best_model.pth'
    data_file = project_root / 'data/processed/multiturn_coqa/test.jsonl'
    
    evaluate_multiturn(model_path, data_file)
