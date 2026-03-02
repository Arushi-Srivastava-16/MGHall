"""
Evaluate Adversarial Robustness.

Runs PGD attack on the test set and reports Attack Success Rate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from src.graph_construction.crg_builder import build_crg
from src.data_processing.unified_schema import ReasoningChain, ReasoningStep, DependencyGraph, Domain
from models.gnn_architectures.safety_gat import SafetyGAT
from src.adversarial.pgd_attack import PGDAttack

# Reusing helper
def dict_to_reasoning_chain(data: dict) -> ReasoningChain:
    steps = []
    nodes = []
    if 'reasoning_steps' not in data:
         data['reasoning_steps'] = [{'step_id': 0, 'text': data['query'], 'is_correct': True, 'depends_on': []}]
    for i, step_data in enumerate(data['reasoning_steps']):
        step = ReasoningStep(
            step_id=step_data.get('step_id', i),
            text=step_data['text'],
            is_correct=step_data.get('is_correct', True),
            is_origin=step_data.get('is_origin', False),
            depends_on=step_data.get('depends_on', [i-1] if i > 0 else [])
        )
        steps.append(step)
        nodes.append(step.step_id)
    edges = []
    if len(nodes) > 1:
        for i in range(len(nodes) - 1):
            edges.append([nodes[i], nodes[i+1]])
    if len(nodes) == 1:
        edges.append([0, 0])
    return ReasoningChain(
        domain=Domain.MATH, 
        query_id="robust_sample",
        query=data['query'],
        ground_truth="N/A",
        reasoning_steps=steps,
        dependency_graph=DependencyGraph(nodes=nodes, edges=edges)
    )

class SemanticFeatureExtractor:
    def __init__(self, device='cpu'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.cache = {}
    def get_features(self, texts):
        uncached = [t for t in texts if t not in self.cache]
        if uncached:
            embs = self.model.encode(uncached, convert_to_tensor=True)
            for t, e in zip(uncached, embs):
                self.cache[t] = e
        return torch.stack([self.cache[t] for t in texts])

def evaluate_robustness():
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data/processed/safety/safety_train.jsonl'
    model_path = project_root / 'models/checkpoints/safety_run/safety_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Model
    model = SafetyGAT(input_dim=395).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    encoder = SemanticFeatureExtractor(device=device)
    
    # Init Attack
    # Epsilon 0.1 is a reasonable perturbation for normalized embeddings
    attacker = PGDAttack(model, epsilon=0.1, alpha=0.02, num_steps=10, device=device)
    
    # Load Data
    dataset = []
    with open(data_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
            
    # Use test split (last 20%)
    split_idx = int(len(dataset) * 0.8)
    test_data = dataset[split_idx:]
    print(f"Evaluating robustness on {len(test_data)} samples...")
    
    clean_preds = []
    robust_preds = []
    targets = []
    perturbations = []
    
    for item in tqdm(test_data):
        chain = dict_to_reasoning_chain(item)
        texts = [step.text for step in chain.reasoning_steps]
        features = encoder.get_features(texts)
        if features.size(1) < 395:
            padding = torch.zeros(features.size(0), 395 - features.size(1)).to(device)
            features = torch.cat([features, padding], dim=1)
        
        crg = build_crg(chain, node_features=features)
        
        x = features
        eid = torch.tensor(crg.edge_index, dtype=torch.long).to(device)
        bid = torch.zeros(x.size(0), dtype=torch.long).to(device)
        target = torch.tensor([1.0 if item['is_safe'] else 0.0], dtype=torch.float).to(device)
        
        # Clean Prediction
        with torch.no_grad():
            out_clean = model(x, eid, batch=bid)
            prob_clean = torch.sigmoid(out_clean['safety_pred']).item()
            pred_clean = 1 if prob_clean > 0.6 else 0
        
        # Attack!
        x_adv = attacker.attack(x, eid, bid, target)
        
        # Robust Prediction
        with torch.no_grad():
            out_adv = model(x_adv, eid, batch=bid)
            prob_adv = torch.sigmoid(out_adv['safety_pred']).item()
            pred_adv = 1 if prob_adv > 0.6 else 0
            
        clean_preds.append(pred_clean)
        robust_preds.append(pred_adv)
        targets.append(int(item['is_safe']))
        
        # Measure perturbation magnitude
        perturbations.append((x_adv - x).norm().item())
        
    acc_clean = accuracy_score(targets, clean_preds)
    acc_robust = accuracy_score(targets, robust_preds)
    avg_perturbation = sum(perturbations) / len(perturbations)
    
    print("\n=== Robustness Evaluation Results ===")
    print(f"Clean Accuracy:  {acc_clean:.4f}")
    print(f"Robust Accuracy: {acc_robust:.4f}")
    print(f"Accuracy Drop:   {acc_clean - acc_robust:.4f}")
    print(f"Avg Perturbation L2: {avg_perturbation:.4f}")
    
    # Calculate Attack Success Rate (samples that were correct but flipped)
    flipped = 0
    correct_clean = 0
    for c, r, t in zip(clean_preds, robust_preds, targets):
        if c == t:
            correct_clean += 1
            if r != t:
                flipped += 1
                
    asr = flipped / correct_clean if correct_clean > 0 else 0
    print(f"Attack Success Rate (ASR): {asr:.4f}")

if __name__ == "__main__":
    evaluate_robustness()
