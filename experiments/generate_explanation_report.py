"""
Generate Explanation Report.

Runs evaluation on the test set and generates a Markdown report with attention heatmaps.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.graph_construction.crg_builder import build_crg
from src.data_processing.unified_schema import ReasoningChain, ReasoningStep, DependencyGraph, Domain
from models.gnn_architectures.safety_gat import SafetyGAT
from src.explainability.explainer import ExplanationGenerator

# Reusing helper (duplicated for standalone script)
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
        query_id="report_sample",
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

def generate_report():
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data/processed/safety/safety_train.jsonl'
    model_path = project_root / 'models/checkpoints/safety_run/safety_model.pth'
    report_path = project_root / 'EXPLANATION_REPORT.md'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SafetyGAT(input_dim=395).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    encoder = SemanticFeatureExtractor(device=device)
    explainer = ExplanationGenerator()
    
    # Load Data (Safe and Unsafe)
    dataset = []
    with open(data_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
            
    # Select 10 Unsafe and 10 Safe
    unsafe_samples = [d for d in dataset if not d['is_safe']][:10]
    safe_samples = [d for d in dataset if d['is_safe']][:10]
    test_samples = unsafe_samples + safe_samples
    
    markdown_content = "# Safety Explanation Report\n\n"
    markdown_content += "Visualizing model decisions and attention weights for 20 samples.\n\n"
    markdown_content += "| Query | Prediction | Confidence | True Label |\n"
    markdown_content += "|---|---|---|---|\n"
    
    print("Generating report...")
    
    for i, item in enumerate(tqdm(test_samples)):
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
        
        with torch.no_grad():
            outputs = model(x, eid, batch=bid, return_attention_weights=True)
            
        prob = torch.sigmoid(outputs['safety_pred']).item()
        
        explanation = explainer.explain_safety_decision(
            item['query'], prob, 
            outputs.get('attention_weights'), 
            texts
        )
        heatmap = explainer.generate_attention_heatmap(texts, outputs.get('attention_weights'))
        
        status_icon = "✅" if explanation['status'] == "SAFE" else "❌"
        true_icon = "✅" if item['is_safe'] else "❌"
        
        markdown_content += f"| {item['query'][:50]}... | {status_icon} {explanation['status']} | {explanation['confidence']} | {true_icon} |\n"
        
        # Add detailed section every 5 samples
        if i % 2 == 0:
            markdown_content += f"\n### Detail: {item['query']}\n"
            markdown_content += f"**Reason**: {explanation['reason']}\n\n"
            markdown_content += "**Attention Analysis**:\n"
            for step, score in heatmap.items():
                markdown_content += f"- {step}: `{score:.4f}`\n"
            markdown_content += "\n---\n"

    with open(report_path, 'w') as f:
        f.write(markdown_content)
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    generate_report()
