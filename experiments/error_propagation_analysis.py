"""
Error Propagation Analysis.

Analyzes how errors propagate through reasoning chains:
1. Origin detection accuracy
2. Propagation path analysis
3. Contamination distance
4. Intervention points
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict


def analyze_error_propagation(
    model,
    test_loader,
    device="cpu"
) -> Dict:
    """
    Analyze error propagation patterns.
    
    Returns:
        Dictionary with propagation statistics
    """
    model.eval()
    model.to(device)
    
    propagation_stats = {
        "total_graphs": 0,
        "graphs_with_errors": 0,
        "origin_detected": 0,
        "origin_missed": 0,
        "propagation_distances": [],
        "contamination_ratios": [],
        "contamination_ratios": [],
        "intervention_points": [],
    }
    
    # Track error probability at each depth: depth -> {total: count, errors: count}
    depth_stats = defaultdict(lambda: {"total": 0, "errors": 0})
    
    per_graph_analysis = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Analyzing propagation", leave=False):
            batch = batch.to(device)
            
            # Get predictions
            outputs = model(batch.x, batch.edge_index, 
                          confidence=getattr(batch, 'confidence', None),
                          batch=batch.batch)
            
            node_preds = (outputs['node_pred'] > 0.5).long().cpu().numpy()
            origin_preds = (outputs['origin_pred'] > 0.5).long().cpu().numpy()
            
            node_labels = batch.y.cpu().numpy()
            origin_labels = batch.y_origin.cpu().numpy()
            edge_index = batch.edge_index.cpu().numpy()
            batch_idx = batch.batch.cpu().numpy()
            
            # Analyze each graph in batch
            num_graphs = batch_idx.max().item() + 1
            
            for graph_id in range(num_graphs):
                # Get nodes in this graph
                mask = batch_idx == graph_id
                graph_nodes = np.where(mask)[0]
                
                if len(graph_nodes) == 0:
                    continue
                
                propagation_stats["total_graphs"] += 1
                
                # Get labels and predictions for this graph
                graph_node_labels = node_labels[mask]
                graph_origin_labels = origin_labels[mask]
                graph_node_preds = node_preds[mask]
                graph_origin_preds = origin_preds[mask]
                
                # Check if graph has errors
                has_errors = (graph_node_labels == 0).any()
                if not has_errors:
                    continue
                
                propagation_stats["graphs_with_errors"] += 1
                
                # Find true origin
                true_origin_idx = np.where(graph_origin_labels == 1)[0]
                pred_origin_idx = np.where(graph_origin_preds == 1)[0]
                
                origin_detected = len(true_origin_idx) > 0 and len(pred_origin_idx) > 0
                if origin_detected and true_origin_idx[0] in pred_origin_idx:
                    propagation_stats["origin_detected"] += 1
                else:
                    propagation_stats["origin_missed"] += 1
                
                # Build graph structure for this subgraph
                # Map global indices to local
                global_to_local = {global_idx: local_idx 
                                  for local_idx, global_idx in enumerate(graph_nodes)}
                
                # Get edges for this graph
                graph_edges = []
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[:, i]
                    if src in graph_nodes and dst in graph_nodes:
                        local_src = global_to_local[src]
                        local_dst = global_to_local[dst]
                        graph_edges.append((local_src, local_dst))
                
                if len(graph_edges) == 0:
                    continue
                
                # Create NetworkX graph
                G = nx.DiGraph()
                G.add_nodes_from(range(len(graph_nodes)))
                G.add_edges_from(graph_edges)
                
                # Analyze propagation from origin
                if len(true_origin_idx) > 0:
                    origin_local = true_origin_idx[0]
                    
                    # Compute distances from origin
                    try:
                        distances = nx.single_source_shortest_path_length(G, origin_local)
                        
                        # Count contaminated nodes at each distance
                        contamination_by_distance = defaultdict(list)
                        for node_idx, distance in distances.items():
                            is_contaminated = graph_node_labels[node_idx] == 0
                            contamination_by_distance[distance].append(is_contaminated)
                        

                        
                        # Update depth statistics
                        for node_idx, distance in distances.items():
                            depth_stats[distance]["total"] += 1
                            if graph_node_labels[node_idx] == 0:  # Error
                                depth_stats[distance]["errors"] += 1
                        
                        # Debug print (remove later)
                        # print(f"Updated stats for graph {graph_id}, distances: {list(distances.values())}")
                        
                        # Compute contamination ratio at each distance
                        for distance, contaminations in contamination_by_distance.items():
                            if distance > 0:  # Exclude origin itself
                                ratio = sum(contaminations) / len(contaminations)
                                propagation_stats["contamination_ratios"].append(ratio)
                                propagation_stats["propagation_distances"].append(distance)
                        
                        # Find intervention points (correct nodes with contaminated descendants)
                        for node_idx in range(len(graph_nodes)):
                            if graph_node_labels[node_idx] == 1:  # Correct node
                                # Check if any descendants are contaminated
                                try:
                                    descendants = nx.descendants(G, node_idx)
                                    if any(graph_node_labels[d] == 0 for d in descendants):
                                        # This is a potential intervention point
                                        propagation_stats["intervention_points"].append(1)
                                except:
                                    pass
                    
                    except nx.NetworkXError:
                        # Graph might not be connected
                        pass
                
                # Store per-graph analysis
                per_graph_analysis.append({
                    "graph_id": int(graph_id),
                    "num_nodes": len(graph_nodes),
                    "num_errors": int((graph_node_labels == 0).sum()),
                    "origin_detected": bool(origin_detected),
                    "true_origin": int(true_origin_idx[0]) if len(true_origin_idx) > 0 else -1,
                    "pred_origin": int(pred_origin_idx[0]) if len(pred_origin_idx) > 0 else -1,
                })
    
    # Compute probability of error at depth D
    depth_stats = defaultdict(lambda: {"total": 0, "errors": 0})
    
    # Compute summary statistics
    summary = {
        "total_graphs": propagation_stats["total_graphs"],
        "graphs_with_errors": propagation_stats["graphs_with_errors"],
        "origin_detection_rate": (propagation_stats["origin_detected"] / 
                                 max(propagation_stats["graphs_with_errors"], 1)),
        "avg_propagation_distance": float(np.mean(propagation_stats["propagation_distances"])) 
                                   if propagation_stats["propagation_distances"] else 0,
        "avg_contamination_ratio": float(np.mean(propagation_stats["contamination_ratios"]))
                                  if propagation_stats["contamination_ratios"] else 0,
        "num_intervention_points": len(propagation_stats["intervention_points"]),
        "per_graph_analysis": per_graph_analysis[:20],  # Sample of first 20 graphs
    }

    # Aggregate depth stats across all graphs in this batch/loader
    # Note: We need to re-iterate or collect them during the main loop. 
    # Let's add the logic inside the loop above instead of here.
    pass 
    
    print(f"DEBUG: depth_stats size: {len(depth_stats)}")
    print(f"DEBUG: depth_stats content: {dict(depth_stats)}")
    return summary, dict(depth_stats)


def main():
    from src.data_processing.graph_dataloader import create_dataloaders
    from src.graph_construction.feature_extractor import FeatureExtractor
    from models.gnn_architectures.gat_model import ConfidenceGatedGAT
    
    print("="*80)
    print("ERROR PROPAGATION ANALYSIS")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Models to analyze
    model_configs = {
        "Math": {
            "checkpoint": project_root / "models/checkpoints/test_run/best_model.pth",
            "data": project_root / "data/processed/test_splits",
            "cache": project_root / "data/graphs/test_cache",
        },
        "Code": {
            "checkpoint": project_root / "models/checkpoints/code_improved_run/best_model.pth",
            "data": project_root / "data/processed/code_test_splits",
            "cache": project_root / "data/graphs/code_cache",
        },
    }
    
    feature_extractor = FeatureExtractor(embedding_dim=384)
    all_results = {}
    
    for domain, config in model_configs.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {domain} Domain")
        print(f"{'='*80}")
        
        if not config["checkpoint"].exists():
            print(f"✗ Checkpoint not found")
            continue
        
        # Load model
        print("Loading model...")
        model = ConfidenceGatedGAT(
            input_dim=384 + 5 + 6,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            use_confidence_gating=True,
        )
        
        try:
            checkpoint = torch.load(config["checkpoint"], map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            continue
        
        # Load data
        print("Loading test data...")
        try:
            _, _, test_loader = create_dataloaders(
                train_path=config["data"] / "train.jsonl",
                val_path=config["data"] / "val.jsonl",
                test_path=config["data"] / "test.jsonl",
                batch_size=16,
                num_workers=0,
                feature_extractor=feature_extractor,
                cache_dir=config["cache"],
            )
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            continue
        
        # Analyze propagation
        print("Analyzing error propagation patterns...")
        results, depth_stats = analyze_error_propagation(model, test_loader)
        all_results[domain] = results
        all_results[f"{domain}_depth_stats"] = depth_stats
        
        # Print results
        print(f"\n📊 PROPAGATION STATISTICS:")
        print(f"  Total graphs analyzed: {results['total_graphs']}")
        print(f"  Graphs with errors: {results['graphs_with_errors']}")
        print(f"  Origin detection rate: {results['origin_detection_rate']*100:.2f}%")
        print(f"  Avg propagation distance: {results['avg_propagation_distance']:.2f} steps")
        print(f"  Avg contamination ratio: {results['avg_contamination_ratio']*100:.2f}%")
        print(f"  Intervention points found: {results['num_intervention_points']}")
    
    # Save results
    output_path = project_root / "experiments" / "error_propagation_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*80)
    print("ERROR PROPAGATION ANALYSIS COMPLETE ✅")
    print("="*80)
    print("\nKey Insights:")
    print("  - Origin detection rate: How often the first error is correctly identified")
    print("  - Propagation distance: How far errors spread from origin")
    print("  - Contamination ratio: Percentage of downstream nodes affected")
    print("  - Intervention points: Correct nodes that could prevent further propagation")


if __name__ == "__main__":
    main()

