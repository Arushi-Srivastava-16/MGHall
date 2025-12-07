# CHG Framework: Causal Hallucination Graph Framework

Multi-domain GNN training for hallucination detection and propagation analysis across Math, Code, and Medical reasoning domains.

## Project Overview

This framework implements a Graph Neural Network (GNN) approach to detect and analyze hallucination propagation in multi-step reasoning chains. Based on the strategic plan for building a comprehensive hallucination detection system, this codebase covers **Phases 0-3** of the implementation.

### Key Features

- **Multi-Domain Support**: Train domain-specific GNNs on Math (PRM800K), Code (HumanEval), and Medical (MedHallu) datasets
- **Unified Data Format**: Consistent schema across all domains with step-level correctness labels
- **Causal Reasoning Graphs (CRG)**: Graph-based representation of reasoning chains with dependency tracking
- **Advanced GNN Architecture**: Graph Attention Networks (GAT) with confidence gating
- **Multi-Task Learning**: Simultaneous node classification, origin detection, and error type prediction
- **Comprehensive Evaluation**: Cross-domain analysis, ablation studies, and detailed error reports

## Repository Structure

```
MGHall/
├── data/
│   ├── raw/                    # Original datasets (PRM800K, HumanEval, MedHallu)
│   ├── processed/              # Converted to unified format
│   ├── graphs/                 # Cached PyG graph objects
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data_processing/        # Dataset converters and loaders
│   │   ├── unified_schema.py   # Unified data format definition
│   │   ├── prm800k_converter.py
│   │   ├── humaneval_converter.py
│   │   ├── medhallu_converter.py
│   │   ├── validator.py         # Data quality validation
│   │   ├── splitter.py          # Train/val/test splitting
│   │   └── graph_dataloader.py  # PyG data loaders
│   ├── graph_construction/      # Graph building and features
│   │   ├── crg_builder.py       # Convert chains to PyG graphs
│   │   ├── dependency_enhancer.py  # Semantic/NLI/LLM dependencies
│   │   ├── feature_extractor.py  # Node features (text, topology, task)
│   │   └── visualizer.py         # Graph statistics and visualization
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py            # Multi-task trainer
│   │   └── wandb_config.py       # Experiment tracking
│   └── evaluation/              # Evaluation and analysis
│       └── evaluator.py          # Metrics and cross-domain evaluation
├── models/
│   ├── gnn_architectures/       # GNN implementations
│   │   └── gat_model.py          # GAT with confidence gating, GCN baseline
│   ├── domain_models/            # Trained domain-specific models
│   └── checkpoints/              # Saved model weights
├── experiments/                 # Training and evaluation scripts
│   ├── configs/                  # Experiment configurations
│   ├── results/                  # Logs, metrics, visualizations
│   ├── train_gnn_math.py
│   ├── train_gnn_code.py
│   ├── train_gnn_medical.py
│   ├── ablation_studies.py
│   └── analysis_report_generator.py
├── notebooks/                   # Data analysis notebooks
├── tests/                       # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git LFS (for PRM800K dataset)

### Setup

```bash
# Clone repository
git clone <repository_url>
cd MGHall

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Key Dependencies

- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- Transformers >= 4.30.0
- Sentence-Transformers >= 2.2.0
- W&B (Weights & Biases) for experiment tracking
- NetworkX, Matplotlib, Seaborn for visualization

## Quick Start

### 1. Data Preparation

The datasets are already downloaded in `data/raw/`:
- **PRM800K**: 97,782 math reasoning chains with step-level labels
- **HumanEval**: 164 code problems (will be augmented to 3K)
- **MedHallu**: 1,000 medical Q&A with hallucination labels

Convert to unified format:

```bash
# Convert PRM800K
cd src/data_processing
python prm800k_converter.py

# Convert HumanEval
python humaneval_converter.py

# Convert MedHallu
python medhallu_converter.py
```

Validate and create splits:

```bash
# Validate converted data
python validator.py

# Create stratified train/val/test splits (70/15/15)
python splitter.py
```

### 2. Training

Train domain-specific GNNs:

```bash
# Math domain
cd experiments
python train_gnn_math.py

# Code domain
python train_gnn_code.py

# Medical domain
python train_gnn_medical.py
```

### 3. Evaluation & Analysis

```bash
# Run ablation studies
python ablation_studies.py

# Generate comprehensive analysis reports
python analysis_report_generator.py
```

## Data Format

### Unified Schema

All datasets are converted to a unified JSON format:

```json
{
  "domain": "math|code|medical",
  "query_id": "unique_id",
  "query": "original problem/question",
  "ground_truth": "correct answer",
  "reasoning_steps": [
    {
      "step_id": 0,
      "text": "step content",
      "is_correct": true,
      "is_origin": false,
      "error_type": "factual|logical|syntax|null",
      "depends_on": [previous_step_ids]
    }
  ],
  "dependency_graph": {
    "nodes": [step_ids],
    "edges": [[from_id, to_id]]
  }
}
```

## Model Architecture

### Graph Attention Network with Confidence Gating

The primary model is a 3-layer GAT with:
- **Attention Mechanism**: 4 attention heads per layer
- **Confidence Gating**: Prevents low-confidence nodes from contaminating neighbors
- **Multi-Task Heads**:
  - Node classification (correctness prediction)
  - Origin detection (first error location)
  - Error type classification (factual/logical/syntax)

### Baselines

- **Simple GCN**: Graph Convolutional Network without attention
- **Sequential LSTM**: No graph structure, treats chains as sequences

## Expected Results

Based on the strategic plan:

| Domain | Dataset | Expected Accuracy | Target Origin Detection |
|--------|---------|------------------|------------------------|
| Math | PRM800K | 75-85% | >85% |
| Code | HumanEval | 65-75% | >75% |
| Medical | MedHallu | 60-70% | >70% |

## Experiment Tracking

We use Weights & Biases for experiment tracking:

```python
# Initialize in training script
from src.training.wandb_config import init_wandb

init_wandb(
    project_name="chg-framework",
    experiment_name="math-gnn-training",
    config=config,
    tags=["math", "prm800k", "gat"],
)
```

## Advanced Usage

### Custom Feature Extraction

```python
from src.graph_construction.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(embedding_model="all-MiniLM-L6-v2")
features = extractor.extract_all_features(
    chain,
    include_text=True,
    include_topology=True,
    include_task=True,
)
```

### Dependency Enhancement

```python
from src.graph_construction.dependency_enhancer import DependencyEnhancer

enhancer = DependencyEnhancer(use_nli=True)
enhanced_chain = enhancer.enhance(chain, method="semantic", threshold=0.7)
```

### Cross-Domain Evaluation

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model)
results = evaluator.cross_domain_evaluation(
    loaders={"math": math_loader, "code": code_loader},
    trained_on="math",
)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_data_processing.py

# With coverage
pytest --cov=src tests/
```

## Project Status: Phases 0-3 Complete ✅

- ✅ **Phase 0**: Foundation & Infrastructure
- ✅ **Phase 1**: Data Foundation (converters, validation, splits)
- ✅ **Phase 2**: CRG Construction Pipeline (graphs, features, loaders)
- ✅ **Phase 3**: GNN Architecture & Training (GAT, training, evaluation, ablations)

### Next Steps (Future Phases)

- **Phase 4**: Proactive Prediction (vulnerability predictor, interventional control)
- **Phase 5**: Multi-Model Fingerprinting (cross-model analysis)
- **Phase 6**: Theoretical Formalization (hallucination propagation theorems)
- **Phase 7**: Paper Writing
- **Phase 8**: HART System (production deployment)

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{chg-framework-2024,
  title={CHG Framework: Multi-Domain Hallucination Detection via Causal Reasoning Graphs},
  author={Your Name},
  year={2024},
  note={Research implementation}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Acknowledgments

- **PRM800K Dataset**: OpenAI Research
- **HumanEval**: OpenAI Codex
- **MedHallu**: UT Austin AI Health
- **PyTorch Geometric**: PyG Team

