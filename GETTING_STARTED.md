# Getting Started with CHG Framework

This guide will help you get up and running with the CHG Framework for multi-domain hallucination detection.

## Prerequisites Checklist

- [ ] Python 3.8 or later installed
- [ ] GPU with CUDA support (recommended but not required)
- [ ] Git LFS installed (`brew install git-lfs` on Mac, or visit https://git-lfs.github.com/)
- [ ] At least 10GB free disk space
- [ ] (Optional) Weights & Biases account for experiment tracking

## Step-by-Step Guide

### Step 1: Environment Setup (5 minutes)

```bash
# Navigate to project directory
cd /Users/arushisrivastava/Documents/GitHub/MGHall

# Activate virtual environment (already exists)
source venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### Step 2: Verify Dataset Downloads (2 minutes)

Check that datasets are in place:

```bash
# Check PRM800K
ls -lh data/raw/prm800k/prm800k/data/*.jsonl

# Expected output:
# phase1_test.jsonl   (812K)
# phase1_train.jsonl  (7.5M)
# phase2_test.jsonl   (12M)
# phase2_train.jsonl  (435M)

# Check HumanEval
ls -lh data/raw/human-eval/data/HumanEval.jsonl.gz

# Check MedHallu
ls -lh data/raw/medhallu/pqa_labeled/train/

# If datasets are missing, run:
python download_datasets.py
```

### Step 3: Data Conversion (15-30 minutes)

Convert raw datasets to unified format:

```bash
# Convert PRM800K (this will take ~10-15 minutes)
python src/data_processing/prm800k_converter.py

# Expected output: Converted JSONL files in data/processed/
# Convert HumanEval (~2 minutes)
python src/data_processing/humaneval_converter.py

# Convert MedHallu (~3 minutes)
python src/data_processing/medhallu_converter.py
```

### Step 4: Validate Data Quality (5 minutes)

```bash
# Run validation pipeline
python src/data_processing/validator.py

# This will generate quality reports showing:
# - Validity rate
# - Domain distribution
# - Chain length statistics
# - Error rate statistics
```

### Step 5: Create Train/Val/Test Splits (3 minutes)

```bash
# Create stratified 70/15/15 splits
python src/data_processing/splitter.py

# Output files will be in data/processed/splits/:
# - train.jsonl
# - val.jsonl
# - test.jsonl
# - split_statistics.json
```

### Step 6: Train Your First Model (1-2 hours)

Start with the Math domain (PRM800K):

```bash
cd experiments

# Train Math GNN (will take 1-2 hours depending on GPU)
python train_gnn_math.py

# Monitor training progress in terminal
# Checkpoints saved to: models/checkpoints/math/
```

**Training Tips:**
- Use `nohup python train_gnn_math.py > train.log 2>&1 &` to run in background
- Check `models/checkpoints/math/training_results.json` for final metrics
- If using W&B, check https://wandb.ai for live metrics

### Step 7: Evaluate the Model (10 minutes)

```bash
# The training script automatically evaluates on test set
# Check the final output or training_results.json

# For cross-domain evaluation, use the evaluator:
python -c "
from src.evaluation.evaluator import ModelEvaluator
from models.gnn_architectures.gat_model import ConfidenceGatedGAT
import torch

model = ConfidenceGatedGAT(input_dim=395, hidden_dim=128, num_layers=3, num_heads=4)
model.load_state_dict(torch.load('models/checkpoints/math/best_model.pth')['model_state_dict'])

evaluator = ModelEvaluator(model)
# ... add evaluation code
"
```

### Step 8: Run Ablation Studies (2-3 hours)

Compare different model architectures:

```bash
# Run all ablation experiments
python ablation_studies.py

# This will train and compare:
# 1. Full GAT with confidence gating
# 2. GAT without confidence gating
# 3. Simple GCN (no attention)
# 4. Sequential LSTM baseline

# Results saved to: experiments/results/ablations/
```

### Step 9: Generate Analysis Reports

```bash
# Create comprehensive analysis including:
# - Cross-domain performance
# - Error analysis
# - Attention visualizations
# - Model comparison heatmaps

python analysis_report_generator.py

# Reports saved to: experiments/results/analysis/
```

## Common Issues & Solutions

### Issue 1: Out of Memory (OOM)

**Solution:** Reduce batch size in training scripts

```python
# In train_gnn_math.py, change:
config["batch_size"] = 16  # to
config["batch_size"] = 8   # or even 4
```

### Issue 2: Datasets Library Not Found

**Solution:**
```bash
pip install datasets
```

### Issue 3: CUDA Not Available

**Solution:** The framework works on CPU, but will be slower. To use CPU explicitly:

```python
# In training scripts, change:
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # to
device = 'cpu'
```

### Issue 4: Sentence Transformers Download Slow

**Solution:** Models will download on first use. Be patient or pre-download:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Downloads once
```

## Quick Test Script

To verify everything is working:

```bash
# Create a test script
cat > test_setup.py << 'EOF'
"""Quick test to verify setup."""
import torch
from pathlib import Path

# Check data
data_dir = Path("data/processed/splits")
assert (data_dir / "train.jsonl").exists(), "Train split not found"
print("✓ Data splits found")

# Check PyTorch
assert torch.cuda.is_available() or True, "PyTorch OK"
print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")

# Check PyG
import torch_geometric
print(f"✓ PyTorch Geometric {torch_geometric.__version__}")

# Check transformers
from sentence_transformers import SentenceTransformer
print("✓ Sentence Transformers OK")

print("\n✅ All systems ready!")
EOF

python test_setup.py
```

## Next Steps

Now that you're set up:

1. **Explore the data**: Open `notebooks/` for data analysis
2. **Train other domains**: Run `train_gnn_code.py` and `train_gnn_medical.py`
3. **Experiment**: Modify hyperparameters in training scripts
4. **Analyze**: Use the evaluation framework to understand model behavior
5. **Contribute**: Implement Phase 4+ features!

## Getting Help

- **Documentation**: See `README.md` for detailed documentation
- **Code Examples**: Check scripts in `experiments/`
- **Issues**: Open a GitHub issue for bugs or questions

## Performance Expectations

On a modern GPU (e.g., RTX 3080):
- **Data conversion**: ~20 minutes total
- **Training (Math)**: ~1-2 hours
- **Training (Code)**: ~30-60 minutes
- **Training (Medical)**: ~45-90 minutes
- **Ablation studies**: ~2-3 hours

On CPU only:
- Multiply above times by 5-10x

## Success Checklist

After completing this guide, you should have:

- [ ] Environment set up and verified
- [ ] All 3 datasets converted to unified format
- [ ] Train/val/test splits created
- [ ] At least one domain model trained
- [ ] Test results showing reasonable accuracy
- [ ] Understanding of the codebase structure

Congratulations! You're ready to use the CHG Framework for hallucination detection research! 🎉

