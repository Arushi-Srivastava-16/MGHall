# Dataset Download Guide for Multi-Domain GNN Training

## ✅ PRM800K Dataset (Math Domain) - DOWNLOADED

**Status:** Successfully downloaded  
**Location:** `/prm800k/prm800k/data/`

### Dataset Statistics:
- **Phase 1 Train:** 949 samples (7.5 MB)
- **Phase 1 Test:** 106 samples (812 KB)
- **Phase 2 Train:** 97,782 samples (435 MB) ⭐ Main training data
- **Phase 2 Test:** 2,762 samples (12 MB)
- **Total:** ~101,599 labeled solutions with step-level correctness ratings

### Data Format:
Each line in the `.jsonl` files contains:
- `question`: Problem text, ground truth solution, and answer
- `label.steps[]`: Step-level ratings (-1, 0, +1) for each solution step
- `pre_generated_steps[]`: The model-generated solution steps
- `pre_generated_answer`: Model's final answer
- `pre_generated_verifier_score`: PRM score for the solution

### Key Fields for GNN Training:
```python
{
  "question": {
    "problem": "...",           # Math problem text
    "ground_truth_answer": "...", # Correct answer
    "pre_generated_steps": [...]  # Solution steps to label
  },
  "label": {
    "steps": [
      {
        "completions": [{
          "text": "...",        # Step text
          "rating": -1/0/+1     # Correctness label
        }]
      }
    ]
  }
}
```

### Usage:
```python
import json

# Load training data
with open('prm800k/prm800k/data/phase2_train.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        steps = sample['label']['steps']
        # Extract step-level labels for GNN training
```

---

## 📋 Remaining Datasets to Download

### 1. MedHallu (Medical Domain)
**Target:** 10K examples  
**Status:** ⏳ Ready to download  
**Expected Accuracy:** 60-70%

**Download Instructions:**
```python
from datasets import load_dataset

# Load labeled split
medhallu_labeled = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")

# Load artificial split
medhallu_artificial = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
```

**HuggingFace:** https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu

### 2. Stanford Legal (Legal Domain)
**Target:** 5K examples  
**Status:** ⏳ Not downloaded  
**Expected Accuracy:** 50-60%

**How to find/download:**
- Search for "Stanford Legal dataset" or "legal reasoning dataset"
- Check: `github.com/stanfordnlp/legal-datasets`
- May be part of larger legal NLP collections

### 3. HumanEval (Code Domain)
**Target:** 3K examples  
**Status:** ⏳ Ready to download  
**Expected Accuracy:** 65-75%

**Download Instructions:**
```bash
git clone https://github.com/openai/human-eval.git
```

**GitHub:** https://github.com/openai/human-eval  
**HuggingFace Alternative:** `huggingface.co/datasets/openai_humaneval`

### 4. Commonsense Synthetic (Commonsense Domain)
**Target:** 5K examples  
**Status:** ⏳ Not downloaded  
**Expected Accuracy:** 80-85%

**How to find/download:**
- May need to generate synthetically
- Check: CommonsenseQA, WinoGrande, or similar datasets
- Consider using LLM-generated commonsense reasoning problems

---

## 🚀 Quick Start Commands

### For PRM800K (Already Done):
```bash
cd /Users/arushisrivastava/Documents/GitHub/MGHall
git clone https://github.com/openai/prm800k.git
cd prm800k
git lfs pull  # If files appear as pointers
```

### For Other Datasets (When Found):
```bash
# HuggingFace datasets
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('dataset_name'); ds.save_to_disk('path')"

# Git repositories
git clone <repository_url>

# Direct downloads
wget <url> -O dataset.zip
unzip dataset.zip
```

---

## 📊 Expected Dataset Summary

| Domain | Dataset | Size | Status | Location |
|--------|---------|------|--------|----------|
| Math | PRM800K | 800K | ✅ Downloaded | `prm800k/prm800k/data/` |
| Medical | MedHallu | 10K | ⏳ Pending | TBD |
| Legal | Stanford Legal | 5K | ⏳ Pending | TBD |
| Code | HumanEval+ | 3K | ⏳ Pending | TBD |
| Commonsense | Synthetic | 5K | ⏳ Pending | TBD |

---

## 🔍 Next Steps

1. **Verify PRM800K data integrity:**
   ```bash
   cd prm800k/prm800k/data
   wc -l *.jsonl  # Should match counts above
   ```

2. **Search for remaining datasets:**
   - Use HuggingFace dataset search
   - Check GitHub for dataset repositories
   - Review papers citing these datasets

3. **Prepare data preprocessing pipeline:**
   - Standardize format across all domains
   - Extract step-level labels
   - Create graph structures for GNN input

4. **Set up GNN training framework:**
   - Choose GNN architecture (GCN, GAT, GraphSAGE, etc.)
   - Define node/edge features from solution steps
   - Implement domain-specific training loops

---

## 📝 Notes

- PRM800K uses **Git LFS** for large files - already handled ✅
- All datasets should be converted to a unified format for multi-domain training
- Consider data augmentation for smaller datasets (Medical, Legal, Code)
- Phase 2 data from PRM800K is the main training set (97K+ samples)

