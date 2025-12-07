# Dataset Download Status

## ✅ Successfully Downloaded

### 1. PRM800K (Math Domain) - COMPLETE
- **Location:** `prm800k/prm800k/data/`
- **Size:** ~465 MB total
- **Samples:**
  - Phase 1 Train: 949 samples (7.5 MB)
  - Phase 1 Test: 106 samples (812 KB)
  - Phase 2 Train: **97,782 samples (435 MB)** ⭐ Main dataset
  - Phase 2 Test: 2,762 samples (12 MB)
- **Total:** ~101,599 labeled solutions
- **Format:** JSONL with step-level correctness ratings (-1, 0, +1)
- **Expected GNN Accuracy:** 75-85%

**Data Structure:**
```json
{
  "question": {
    "problem": "Math problem text",
    "ground_truth_answer": "Correct answer",
    "pre_generated_steps": ["step1", "step2", ...]
  },
  "label": {
    "steps": [{
      "completions": [{
        "text": "step text",
        "rating": -1/0/+1
      }]
    }]
  }
}
```

### 2. HumanEval (Code Domain) - COMPLETE
- **Location:** `human-eval/data/`
- **Size:** Compressed (HumanEval.jsonl.gz)
- **Samples:** 164 programming problems
- **Format:** JSONL with Python code prompts, solutions, and tests
- **Expected GNN Accuracy:** 65-75%

**Data Structure:**
```json
{
  "task_id": "HumanEval/0",
  "prompt": "Function signature and docstring",
  "canonical_solution": "Correct implementation",
  "test": "Test cases"
}
```

**Note:** For 3K examples target, you may need to:
- Generate multiple solutions per problem
- Use HumanEval+ extensions
- Augment with similar code datasets

---

## ⏳ Pending Downloads

### 3. MedHallu (Medical Domain)
- **Status:** Ready to download
- **Target:** 10K examples
- **Expected Accuracy:** 60-70%
- **Download Command:**
  ```bash
  pip install datasets
  python3 -c "from datasets import load_dataset; \
    ds = load_dataset('UTAustin-AIHealth/MedHallu', 'pqa_labeled'); \
    ds.save_to_disk('medhallu')"
  ```
- **HuggingFace:** https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu

### 4. Stanford Legal (Legal Domain)
- **Status:** Manual download required
- **Target:** 5K examples
- **Expected Accuracy:** 50-60%
- **Resources:**
  - Stanford NLP: https://nlp.stanford.edu/projects/legal/
  - May need to search for "Stanford Legal NLP dataset" or "legal case dataset"
  - Alternative: Check HuggingFace for legal datasets

### 5. Commonsense Synthetic (Commonsense Domain)
- **Status:** May need to generate
- **Target:** 5K examples
- **Expected Accuracy:** 80-85%
- **Options:**
  - CommonsenseQA: https://www.tau-nlp.org/commonsenseqa
  - WinoGrande: https://winogrande.allenai.org/
  - Generate synthetically using LLMs
  - HuggingFace: Search for "commonsense" datasets

---

## 📊 Current Dataset Summary

| Domain | Dataset | Status | Samples | Size | Location |
|--------|---------|--------|---------|------|----------|
| Math | PRM800K | ✅ | ~101K | 465 MB | `prm800k/prm800k/data/` |
| Code | HumanEval | ✅ | 164 | <1 MB | `human-eval/data/` |
| Medical | MedHallu | ⏳ | 0 | - | Download needed |
| Legal | Stanford Legal | ⏳ | 0 | - | Download needed |
| Commonsense | Synthetic | ⏳ | 0 | - | Generate/Download needed |

---

## 🚀 Quick Download Commands

### Install Dependencies
```bash
pip install datasets
```

### Download MedHallu
```bash
python3 download_datasets.py
# Or manually:
python3 -c "from datasets import load_dataset; \
  labeled = load_dataset('UTAustin-AIHealth/MedHallu', 'pqa_labeled'); \
  labeled.save_to_disk('medhallu/pqa_labeled')"
```

### Verify Downloads
```bash
# Check PRM800K
wc -l prm800k/prm800k/data/*.jsonl

# Check HumanEval
gunzip -c human-eval/data/HumanEval.jsonl.gz | wc -l

# Run download script
python3 download_datasets.py
```

---

## 📝 Next Steps

1. **Download MedHallu:**
   ```bash
   pip install datasets
   python3 download_datasets.py
   ```

2. **Find Stanford Legal Dataset:**
   - Visit Stanford NLP legal project page
   - Check HuggingFace for legal reasoning datasets
   - Consider alternatives like LegalBench

3. **Prepare Commonsense Dataset:**
   - Download CommonsenseQA or WinoGrande
   - Or generate synthetic commonsense problems

4. **Data Preprocessing:**
   - Standardize format across all domains
   - Extract step-level labels for GNN training
   - Create graph structures (nodes=steps, edges=dependencies)

5. **GNN Training Setup:**
   - Choose architecture (GCN, GAT, GraphSAGE)
   - Define node/edge features
   - Implement domain-specific training loops

---

## 🔍 Dataset Details

### PRM800K Usage Example
```python
import json

with open('prm800k/prm800k/data/phase2_train.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        problem = sample['question']['problem']
        steps = sample['label']['steps']
        for step in steps:
            rating = step['completions'][0]['rating']  # -1, 0, or +1
            text = step['completions'][0]['text']
```

### HumanEval Usage Example
```python
import json
import gzip

with gzip.open('human-eval/data/HumanEval.jsonl.gz', 'rt') as f:
    for line in f:
        problem = json.loads(line)
        task_id = problem['task_id']
        prompt = problem['prompt']
        solution = problem['canonical_solution']
        tests = problem['test']
```

---

## 📚 References

- **PRM800K:** https://github.com/openai/prm800k
- **HumanEval:** https://github.com/openai/human-eval
- **MedHallu:** https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
- **Paper (PRM800K):** https://arxiv.org/abs/2305.20050

