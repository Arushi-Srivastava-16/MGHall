# Scripts Directory

Helper scripts for running common tasks.

## convert_all_datasets.sh

Converts all datasets (PRM800K, HumanEval, MedHallu) to unified format, validates them, and creates train/val/test splits.

**Usage:**
```bash
./scripts/convert_all_datasets.sh
```

**What it does:**
1. Converts PRM800K dataset
2. Converts HumanEval dataset
3. Converts MedHallu dataset
4. Validates all converted data
5. Creates stratified train/val/test splits (70/15/15)

**Requirements:**
- Virtual environment activated (script will activate if found)
- Raw datasets in `data/raw/`
- About 15-30 minutes to complete

**Output:**
- Converted data: `data/processed/*.jsonl`
- Quality reports: `data/processed/*_quality_report.json`
- Splits: `data/processed/splits/{train,val,test}.jsonl`

## Alternative: Run Individually

You can also run each converter individually from the `src/data_processing/` directory:

```bash
cd src/data_processing

# Convert each dataset
python prm800k_converter.py
python humaneval_converter.py
python medhallu_converter.py

# Validate
python validator.py

# Create splits
python splitter.py
```


