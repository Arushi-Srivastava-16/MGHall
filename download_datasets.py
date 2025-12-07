#!/usr/bin/env python3
"""
Script to download all datasets for multi-domain GNN training.
"""

import os
import json
import subprocess
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

def download_prm800k():
    """PRM800K is already downloaded via git clone."""
    prm800k_path = BASE_DIR / "prm800k"
    if prm800k_path.exists():
        print("✅ PRM800K already downloaded")
        return True
    else:
        print("❌ PRM800K not found. Run: git clone https://github.com/openai/prm800k.git")
        return False

def download_humaneval():
    """HumanEval is already downloaded via git clone."""
    humaneval_path = BASE_DIR / "human-eval"
    if humaneval_path.exists():
        print("✅ HumanEval already downloaded")
        return True
    else:
        print("❌ HumanEval not found. Run: git clone https://github.com/openai/human-eval.git")
        return False

def download_medhallu():
    """Download MedHallu from HuggingFace."""
    try:
        from datasets import load_dataset
        
        print("📥 Downloading MedHallu dataset from HuggingFace...")
        output_dir = BASE_DIR / "medhallu"
        output_dir.mkdir(exist_ok=True)
        
        # Download labeled split
        print("  - Downloading pqa_labeled split...")
        labeled = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")
        labeled.save_to_disk(str(output_dir / "pqa_labeled"))
        
        # Download artificial split
        print("  - Downloading pqa_artificial split...")
        artificial = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
        artificial.save_to_disk(str(output_dir / "pqa_artificial"))
        
        print(f"✅ MedHallu downloaded to {output_dir}")
        return True
    except ImportError:
        print("❌ 'datasets' library not installed. Run: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ Error downloading MedHallu: {e}")
        return False

def download_stanford_legal():
    """Stanford Legal dataset - manual download required."""
    print("⚠️  Stanford Legal dataset requires manual download:")
    print("   Visit: https://nlp.stanford.edu/projects/legal/")
    print("   Or search for: 'Stanford Legal NLP dataset'")
    return False

def get_dataset_stats():
    """Print statistics about downloaded datasets."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # PRM800K
    prm800k_data = BASE_DIR / "prm800k" / "prm800k" / "data"
    if prm800k_data.exists():
        phase2_train = prm800k_data / "phase2_train.jsonl"
        if phase2_train.exists():
            with open(phase2_train, 'r') as f:
                count = sum(1 for _ in f)
            print(f"📊 PRM800K Phase 2 Train: {count:,} samples")
    
    # HumanEval
    humaneval_data = BASE_DIR / "human-eval" / "data"
    if humaneval_data.exists():
        humaneval_file = humaneval_data / "HumanEval.jsonl"
        if humaneval_file.exists():
            with open(humaneval_file, 'r') as f:
                count = sum(1 for _ in f)
            print(f"📊 HumanEval: {count:,} samples")
    
    # MedHallu
    medhallu_dir = BASE_DIR / "medhallu"
    if medhallu_dir.exists():
        try:
            from datasets import load_from_disk
            labeled_path = medhallu_dir / "pqa_labeled"
            if labeled_path.exists():
                ds = load_from_disk(str(labeled_path))
                print(f"📊 MedHallu (labeled): {len(ds):,} samples")
        except:
            pass
    
    print("="*60 + "\n")

def main():
    print("🚀 Multi-Domain GNN Dataset Downloader")
    print("="*60)
    
    results = {
        "PRM800K": download_prm800k(),
        "HumanEval": download_humaneval(),
        "MedHallu": download_medhallu(),
        "Stanford Legal": download_stanford_legal(),
    }
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for dataset, status in results.items():
        status_icon = "✅" if status else "⏳"
        print(f"{status_icon} {dataset}")
    
    get_dataset_stats()
    
    print("\n💡 Next Steps:")
    print("   1. Install datasets library: pip install datasets")
    print("   2. Run this script again to download MedHallu")
    print("   3. Manually download Stanford Legal dataset")
    print("   4. Prepare data preprocessing pipeline for GNN training")

if __name__ == "__main__":
    main()

