#!/bin/bash
# Script to convert all datasets to unified format

set -e  # Exit on error

echo "================================"
echo "Dataset Conversion Pipeline"
echo "================================"

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "Step 1: Converting PRM800K (Math domain)..."
echo "-------------------------------------------"
cd src/data_processing
python prm800k_converter.py
cd ../..

echo ""
echo "Step 2: Converting HumanEval (Code domain)..."
echo "----------------------------------------------"
cd src/data_processing
python humaneval_converter.py
cd ../..

echo ""
echo "Step 3: Converting MedHallu (Medical domain)..."
echo "------------------------------------------------"
cd src/data_processing
python medhallu_converter.py
cd ../..

echo ""
echo "Step 4: Validating converted data..."
echo "-------------------------------------"
cd src/data_processing
python validator.py
cd ../..

echo ""
echo "Step 5: Creating train/val/test splits..."
echo "------------------------------------------"
cd src/data_processing
python splitter.py
cd ../..

echo ""
echo "================================"
echo "✅ Conversion Complete!"
echo "================================"
echo ""
echo "Converted data location: data/processed/"
echo "Splits location: data/processed/splits/"
echo ""
echo "Next steps:"
echo "  1. Review split statistics in data/processed/splits/split_statistics.json"
echo "  2. Start training: cd experiments && python train_gnn_math.py"


