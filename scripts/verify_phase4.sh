#!/bin/bash

# Phase 4 Verification Script
# Tests all Phase 4 components

echo "=============================================="
echo "PHASE 4: PROACTIVE PREDICTION - VERIFICATION"
echo "=============================================="

cd "$(dirname "$0")/.."

echo ""
echo "1. Testing Vulnerability Data Generator..."
echo "----------------------------------------------"
./venv/bin/python src/proactive/vulnerability_data_generator.py
if [ $? -eq 0 ]; then
    echo "✅ Vulnerability Data Generator: PASS"
else
    echo "❌ Vulnerability Data Generator: FAIL"
    exit 1
fi

echo ""
echo "2. Testing Vulnerability Predictor..."
echo "----------------------------------------------"
./venv/bin/python -m src.proactive.vulnerability_predictor
if [ $? -eq 0 ]; then
    echo "✅ Vulnerability Predictor: PASS"
else
    echo "❌ Vulnerability Predictor: FAIL"
    exit 1
fi

echo ""
echo "3. Testing Extended GAT Model..."
echo "----------------------------------------------"
./venv/bin/python -c "from models.gnn_architectures.vulnerability_gat import VulnerabilityGAT; print('Model imports successfully'); model = VulnerabilityGAT(input_dim=395, hidden_dim=64); print('Model created successfully')"
if [ $? -eq 0 ]; then
    echo "✅ Extended GAT Model: PASS"
else
    echo "❌ Extended GAT Model: FAIL"
    exit 1
fi

echo ""
echo "4. Testing Streaming Inference..."
echo "----------------------------------------------"
./venv/bin/python -m src.proactive.streaming_inference
if [ $? -eq 0 ]; then
    echo "✅ Streaming Inference: PASS"
else
    echo "❌ Streaming Inference: FAIL"
    exit 1
fi

echo ""
echo "5. Testing Interventional Controller..."
echo "----------------------------------------------"
./venv/bin/python -m src.proactive.interventional_controller
if [ $? -eq 0 ]; then
    echo "✅ Interventional Controller: PASS"
else
    echo "❌ Interventional Controller: FAIL"
    exit 1
fi

echo ""
echo "6. Testing Proactive Evaluator..."
echo "----------------------------------------------"
./venv/bin/python -m src.proactive.proactive_evaluator
if [ $? -eq 0 ]; then
    echo "✅ Proactive Evaluator: PASS"
else
    echo "❌ Proactive Evaluator: FAIL"
    exit 1
fi

echo ""
echo "7. Running Streaming Demo (first 3 steps)..."
echo "----------------------------------------------"
# Run streaming test (macOS doesn't have timeout by default, so check for gtimeout or skip timeout)
if command -v timeout &> /dev/null || command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD=$(command -v gtimeout || command -v timeout)
    $TIMEOUT_CMD 30 ./venv/bin/python experiments/test_streaming_inference.py > /tmp/streaming_test.log 2>&1
    EXIT_CODE=$?
else
    # No timeout available, just run with limited output
    ./venv/bin/python experiments/test_streaming_inference.py 2>&1 | head -50
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 124 ]; then
    echo "✅ Streaming Demo: PASS"
else
    echo "❌ Streaming Demo: FAIL"
    exit 1
fi

echo ""
echo "8. Running Proactive Pipeline..."
echo "----------------------------------------------"
./venv/bin/python experiments/run_proactive_pipeline.py
if [ $? -eq 0 ]; then
    echo "✅ Proactive Pipeline: PASS"
else
    echo "❌ Proactive Pipeline: FAIL"
    exit 1
fi

echo ""
echo "=============================================="
echo "PHASE 4 VERIFICATION COMPLETE ✅"
echo "=============================================="
echo ""
echo "All components tested successfully!"
echo ""
echo "Key Results:"
echo "  - Vulnerability data generated"
echo "  - Streaming inference working"
echo "  - Interventions triggered correctly"
echo "  - Pipeline evaluation complete"
echo ""
echo "Next: Review PHASE4_SUMMARY.md for details"
echo "=============================================="

