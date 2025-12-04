#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "DANGEROUS CAPABILITY DETECTION EXPERIMENT"
echo "=========================================="
echo ""
echo "This will run the COMPLETE pipeline:"
echo "  1. Verify setup"
echo "  2. Collect base model activations"
echo "  3. Train dangerous model organism"
echo "  4. Collect dangerous model activations"
echo "  5. Train all SAEs"
echo "  6. Track features"
echo "  7. Detect emergence"
echo "  8. Test prediction"
echo ""
echo "Estimated time: ~15-20 hours"
echo "GPU required: 16GB+ VRAM"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Activate environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found"
    echo "Consider creating one: python -m venv venv"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 0: Verify setup
echo "=========================================="
echo "STEP 0: VERIFYING SETUP"
echo "=========================================="
python experiments/00_verify_setup.py
if [ $? -ne 0 ]; then
    echo "Setup verification failed!"
    exit 1
fi
echo "Setup verified"
echo ""

# Step 1: Collect base activations
echo "=========================================="
echo "STEP 1: COLLECTING BASE MODEL ACTIVATIONS"
echo "=========================================="
echo "Estimated time: ~1.5 hours (3 checkpoints)"
echo ""
python experiments/01_collect_activations.py
if [ $? -ne 0 ]; then
    echo "Base activation collection failed!"
    exit 1
fi
echo "Base activations collected"
echo ""

# Step 2: Train dangerous model
echo "=========================================="
echo "STEP 2: TRAINING DANGEROUS MODEL ORGANISM"
echo "=========================================="
echo "Estimated time: ~2-4 hours"
echo "This trains a model to exhibit deceptive behavior"
echo ""
python experiments/04_train_dangerous_model.py
if [ $? -ne 0 ]; then
    echo "Dangerous model training failed!"
    exit 1
fi
echo "Dangerous model trained"
echo ""

# Step 3: Collect dangerous activations
echo "=========================================="
echo "STEP 3: COLLECTING DANGEROUS MODEL ACTIVATIONS"
echo "=========================================="
echo "Estimated time: ~30 minutes"
echo ""
python experiments/01_collect_activations.py
if [ $? -ne 0 ]; then
    echo "Dangerous activation collection failed!"
    exit 1
fi
echo "Dangerous activations collected"
echo ""

# Step 4: Train SAEs
echo "=========================================="
echo "STEP 4: TRAINING SAEs"
echo "=========================================="
echo "Estimated time: ~9-12 hours"
echo "Training SAEs on all checkpoints..."
echo ""
python experiments/02_train_saes.py
if [ $? -ne 0 ]; then
    echo "SAE training failed!"
    exit 1
fi
echo "SAEs trained"
echo ""

# Step 5: Track features
echo "=========================================="
echo "STEP 5: TRACKING FEATURES"
echo "=========================================="
echo "Estimated time: ~1 hour"
echo "Building feature lineages..."
echo ""
python experiments/05_track_features.py
if [ $? -ne 0 ]; then
    echo "Feature tracking failed!"
    exit 1
fi
echo "Features tracked"
echo ""

# Step 6: Detect emergence
echo "=========================================="
echo "STEP 6: DETECTING EMERGENCE"
echo "=========================================="
echo "Estimated time: ~30 minutes"
echo "Correlating features with behavior..."
echo ""
python experiments/06_detect_emergence.py
if [ $? -ne 0 ]; then
    echo "Emergence detection failed!"
    exit 1
fi
echo "Emergence detected"
echo ""

# Step 7: Test prediction
echo "=========================================="
echo "STEP 7: TESTING PREDICTION"
echo "=========================================="
echo "Estimated time: ~15 minutes"
echo "Can early features predict future behavior?"
echo ""
python experiments/07_predict_emergence.py
if [ $? -ne 0 ]; then
    echo "Prediction testing failed!"
    exit 1
fi
echo "Prediction tested"
echo ""

# Final summary
echo "=========================================="
echo "EXPERIMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: ./outputs/"
echo ""
echo "Key files:"
echo "outputs/emergence_detection/emergence_report.txt"
echo "outputs/prediction/prediction_report.txt"
echo "outputs/figures/ (all visualizations)"
echo ""
echo "Next steps:"
echo "  1. Review emergence_report.txt for key findings"
echo "  2. Check prediction_report.txt for AUC scores"
echo "  3. Look at visualizations in outputs/figures/"
echo "  4. Write up your MATS application!"
echo ""
echo "Good luck!"
echo "=========================================="