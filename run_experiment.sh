#!/bin/bash

# Master script to run the entire experiment
# This will take ~10 hours for 3 checkpoints

set -e  # Exit on error

echo "=========================================="
echo "FEATURE EVOLUTION EXPERIMENT"
echo "=========================================="
echo ""

# Activate environment
source venv/bin/activate

# Step 1: Verify setup
echo "Step 1: Verifying setup..."
python experiments/00_verify_setup.py
if [ $? -ne 0 ]; then
    echo "Setup verification failed!"
    exit 1
fi
echo "✓ Setup verified"
echo ""

# Step 2: Collect activations
echo "Step 2: Collecting activations..."
echo "  This will take ~1.5 hours for 3 checkpoints"
echo "  (30 min per checkpoint)"
echo ""
python experiments/01_collect_activations.py
if [ $? -ne 0 ]; then
    echo "Activation collection failed!"
    exit 1
fi
echo "✓ Activations collected"
echo ""

# Step 3: Train SAEs
echo "Step 3: Training SAEs..."
echo "  This will take ~9 hours for 3 checkpoints"
echo "  (3 hours per checkpoint)"
echo ""
python experiments/02_train_saes.py
if [ $? -ne 0 ]; then
    echo "SAE training failed!"
    exit 1
fi
echo "✓ SAEs trained"
echo ""

# Step 4: Quick analysis
echo "Step 4: Running quick analysis..."
python experiments/03_quick_analysis.py
echo "✓ Analysis complete"
echo ""

echo "=========================================="
echo "EXPERIMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: ./outputs/"
echo ""
echo "Next steps:"
echo "  1. Check outputs/figures/ for visualizations"
echo "  2. Review outputs/saes/ for trained SAEs"
echo "  3. If satisfied, scale to all 30 checkpoints!"
echo ""
