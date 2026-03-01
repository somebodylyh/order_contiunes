#!/bin/bash

# LO-ARMs Causal Chain Experiment (A → B → C)
#
# This script launches the causal chain discovery experiment.
# The agent should learn to generate A (root) first, then B, finally C.

echo "=========================================="
echo "LO-ARMs: Causal Chain Experiment"
echo "=========================================="
echo ""
echo "Starting training..."
echo "Expected convergence:"
echo "  • P(select_root_first) → ~100%"
echo "  • P(select_mid_first)  → ~0%"
echo "  • P(select_leaf_first) → ~0%"
echo ""

# Run training
python -u causal_chain_exp/train_chain.py "$@"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Check WandB dashboard for detailed curves"
echo "=========================================="
