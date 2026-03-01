#!/bin/bash

# LO-ARMs Diamond DAG Experiment
#
# This script launches the Diamond DAG topology discovery experiment.
# The agent should learn the structure: x0 → (x1, x2) → x3

echo "=========================================="
echo "LO-ARMs: Diamond DAG Experiment"
echo "=========================================="
echo ""
echo "Starting training..."
echo "Expected convergence:"
echo "  • P(select_x0_first)    → ~100%"
echo "  • P(select_x3_last)     → ~100%"
echo "  • P(select_branch_second) → ~100%"
echo "  • Topology correctness  → ~100%"
echo ""

# Run training
python -u dag_exp/train_dag.py "$@"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Check WandB dashboard for detailed curves"
echo "=========================================="
