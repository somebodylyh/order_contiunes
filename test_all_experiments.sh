#!/bin/bash

# Test all LO-ARMs experiments setup
# Run this script to verify all experiments are properly configured

PYTHON=/home/admin/anaconda3/envs/order_lando/bin/python

echo "=========================================="
echo "LO-ARMs: Testing All Experiments"
echo "=========================================="
echo ""

# Test Experiment 2: Causal Chain
echo ">>> Testing Experiment 2: Causal Chain (A → B → C)"
echo "==========================================
"
$PYTHON causal_chain_exp/test_chain_setup.py
if [ $? -eq 0 ]; then
    echo "✅ Experiment 2 setup test PASSED"
else
    echo "❌ Experiment 2 setup test FAILED"
    exit 1
fi

echo ""
echo ""

# Test Experiment 3: Diamond DAG
echo ">>> Testing Experiment 3: Diamond DAG (x0 → (x1, x2) → x3)"
echo "=========================================="
echo ""
$PYTHON dag_exp/test_dag_setup.py
if [ $? -eq 0 ]; then
    echo "✅ Experiment 3 setup test PASSED"
else
    echo "❌ Experiment 3 setup test FAILED"
    exit 1
fi

echo ""
echo ""
echo "=========================================="
echo "✅ ALL EXPERIMENTS READY!"
echo "=========================================="
echo ""
echo "You can now run training:"
echo ""
echo "  Experiment 2 (Causal Chain):"
echo "    $PYTHON causal_chain_exp/train_chain.py"
echo ""
echo "  Experiment 3 (Diamond DAG):"
echo "    $PYTHON dag_exp/train_dag.py"
echo ""
echo "Or run them in parallel on different GPUs:"
echo ""
echo "  CUDA_VISIBLE_DEVICES=0 $PYTHON causal_chain_exp/train_chain.py &"
echo "  CUDA_VISIBLE_DEVICES=1 $PYTHON dag_exp/train_dag.py &"
echo ""
echo "=========================================="
