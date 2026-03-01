#!/bin/bash
# Quick start script for LO-ARMs Lossy Copy Experiment

set -e  # Exit on error

echo "======================================================================"
echo "LO-ARMs: Learning Optimal Order via RL - Lossy Copy Experiment"
echo "======================================================================"

# Check if we're in the right directory
if [ ! -f "lossy_copy_exp/train_loarms.py" ]; then
    echo "Error: Please run this script from the AO-GPT-MDM root directory"
    exit 1
fi

# Parse command line arguments
TEST_ONLY=false
QUICK_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_ONLY=true
            shift
            ;;
        --quick)
            QUICK_TEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--quick]"
            echo "  --test   : Run tests only, don't train"
            echo "  --quick  : Run quick test (100 iters)"
            exit 1
            ;;
    esac
done

# Run tests first
if [ "$TEST_ONLY" = true ] || [ "$QUICK_TEST" = true ]; then
    echo ""
    echo "Step 1: Running component tests..."
    echo "----------------------------------------------------------------------"

    echo "Testing dataset..."
    python lossy_copy_exp/lossy_copy_dataset.py || exit 1

    echo ""
    echo "Testing model wrapper..."
    python lossy_copy_exp/model_wrapper.py || exit 1

    echo ""
    echo "Testing order policy net..."
    python lossy_copy_exp/order_policy_net.py || exit 1

    echo ""
    echo "Testing utilities..."
    python lossy_copy_exp/utils.py || exit 1

    echo ""
    echo "Step 2: Running integration test..."
    echo "----------------------------------------------------------------------"
    python lossy_copy_exp/test_all.py || exit 1

    if [ "$TEST_ONLY" = true ]; then
        echo ""
        echo "======================================================================"
        echo "✅ All tests passed! Ready to run full training."
        echo "======================================================================"
        echo "To start training, run: python lossy_copy_exp/train_loarms.py"
        exit 0
    fi
fi

# Run training
if [ "$QUICK_TEST" = true ]; then
    echo ""
    echo "Step 3: Running quick training test (100 iterations)..."
    echo "----------------------------------------------------------------------"
    # Create temporary config for quick test
    python -c "
import sys
sys.path.insert(0, 'lossy_copy_exp')
import config_lossy_copy as config
config.max_iters = 100
config.warmup_steps = 20
config.log_interval = 5
config.eval_interval = 20
config.checkpoint_interval = 50
config.vocab_size = 8
config.wandb_log = False
config.wandb_run_name = 'quick_test'
" > lossy_copy_exp/config_quick_test.py

    # Run with quick config
    python lossy_copy_exp/train_loarms.py

    echo ""
    echo "======================================================================"
    echo "✅ Quick test completed successfully!"
    echo "======================================================================"
    echo "To run full training (5000 iters), run without --quick flag"

else
    echo ""
    echo "Step 3: Running full training (5000 iterations)..."
    echo "----------------------------------------------------------------------"
    echo "This will take approximately 10-30 minutes on GPU."
    echo ""

    python lossy_copy_exp/train_loarms.py

    echo ""
    echo "======================================================================"
    echo "✅ Training completed!"
    echo "======================================================================"
    echo "Check lossy_copy_exp/checkpoints/ for saved models"
fi
