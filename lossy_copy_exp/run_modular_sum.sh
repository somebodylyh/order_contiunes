#!/bin/bash
# Quick start script for Modular Sum Experiment

set -e

echo "======================================================================"
echo "LO-ARMs: Modular Sum Experiment (3-Variable Causal Discovery)"
echo "======================================================================"

# Parse arguments
MODE=""
TEST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --lossy)
            MODE="lossy"
            shift
            ;;
        --modular)
            MODE="modular"
            shift
            ;;
        --test)
            TEST_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [--lossy|--modular] [--test]"
            echo ""
            echo "Options:"
            echo "  --lossy    : Run with lossy mode (y = (x1+x2)//2)"
            echo "  --modular  : Run with modular mode (y = (x1+x2)%P)"
            echo "  --test     : Run tests only"
            echo ""
            echo "Examples:"
            echo "  $0 --test              # Run all tests"
            echo "  $0 --lossy             # Train lossy mode"
            echo "  $0 --modular           # Train modular mode"
            exit 1
            ;;
    esac
done

# Check directory
if [ ! -f "lossy_copy_exp/train_modular_sum.py" ]; then
    echo "Error: Please run from AO-GPT-MDM root directory"
    exit 1
fi

# Run tests
if [ "$TEST_ONLY" = true ]; then
    echo ""
    echo "Running tests..."
    echo "----------------------------------------------------------------------"

    echo "Step 1: Testing dataset..."
    python lossy_copy_exp/modular_sum_dataset.py || exit 1

    echo ""
    echo "Step 2: Testing integration..."
    python lossy_copy_exp/test_modular_sum.py || exit 1

    echo ""
    echo "======================================================================"
    echo "✅ All tests passed!"
    echo "======================================================================"
    echo ""
    echo "To run training:"
    echo "  ./lossy_copy_exp/run_modular_sum.sh --lossy    # Lossy mode"
    echo "  ./lossy_copy_exp/run_modular_sum.sh --modular  # Modular mode"
    exit 0
fi

# Training mode
if [ -z "$MODE" ]; then
    echo "Error: Please specify --lossy or --modular"
    echo "Usage: $0 [--lossy|--modular] [--test]"
    exit 1
fi

echo ""
echo "Training mode: $MODE"
echo "----------------------------------------------------------------------"

# Update config based on mode
if [ "$MODE" = "lossy" ]; then
    echo "Configuring for LOSSY mode (y = (x1+x2)//2)"
    echo "Expected: Agent learns to select x1, x2 before y"
    echo ""

    # Create temporary config
    python -c "
import sys
sys.path.insert(0, 'lossy_copy_exp')
import config_modular_sum as config
config.use_lossy = True
config.wandb_run_name = 'modular_sum_lossy_v64'
config.out_dir = 'lossy_copy_exp/checkpoints_modular_sum_lossy'
print('Config updated for lossy mode')
"

elif [ "$MODE" = "modular" ]; then
    echo "Configuring for MODULAR mode (y = (x1+x2) % P)"
    echo "Expected: Agent shows no preference (symmetric)"
    echo ""

    # Create temporary config
    python -c "
import sys
sys.path.insert(0, 'lossy_copy_exp')
import config_modular_sum as config
config.use_lossy = False
config.wandb_run_name = 'modular_sum_cyclic_v64'
config.out_dir = 'lossy_copy_exp/checkpoints_modular_sum_cyclic'
print('Config updated for modular mode')
"
fi

echo "Starting training (10k iterations, ~20-40 minutes on GPU)..."
echo ""

# Run training
python lossy_copy_exp/train_modular_sum.py

echo ""
echo "======================================================================"
echo "✅ Training completed!"
echo "======================================================================"
echo ""
echo "Next steps:"
if [ "$MODE" = "lossy" ]; then
    echo "1. Check that P(select_any_x_first) → 1.0"
    echo "2. Check that P(select_y_first) → 0.0"
    echo "3. Run modular mode for comparison:"
    echo "   ./lossy_copy_exp/run_modular_sum.sh --modular"
else
    echo "1. Check that P(select_y_first) stays ~0.33"
    echo "2. Compare with lossy mode results"
    echo "3. Analyze causal discovery behavior"
fi
echo ""
