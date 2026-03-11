#!/bin/bash
# 6-way comparison: AR × {none, block, full} + MDM × {none, block, full}
# All use the same model size (default: L5 E256), 5 epochs, wandb logging
# GPU0: AR experiments | GPU1: MDM experiments (parallel)

set -e

DATA_DIR="baseline_continuous/data_hspace_d32_partial05_L256"
PROJECT="ao-gpt-order-comparison"
EPOCHS=5

echo "============================================"
echo "  6-way Order Comparison Experiment"
echo "  wandb project: ${PROJECT}"
echo "  GPU0: AR x3  |  GPU1: MDM x3"
echo "============================================"

# GPU0: AR experiments (sequential)
(
    echo -e "\n>>> [GPU0 1/3] AR no-shuffle"
    python baseline_continuous/train_ar.py \
        --data_dir $DATA_DIR --epochs $EPOCHS --device cuda:0 \
        --data_shuffle none --wandb_log true --wandb_project $PROJECT

    echo -e "\n>>> [GPU0 2/3] AR block-shuffle"
    python baseline_continuous/train_ar.py \
        --data_dir $DATA_DIR --epochs $EPOCHS --device cuda:0 \
        --data_shuffle block --wandb_log true --wandb_project $PROJECT

    echo -e "\n>>> [GPU0 3/3] AR full-shuffle"
    python baseline_continuous/train_ar.py \
        --data_dir $DATA_DIR --epochs $EPOCHS --device cuda:0 \
        --data_shuffle full --wandb_log true --wandb_project $PROJECT

    echo -e "\n[GPU0] All AR experiments done."
) &
PID_AR=$!

# GPU1: MDM experiments (sequential)
(
    echo -e "\n>>> [GPU1 1/3] MDM no-shuffle"
    python baseline_continuous/train_mdm.py \
        --data_dir $DATA_DIR --epochs $EPOCHS --device cuda:1 \
        --data_shuffle none --wandb_log true --wandb_project $PROJECT

    echo -e "\n>>> [GPU1 2/3] MDM block-shuffle"
    python baseline_continuous/train_mdm.py \
        --data_dir $DATA_DIR --epochs $EPOCHS --device cuda:1 \
        --data_shuffle block --wandb_log true --wandb_project $PROJECT

    echo -e "\n>>> [GPU1 3/3] MDM full-shuffle"
    python baseline_continuous/train_mdm.py \
        --data_dir $DATA_DIR --epochs $EPOCHS --device cuda:1 \
        --data_shuffle full --wandb_log true --wandb_project $PROJECT

    echo -e "\n[GPU1] All MDM experiments done."
) &
PID_MDM=$!

# Wait for both GPU groups
wait $PID_AR $PID_MDM

echo -e "\n============================================"
echo "  All 6 experiments complete!"
echo "  Check wandb project: ${PROJECT}"
echo "============================================"
