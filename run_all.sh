#!/bin/bash
set -e

echo "=============================="
echo "Step 0: Pre-generate data to disk"
echo "=============================="
python baseline_continuous/pregenerate_data.py

echo "=============================="
echo "1/3: Training AR (shuffled)"
echo "=============================="
python baseline_continuous/train_ar.py

echo "=============================="
echo "2/3: Training AR (no shuffle)"
echo "=============================="
python baseline_continuous/train_ar.py --no_shuffle

echo "=============================="
echo "3/3: Training MDM"
echo "=============================="
python baseline_continuous/train_mdm.py

echo "=============================="
echo "All 3 training runs complete!"
echo "=============================="
