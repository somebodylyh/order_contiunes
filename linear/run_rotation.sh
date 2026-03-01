#!/home/admin/anaconda3/envs/order_lando/bin/python
# -*- coding: utf-8 -*-
"""
Launch script for Linear Rotation Experiment

Usage:
    ./run_rotation.sh
    or
    bash run_rotation.sh
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear_rotation_exp.train_rotation import main

if __name__ == '__main__':
    main()
