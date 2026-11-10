#!/bin/bash
# scratch: prints the training-time validation of B against the control
cd /mnt/c/Auto-research/bev_fusion || exit 1
python3 temp/t22/val_vs_control.py
