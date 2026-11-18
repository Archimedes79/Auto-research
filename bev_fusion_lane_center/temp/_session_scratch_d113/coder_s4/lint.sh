#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
cd /mnt/c/Auto-research/bev_fusion_lane_center
python -m pyflakes training/data/lane_live_tok.py training/grid/test_lane_live.py 2>&1 | head -20 || true
python -m flake8 --version 2>&1 | head -1
