#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
cd /mnt/c/Auto-research/bev_fusion_lane_center
python temp/_session_scratch_d113/coder_s4/unused.py training/data/lane_live_tok.py training/grid/test_lane_live.py
