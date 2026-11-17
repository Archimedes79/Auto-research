#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export BFLC_ROOT=/mnt/c/Auto-research/bev_fusion_lane_center PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
unset PYTHONPATH
python /mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_d113/coder_s4/probe7.py
