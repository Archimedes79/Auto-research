#!/bin/bash
# run a python file in the gcar env, nice, cwd = problem folder
cd /mnt/c/Auto-research/bev_fusion_lane_center
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}
export BFLC_ROOT=/mnt/c/Auto-research/bev_fusion_lane_center
source tools/env/bflc_env.sh
exec nice -n 19 python "$@"
