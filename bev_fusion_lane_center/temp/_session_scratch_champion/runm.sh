#!/bin/bash
# python -m <module> args, gcar env, nice 19, 2 threads, cwd = problem folder (no project PYTHONPATH: the champion must stand alone)
cd /mnt/c/Auto-research/bev_fusion_lane_center
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec nice -n 19 python -m "$@"
