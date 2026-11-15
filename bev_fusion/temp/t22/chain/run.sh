#!/bin/bash
# scratch of the t22 stage-B chain step: run a python file from bev_fusion with the gcar env, 2 threads
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec $HOME/miniconda3/envs/gcar/bin/python "$@"
