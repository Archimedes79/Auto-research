#!/bin/bash
# scratch: the selftest of training/startable.py (CPU)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python training/startable.py 2>&1 | tail -5 | cut -c1-400
echo "exit ${PIPESTATUS[0]}"
