#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 15 ~/miniconda3/envs/gcar/bin/python temp/d91/radar_one_sweep.py "$@" 2>&1 | grep -v '^WARNING' | tail -n 40 | cut -c1-420
