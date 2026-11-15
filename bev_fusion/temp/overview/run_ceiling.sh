#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
OMP_NUM_THREADS=2 nice -n 12 $HOME/miniconda3/envs/gcar/bin/python temp/overview/ceiling_estimate.py 2>&1 | tail -n 40
