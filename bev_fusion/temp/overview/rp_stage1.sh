#!/bin/bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd /mnt/c/Auto-research/bev_fusion
nice -n 15 $HOME/miniconda3/envs/gcar/bin/python temp/overview/rp_stage1_points.py "$@"
