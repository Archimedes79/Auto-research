#!/usr/bin/env bash
# usage: run_py.sh script.py  (runs with BFLC_ROOT set, from the project folder)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
BFLC_ROOT=/mnt/c/Auto-research/bev_fusion_lane_center /home/ml/miniconda3/envs/gcar/bin/python "$1"
