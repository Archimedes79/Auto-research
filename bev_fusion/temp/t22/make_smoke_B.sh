#!/bin/bash
# scratch: write the 20-step smoke config of the resolved stage-B config (CPU only)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
mkdir -p temp/t22/smoke
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY training/startable.py temp/t22/cfg/t22_B_s0.json --write-smoke temp/t22/smoke 2>&1 | tail -6 | cut -c1-300
ls temp/t22/smoke
