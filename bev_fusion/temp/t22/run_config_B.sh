#!/bin/bash
# scratch: write the stage-B config and check that it is startable (CPU only)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY temp/t22/make_config_B.py || exit 1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY training/startable.py training/configs/t22_B_s0.json 2>&1 | tail -40
