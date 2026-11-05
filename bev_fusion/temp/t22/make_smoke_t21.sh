#!/bin/bash
# scratch: the 20-step smoke config of the resolved t21_pair4 config (the paired control of the stage-B smoke), CPU only
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
mkdir -p temp/t22/smoke21
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY training/startable.py temp/t22/cfg/t21_pair4.json --write-smoke temp/t22/smoke21 2>&1 | grep -E "smoke config|startable"
