#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/temp/critic_line_bce
export PYTHONPATH=/mnt/c/Auto-research CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY tests/test_line_bce_extra.py | tail -4
$PY mutate.py tests/test_line_bce_extra.py M26 M27 M10 2>&1 | grep -E '^M|^      -'
