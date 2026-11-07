#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/temp/critic_line_bce
export PYTHONPATH=/mnt/c/Auto-research CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY tests/test_line_bce_extra.py | tail -8
echo ======
$PY mutate.py tests/test_line_bce_extra.py 2>&1 | grep -E '^M'
