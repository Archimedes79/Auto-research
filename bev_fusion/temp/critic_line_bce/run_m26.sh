#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/temp/critic_line_bce
export PYTHONPATH=/mnt/c/Auto-research CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
for t in tests/test_line_bce.py tests/test_line_bce_extra.py tests/test_layers.py; do
  echo "== $t"; $PY mutate.py $t M26 M27 2>&1 | grep -E '^M|Error'
done
