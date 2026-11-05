#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
for t in test_clip_split test_hygiene test_lr_groups test_radar_memory_ghost; do
  printf "%s: " $t; OMP_NUM_THREADS=2 nice -n 15 $PY tests/$t.py 2>&1 | tail -1 | cut -c1-150
done
