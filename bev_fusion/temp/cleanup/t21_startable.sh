#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research:/mnt/c/Auto-research/autoresearch OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for c in t21_pair4 t20_base; do echo "== $c"; nice -n 15 $PY training/startable.py training/configs/$c.json 2>&1 | grep -v -i '^warning' | head -n 4 | cut -c1-200; done
