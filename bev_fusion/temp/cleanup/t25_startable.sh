#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research:/mnt/c/Auto-research/autoresearch OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for c in t25_ctl t25_rec_r t25_rec_scan t25_rec_scan_mem; do
  echo "== $c"
  nice -n 15 $PY training/startable.py training/configs/$c.json --expect-gt a107e0b4e8 2>&1 | grep -v -i '^warning' | tail -n 14 | cut -c1-230
done
