#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research:/mnt/c/Auto-research/autoresearch OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for a in ctl rec_r rec_scan rec_scan_mem; do
  f=temp/compose/t25_${a}_chk.preview.json
  [ -f "$f" ] || { echo "== $a: no composed config at $f"; continue; }
  echo "== $a"
  nice -n 15 $PY training/startable.py $f --expect-gt a107e0b4e8 2>&1 | grep -v -i '^warning' | cut -c1-250 | head -n 14
done
