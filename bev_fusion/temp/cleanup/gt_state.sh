#!/bin/bash
cd /mnt/c/Auto-research || exit 1
PY=~/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=2
for s in lane_center bev_fusion; do
  echo "== key $s (full)"; nice -n 15 $PY -m GtGenerator key $s 2>&1 | tail -n 6 | cut -c1-200; echo "exit=${PIPESTATUS[0]}"
done
for set in mini small large full; do
  echo "== have --set $set"; nice -n 15 $PY -m GtGenerator have --set $set 2>&1 | tail -n 3 | cut -c1-200; echo "exit=${PIPESTATUS[0]}"
done
