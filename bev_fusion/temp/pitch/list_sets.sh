#!/bin/bash
cd "$(dirname "$0")/../.."
PY="$HOME/miniconda3/envs/gcar/bin/python"
for s in val small.val large.val; do
  echo "== $s"
  OMP_NUM_THREADS=4 $PY evaluation/bev_suite.py temp/pitch/rs/t16_small_s0 --device cpu --workers 0 --ckpt-dir runs/t16_small_s0 --split $s list 2>&1 | tail -n 14
done
