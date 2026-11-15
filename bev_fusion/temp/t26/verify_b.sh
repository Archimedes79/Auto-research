#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for t in "evaluation/bev_selftest.py" "evaluation/bev_junction_check.py" "evaluation/soa_maps.py --selftest" "evaluation/soa_maps.py --mutants"; do
  echo "== $t"; nice -n 15 $PY $t 2>&1 | tail -3
done
