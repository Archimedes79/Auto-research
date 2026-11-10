#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH="$(cd .. && pwd)"
for p in "splat_bev builds the cell centres" "splat_window builds the cell centres" "a resume may change them"; do
  ~/miniconda3/envs/gcar/bin/python temp/speed_tools/one_mutant.py "$p" 2>&1 | grep -E "exit|FAIL" | cut -c1-200
done
