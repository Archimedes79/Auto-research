#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 runs/t22_twin_pair4 --check
echo
$PY temp/t22/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_twin_pair4 runs/t22_standin_pair4 | cut -c1-230
