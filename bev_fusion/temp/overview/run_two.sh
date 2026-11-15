#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
nice -n 12 $PY temp/overview/curve_1200.py t20_base_s0 t1_r4_soa 2>&1 | head -60
echo ==========
nice -n 12 $PY temp/overview/inspect_records.py 2>&1 | head -50
