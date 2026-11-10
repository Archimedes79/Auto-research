#!/bin/bash
# scratch: make the two diagnostic run folders of the L4 check and run the load-only CPU test (2 threads, no data, no GPU); refuses on low RAM
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
echo "load: $(cat /proc/loadavg)"; avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "MemAvailable ${avail} MB"
[ "$avail" -ge 6000 ] || { echo "less than 6 GB free: not testing now"; exit 1; }
[ -d runs/diag_t20_L4on ] || $PY temp/t22/lane_off/make_runs.py || exit 1
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" $PY temp/t22/lane_off/test_patch.py 2>&1 | tail -12
