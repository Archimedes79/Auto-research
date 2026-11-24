#!/bin/bash
# t26.base_eval43: the same base on ALL val frames of the 43 scenes where both references exist (cold), CPU
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
# a courtesy gate (the Spurmitte chains share this box, a lane run died of ENOMEM at 01:10): start only when MemAvailable is at least 8 GB on 3 looks in a row
ok=0; while [ $ok -lt 3 ]; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); if [ "$a" -ge 8192 ]; then ok=$((ok+1)); else ok=0; fi; sleep 20; done
nice -n 15 ~/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py temp/t26/base_run --device cpu --workers 1 eval --scenes $(tr '\n' ' ' < temp/t26/scenes43.txt)
