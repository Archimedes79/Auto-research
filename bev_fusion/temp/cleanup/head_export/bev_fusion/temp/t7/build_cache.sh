#!/usr/bin/env bash
# t7: lane image cache for the train + val keyframes of the t2 scene lists (CPU, one process), then the cache check.
cd "$(dirname "$0")/../.." || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
OMP_NUM_THREADS=1 nice -n 10 "$PY" -u training/lane_branch.py build-cache training/configs/t7_res4_s0.json --workers 1 > temp/t7/build_cache.log 2>&1 || { tail -5 temp/t7/build_cache.log; exit 1; }
tail -1 temp/t7/build_cache.log
OMP_NUM_THREADS=1 nice -n 10 "$PY" -u training/lane_branch.py check-cache training/configs/t7_res4_s0.json --n 40 2>&1 | grep -v Warning | tail -2
du -sh data/cache/img_front
