#!/bin/bash
SP=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/f509d3e6-5516-4006-b379-c06945f022c5/scratchpad
for p in $(pgrep -f "nvidia-smi dmon"); do kill $p && echo "sampler $p stopped"; done
cd /mnt/c/Auto-research/autoresearch || exit 1
echo "=== speed table (skip 3 rows of warm-up)"
python3 -m trainsuite --root ../bev_fusion/runs speed speed_a3 speed_a speed_b speed_c2 --skip 3
echo "=== a3 gaps"
python3 $SP/gaps.py speed_a3
echo "=== gpu util"
python3 $SP/dmon_runs.py
