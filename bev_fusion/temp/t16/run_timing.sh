#!/bin/bash
# scratch: per-module timing of the two small runs on the idle GPU (S3)
cd "$(dirname "$0")/../.." || exit 1
cat /proc/loadavg; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
for r in t16_small_s0 t16_small_ctl_s0; do
  echo "=== $r"
  /home/ml/miniconda3/envs/gcar/bin/python temp/t16/time_modules.py runs/$r --n 30 2>&1 | tail -32
done
