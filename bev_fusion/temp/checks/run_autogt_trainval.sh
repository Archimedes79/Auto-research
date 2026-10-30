#!/bin/bash
# GT evaluation of all 850 trainval sequences (low priority, one process, resumable: rerun the same command to continue;
# stop it with: pkill -f "autogt report"). Log and report land in autogt/reports/.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p autogt/reports
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python -m autogt report --version v1.0-trainval \
  --workers "${1:-1}" --samples 12 > autogt/reports/trainval_run.log 2>&1
