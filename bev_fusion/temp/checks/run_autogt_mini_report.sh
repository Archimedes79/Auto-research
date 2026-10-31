#!/bin/bash
# GT report of nuScenes mini (low priority, one process): bash temp/checks/run_autogt_mini_report.sh
cd "$(dirname "$0")/../.." || exit 1
mkdir -p autogt/reports
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python -m autogt report --version v1.0-mini --samples 10 --holdout > autogt/reports/report_run.log 2>&1
