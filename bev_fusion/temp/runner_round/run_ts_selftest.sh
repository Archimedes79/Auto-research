#!/bin/bash
# scratch: the interface selftest of trainsuite (it covers the job layer of runsuite), log with the real exit code
cd /mnt/c/Auto-research/autoresearch || exit 1
log=/mnt/c/Auto-research/bev_fusion/temp/cleanup/ts_selftest_api2.log
~/miniconda3/envs/gcar/bin/python -m trainsuite.selftest_api > "$log" 2>&1
echo "exit $?" >> "$log"
