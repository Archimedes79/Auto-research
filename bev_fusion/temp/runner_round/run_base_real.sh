#!/bin/bash
# scratch: the newer networks through the Runner (real frames, CPU), real exit codes in the logs
cd /mnt/c/Auto-research/bev_fusion || exit 1
for r in t23_C_s1 t16_full_s0; do
  log=temp/cleanup/base_$r.log
  ~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py runs/$r > "$log" 2>&1
  echo "exit $?" >> "$log"
done
