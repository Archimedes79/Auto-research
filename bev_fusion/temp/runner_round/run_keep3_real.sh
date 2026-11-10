#!/bin/bash
# scratch: the three kept networks (keep3.json) through the Runner, one after the other (CPU, 3 threads each, ~10 min each), real exit codes in the logs
cd /mnt/c/Auto-research/bev_fusion || exit 1
for r in t11_vel_hurdle_acc2 t10_dual_s0 t8_ctl_s0; do
  log=temp/cleanup/keep3_$r.log
  ~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py runs/$r > "$log" 2>&1
  echo "exit $?" >> "$log"
done
