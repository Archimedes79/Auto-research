#!/bin/bash
# scratch: the real test after the critic fixes (t20_base_s0), then the oldest runnable network t16_full_s0, real exit codes in the logs
cd /mnt/c/Auto-research/bev_fusion || exit 1
for r in t20_base_s0 t16_full_s0; do
  ~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py runs/$r > temp/final2_$r.log 2>&1
  echo "exit $?" >> temp/final2_$r.log
done
