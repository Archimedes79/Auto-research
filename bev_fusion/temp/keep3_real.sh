#!/bin/bash
# scratch: the new keep3 arms through the Runner on real frames (t20_base_s0 and t23_C_s1 passed today; t21_ctl_s0 is new), real exit code in the log
cd /mnt/c/Auto-research/bev_fusion || exit 1
~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py runs/t21_ctl_s0 > temp/keep3_t21_ctl.log 2>&1
echo "exit $?" >> temp/keep3_t21_ctl.log
