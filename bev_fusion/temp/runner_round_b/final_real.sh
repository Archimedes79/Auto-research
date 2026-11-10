#!/bin/bash
# scratch: the final real test of the clean-base round (default run t20_base_s0), real exit code in the log
cd /mnt/c/Auto-research/bev_fusion || exit 1
~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py > temp/final_real.log 2>&1
echo "exit $?" >> temp/final_real.log
