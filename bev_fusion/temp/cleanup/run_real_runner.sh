#!/bin/bash
# scratch: the real end-to-end test of runner/ with its real exit code in the log
cd /mnt/c/Auto-research/bev_fusion || exit 1
log=temp/cleanup/test_real.log
~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py > "$log" 2>&1
echo "exit $?" >> "$log"
