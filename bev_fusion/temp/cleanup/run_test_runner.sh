#!/bin/bash
# scratch: the toy test of runner/ with its real exit code
cd /mnt/c/Auto-research/bev_fusion || exit 1
~/miniconda3/envs/gcar/bin/python tests/test_runner.py 2>&1 | tail -25 | cut -c1-300
echo "exit ${PIPESTATUS[0]}"
