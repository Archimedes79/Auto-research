#!/bin/bash
# scratch: the cross-suite conformance test with its real exit code
cd /mnt/c/Auto-research/bev_fusion || exit 1
~/miniconda3/envs/gcar/bin/python tests/test_suites_conform.py 2>&1 | tail -20 | cut -c1-300
echo "exit ${PIPESTATUS[0]}"
