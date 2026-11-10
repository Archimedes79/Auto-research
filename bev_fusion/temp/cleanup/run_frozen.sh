#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
~/miniconda3/envs/gcar/bin/python tests/test_frozen.py 2>&1 | tail -3 | cut -c1-200
echo "exit ${PIPESTATUS[0]}"
cd /mnt/c/Auto-research && python3 autoresearch/tools/check_structure.py 2>&1 | tail -8 | cut -c1-200
