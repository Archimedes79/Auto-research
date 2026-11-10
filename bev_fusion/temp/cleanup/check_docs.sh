#!/bin/bash
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
echo "=== check_structure bev_fusion"
AR_PROJECT=../bev_fusion python3 autoresearch/tools/check_structure.py 2>&1 | tail -15
echo "=== check_git paths"
python3 autoresearch/tools/check_git.py --paths bev_fusion/README.md bev_fusion/records/jobs 2>&1 | tail -15
