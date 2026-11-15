#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion_lane_center/experiment_execution/decisions/D91.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/d91_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
