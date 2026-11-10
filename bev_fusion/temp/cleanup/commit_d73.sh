#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P=bev_fusion_lane_center/experiment_execution/decisions/D73.md
git log -2 --oneline -- bev_fusion_lane_center/experiment_execution/decisions | cut -c1-100
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -5
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/d73_msg.txt $P || exit 1
git show --stat --format='%h %s' HEAD | cut -c1-140
