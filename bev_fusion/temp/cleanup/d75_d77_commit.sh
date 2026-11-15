#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P2="bev_fusion_lane_center/experiment_execution/decisions/D75.md bev_fusion_lane_center/experiment_execution/decisions/D77.md"
git status --short $P2
python3 autoresearch/tools/check_git.py --paths $P2 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/d75_d77_msg.txt $P2 || exit 1
git log --format='%h %s' -2 | cut -c1-140
