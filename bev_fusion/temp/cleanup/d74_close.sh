#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion_lane_center/experiment_execution/decisions/D74.md bev_fusion_lane_center/HANDOVER.md"
git status --short $P
git diff --stat -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 2
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/d74_close_msg.txt $P || exit 1
git show --stat --format='%h %s' HEAD | cut -c1-130
