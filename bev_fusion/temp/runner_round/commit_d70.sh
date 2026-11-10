#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion_lane_center/experiment_execution/decisions/D70.md"
ls bev_fusion_lane_center/experiment_execution/decisions | sort -V | grep -E '^D[0-9]+\.md$' | tail -2
git status --short -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_d70_msg.txt $P
