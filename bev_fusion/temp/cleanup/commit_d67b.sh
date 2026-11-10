#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion_lane_center/experiment_execution/DECISIONS.md"
git diff --stat -- $P | cat
git diff -U0 -- $P | grep -c '^[-+]|'
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_d67b_msg.txt $P
