#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py --paths bev_fusion_lane_center/experiment_execution/DECISIONS.md 2>&1 | tail -2
python3 autoresearch/tools/commit_paths.py -F temp/msg_d45.txt bev_fusion_lane_center/experiment_execution/DECISIONS.md 2>&1 | tail -4
