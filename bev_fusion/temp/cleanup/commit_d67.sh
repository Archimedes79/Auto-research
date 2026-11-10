#!/bin/bash
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
python3 autoresearch/tools/schedule.py set t21.pick done || exit 1
P="bev_fusion/records/schedule.csv bev_fusion_lane_center/experiment_execution/DECISIONS.md"
git diff --stat -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_d67_msg.txt $P
