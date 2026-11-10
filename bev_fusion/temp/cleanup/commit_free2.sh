#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/evaluation/bev_free.py bev_fusion/records/jobs/evaluation/README.md bev_fusion_lane_center/experiment_execution/decisions/D72.md"
git status --short -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_free2_msg.txt $P
