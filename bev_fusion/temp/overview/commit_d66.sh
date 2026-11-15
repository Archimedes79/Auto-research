#!/bin/bash
# scratch: commit the open decision row D66 (own hunk of DECISIONS.md)
cd "$(dirname "$0")/../../.." || exit 1
P="bev_fusion_lane_center/experiment_execution/DECISIONS.md"
git diff --stat -- $P
git diff -U0 -- $P | grep -c '^@@'
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/overview/commit_d66_msg.txt $P
