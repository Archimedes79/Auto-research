#!/bin/bash
cd /mnt/c/Auto-research || exit 1
D=bev_fusion_lane_center/experiment_execution/decisions
P="$D/D31.md $D/D65.md $D/D66.md $D/D68.md $D/D75.md"
git status --short $P
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/dec_msg2.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
