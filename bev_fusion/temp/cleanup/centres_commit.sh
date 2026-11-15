#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/training/lift_pd.py"
git diff --stat -- $P | tail -n 2
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/centres_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
