#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/experiments/t25_one_frame_one_scan/experiment.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/t25_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
