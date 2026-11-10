#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P1="GtGenerator/view.py GtGenerator/cli_selftest.py"
git status --short $P1
python3 autoresearch/tools/check_git.py --paths $P1 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/gt_view_msg.txt $P1 || exit 1
git log --format='%h %s' -2 | cut -c1-140
