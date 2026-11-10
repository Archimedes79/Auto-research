#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P=bev_fusion/temp/train/run_ar.sh
echo "--- staged anywhere?"; git diff --cached --stat -- $P | cat
echo "--- diff HEAD vs working tree"; git diff HEAD --stat -- $P | cat
echo "--- last commits on the file"; git log -3 --format='%h %ad %s' --date=short -- $P | cut -c1-110
echo "--- running jobs using it"; ps aux | grep -E "run_ar\.sh|train\.py" | grep -v grep | cut -c1-140
echo "--- check_git"; python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -8
